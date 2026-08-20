
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import math
from collections import defaultdict

from typing import List, Tuple, Deque, Dict, Any, Optional, Set
from enum import Enum
from collections import deque
import numpy as np
from benchmarks.lora.deployment.slurm.launcher_simulator_from_scratch import get_gpu_memory_availability
from benchmarks.lora.placement_algorithm.interface_dynamic_routing_with_probabilities import \
    PlacementAlgorithmInterfaceDynamicRoutingWithProbabilities
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_3 import GPU_MEMORY_AVAILABILITY


OPERATING_POINTS: Dict[str, Dict[int, float]] = {
    "llama-3.1-8b-instruct": {
        8: 7917.918006140972 + 7145.683388373702,
        16: 7738.154029910513 + 6993.366196018666,
        32: 7615.739197738983 + 6880.626177082546,
    },
    "qwen-2.5-7b-instruct": {
        8: 8297.220534804685 + 7336.147497215853,
        16: 8246.72039237496 + 7290.382204101727,
        32: 8245.01841951036 + 7291.0424388114725,
    }
}


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Server:
    id: str


@dataclass
class Adapter:
    id: str
    rank: int


@dataclass
class PlacementResult:
    # For each adapter: list of (server_id, phi) where sum(phi)=1
    routing_table: Dict[str, List[Tuple[str, float]]]

    # For each server: adapter_id -> phi
    server_adapter_phi: Dict[str, Dict[str, float]]


# -----------------------------
# Step 2 helper: stable rounding budgets so sum == #servers
# -----------------------------

def hamilton_apportionment(weights: Dict[int, float], total: int) -> Dict[int, int]:
    """
    Convert nonnegative real weights to integer budgets that sum exactly to 'total'
    using Hamilton (largest remainder) method.

    weights: rank -> weight (e.g., rankUtil / targetUtil)
    """
    # Filter to positive weights; ranks with zero weight get zero budget.
    positive = {r: w for r, w in weights.items() if w > 0}
    if total <= 0 or not positive:
        return {r: 0 for r in weights.keys()}

    s = sum(positive.values())
    if s <= 0:
        return {r: 0 for r in weights.keys()}

    quotas = {r: (w / s) * total for r, w in positive.items()}
    floors = {r: int(math.floor(q)) for r, q in quotas.items()}
    remaining = total - sum(floors.values())

    # Distribute remaining by largest fractional remainders
    remainders = sorted(((quotas[r] - floors[r], r) for r in positive.keys()), reverse=True)
    for i in range(remaining):
        _, r = remainders[i % len(remainders)]
        floors[r] += 1

    # Ensure all ranks exist in output
    out = {r: 0 for r in weights.keys()}
    out.update(floors)
    return out


# -----------------------------
# Step 3: FRACTIONALBINPACKING
# -----------------------------

def fractional_bin_packing(
    adapters: List[Adapter],
    servers: List[Server],
    capacity_tps: float,
    assignment: Dict[str, Dict[str, float]],
    server_load_tps: Dict[str, float],
    server_max_rank: Dict[str, int],
    demand_tps: Dict[str, float],
) -> List[Adapter]:
    """
    Pack adapters onto 'servers' fractionally, each server has capacity_tps (soft target).
    Returns adapters that could not be fully assigned (i.e., need more capacity elsewhere),
    with their remaining fraction implicitly being (1 - sum(phi)).
    """
    if not servers or capacity_tps <= 0:
        # Nothing can be packed here
        return adapters[:]

    # Greedy: larger demands first
    adapters_sorted = sorted(adapters, key=lambda a: demand_tps[a.id], reverse=True)

    leftovers: List[Adapter] = []

    for ad in adapters_sorted:
        total = float(demand_tps[ad.id])
        if total <= 0:
            # Still give it a routing entry: arbitrarily pin to least loaded server with phi=1
            s = min(servers, key=lambda sv: server_load_tps[sv.id])
            assignment[ad.id][s.id] = assignment[ad.id].get(s.id, 0.0) + 1.0
            continue

        remaining_tps = total * (1.0 - sum(assignment[ad.id].values()))

        # Fill across servers with remaining room
        # (soft capacity: we try not to exceed capacity_tps)
        for sv in sorted(servers, key=lambda s: server_load_tps[s.id]):
            if remaining_tps <= 1e-12:
                break
            room = capacity_tps - server_load_tps[sv.id]
            if room <= 1e-12:
                continue

            take = min(room, remaining_tps)
            phi = take / total

            assignment[ad.id][sv.id] = assignment[ad.id].get(sv.id, 0.0) + phi
            server_load_tps[sv.id] += take
            server_max_rank[sv.id] = max(server_max_rank[sv.id], ad.rank)

            remaining_tps -= take

        # If still not fully assigned, mark leftover for Step 4
        if (1.0 - sum(assignment[ad.id].values())) > 1e-9:
            leftovers.append(ad)

    return leftovers


# -----------------------------
# Step 4: ALLOCATEHIGHESTMAXRANK
# -----------------------------

def allocate_highest_max_rank(
    adapter: Adapter,
    servers: List[Server],
    assignment: Dict[str, Dict[str, float]],
    server_load_tps: Dict[str, float],
    server_max_rank: Dict[str, int],
    demand_tps: Dict[str, float],
):
    """
    Allocate the remaining fraction of 'adapter' to a single server:
    Prefer servers that already have a higher (or equal) max rank, and among them
    choose least utilized. If none, choose least utilized overall.
    """
    total = float(demand_tps[adapter.id])
    already = sum(assignment[adapter.id].values())
    remaining_phi = max(0.0, 1.0 - already)
    if remaining_phi <= 1e-12:
        return

    remaining_tps = remaining_phi * total

    # Candidate servers that already run >= this rank (i.e., "higher maximum rank, if possible")
    candidates = [sv for sv in servers if server_max_rank[sv.id] >= adapter.rank]
    if not candidates:
        candidates = servers[:]

    # Pick least utilized among candidates
    sv = min(candidates, key=lambda s: server_load_tps[s.id])

    assignment[adapter.id][sv.id] = assignment[adapter.id].get(sv.id, 0.0) + remaining_phi
    server_load_tps[sv.id] += remaining_tps
    server_max_rank[sv.id] = max(server_max_rank[sv.id], adapter.rank)


# -----------------------------
# Step 5: PERMUTEASSIGNMENT to minimize churn vs previous placement
# -----------------------------

def permute_assignment_to_match_previous(
    servers: List[Server],
    assignment: Dict[str, Dict[str, float]],
    prev_primary: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """
    With identical servers, LoRAServe permutes placements across servers to minimize deviation.
    Here we compute a *server-id relabeling* that maximizes overlap with previous placement.

    prev_primary: adapter_id -> server_id (primary server last timestep)
    Returns: mapping new_server_id -> permuted_server_id (old id space)
    """
    if not prev_primary:
        return {sv.id: sv.id for sv in servers}

    # Previous server -> set(adapters)
    prev_sets: Dict[str, Set[str]] = defaultdict(set)
    for a, s in prev_primary.items():
        prev_sets[s].add(a)

    # Current server -> set(adapters with any phi)
    cur_sets: Dict[str, Set[str]] = defaultdict(set)
    for a, dist in assignment.items():
        for s, phi in dist.items():
            if phi > 0:
                cur_sets[s].add(a)

    prev_server_ids = [sv.id for sv in servers]
    cur_server_ids = [sv.id for sv in servers]

    used_cur: Set[str] = set()
    mapping: Dict[str, str] = {}

    # Greedy maximum overlap matching
    for prev_s in prev_server_ids:
        best_cur = None
        best_overlap = -1
        prev_adapters = prev_sets.get(prev_s, set())

        for cur_s in cur_server_ids:
            if cur_s in used_cur:
                continue
            overlap = len(prev_adapters & cur_sets.get(cur_s, set()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cur = cur_s

        if best_cur is not None:
            mapping[best_cur] = prev_s
            used_cur.add(best_cur)

    # Unmatched keep identity
    for cur_s in cur_server_ids:
        mapping.setdefault(cur_s, cur_s)

    return mapping


def apply_server_permutation(
    assignment: Dict[str, Dict[str, float]],
    server_load_tps: Dict[str, float],
    server_max_rank: Dict[str, int],
    perm: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, int]]:
    """
    Relabel server IDs according to perm[new_id] = old_id target (or just another label).
    """
    new_assignment: Dict[str, Dict[str, float]] = defaultdict(dict)
    new_load: Dict[str, float] = defaultdict(float)
    new_max_rank: Dict[str, int] = defaultdict(int)

    # assignment: adapter -> {server -> phi}
    for a, dist in assignment.items():
        for s, phi in dist.items():
            s2 = perm.get(s, s)
            new_assignment[a][s2] = new_assignment[a].get(s2, 0.0) + phi

    # Loads and max ranks must be permuted similarly:
    for s, load in server_load_tps.items():
        s2 = perm.get(s, s)
        new_load[s2] += load
        new_max_rank[s2] = max(new_max_rank[s2], server_max_rank.get(s, 0))

    return new_assignment, dict(new_load), dict(new_max_rank)


# -----------------------------
# Step 6: Update metadata / routing table (pure data here)
# -----------------------------

def build_server_view(servers: List[Server], assignment: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    view: Dict[str, Dict[str, float]] = {sv.id: {} for sv in servers}
    for a, dist in assignment.items():
        for s, phi in dist.items():
            if phi > 0:
                view.setdefault(s, {})[a] = phi
    return view


def normalize_adapter_distributions(assignment: Dict[str, Dict[str, float]]):
    """
    Ensure each adapter distribution sums exactly to 1 (within eps).
    If sums differ due to floating error, renormalize.
    """
    for a, dist in assignment.items():
        s = sum(dist.values())
        if s <= 0:
            continue
        if abs(s - 1.0) > 1e-9:
            for k in list(dist.keys()):
                dist[k] = dist[k] / s


class PlacementAlgorithmLoRAServe(PlacementAlgorithmInterfaceDynamicRoutingWithProbabilities):

    def __init__(self):
        return

    def define_placement(
            self,
            model: str,
            servers: List[str],
            adapters: List[str],
            adapters_ranks: List[int],
            adapters_rates: List[float],
            mean_input_length: float,
            mean_output_length: float,
    ) -> Tuple[List[int], List[List[str]], List[List[float]]]:
        global OPERATING_POINTS
        """
            Determines the placement of adapters across available servers.

            Parameters:
            - servers (List[int]): A list of server integer IDs.
            - adapters (List[str]): A list of adapter string IDs.
            - adapters_ranks (List[int]): Rank associated with each adapter, linked by position with adapters input parameter.
            - adapters_rates (List[float]): Rate associated with each adapter, linked by position with adapters input parameter.
            - mean_input_length (float): Average length of input tokens per request.
            - mean_output_length (float): Average length of output tokens per request.

            Returns:
            - Tuple[List[int], List[List[str]], List[List[float]]]:
                - First list: Adapter slot associated with each server, linked by position with servers input parameter.
                - Second list: Servers associated with each adapter, linked by position with adapters input parameter.
                - Third list: Probabilities of servers associated with each adapter, linked by position with adapters input parameter.
        """
        # transform input servers and adapters to expected data structure
        servers_new: List[Server] = [Server(id=server_key) for server_key in servers]
        adapters_new: List[Adapter] = [Adapter(
            id=adapter_key,
            rank=adapters_ranks[adapter_index],
        ) for adapter_index, adapter_key in enumerate(adapters)]
        demand_tps: Dict[str, float] = {
            adapter_key: adapters_rates[adapter_index] * (mean_input_length + mean_output_length)
            for adapter_index, adapter_key in enumerate(adapters)
        }

        # run LoRAServe recreated code
        result = self.assign_loraserve(
            servers=servers_new,
            adapters=adapters_new,
            operating_points=OPERATING_POINTS[model],
            demand_tps=demand_tps,
            prev_primary=None,
        )

        '''print("\nRouting table (adapter -> [(server, phi), ...]):")
        for a, dist in result.routing_table.items():
            print(a, dist)
        print("\nServer view:")
        for s in servers_new:
            print(s.id, "adapters=", result.server_adapter_phi.get(s.id, {}))'''

        # transform output to our output
        servers_adapter_slots = [len(result.server_adapter_phi.get(server_key, {})) for server_key in servers]
        adapters_servers = [[key for key, value in result.routing_table[adapter_key]] for adapter_key in adapters]
        adapters_servers_probabilities = [[value for key, value in result.routing_table[adapter_key]] for adapter_key in adapters]

        return servers_adapter_slots, adapters_servers, adapters_servers_probabilities

    def assign_loraserve(
            self,
            servers: List[Server],
            adapters: List[Adapter],
            operating_points: Dict[int, float],  # rank -> max TPS under SLO (operating point)
            demand_tps: Dict[str, float],  # adapter_id -> TPS future
            prev_primary: Optional[Dict[str, str]] = None,  # adapter_id -> primary server id last time step
    ) -> PlacementResult:
        """
        Closest-to-paper structure:
          Step 1) demand estimation + target utilization
          Step 2) per-rank server budget
          Step 3) fractional bin packing for ranks with budget
          Step 4) place leftovers on higher-max-rank servers when possible
          Step 5) permute servers to minimize deviation vs prev assignment
          Step 6) build routing table + server-local metadata views
        """

        if not servers:
            raise ValueError("servers must be non-empty")
        if not adapters:
            # Trivial empty placement
            return PlacementResult(
                routing_table={},
                server_adapter_phi={sv.id: {} for sv in servers},
            )

        # ---------- Step 1: estimate TPS demand per adapter ----------
        # given by input

        # Compute rankUtil and targetUtil (normalized average utilization per server)
        ranks = sorted({ad.rank for ad in adapters})
        rank_util: Dict[int, float] = {}
        total_util = 0.0

        for r in ranks:
            op = operating_points.get(r)
            if op is None or op <= 0:
                raise ValueError(f"Missing/invalid operating point for rank {r}: {op}")
            total_rank_demand = sum(demand_tps[ad.id] for ad in adapters if ad.rank == r)
            rank_util[r] = total_rank_demand / op
            total_util += rank_util[r]

        target_util = total_util / len(servers)
        if target_util > 1.0 + 1.e-9:
            raise Exception('Not enough servers for input workload')

        # ---------- Step 2: server budget per rank ----------
        # Paper shows ROUND(rankUtil[rank]/targetUtil).
        # Since all servers are equal, we apportion budgets so total == #servers (stable, minimal surprises).
        raw_budget_weights = {r: (rank_util[r] / target_util) if target_util > 0 else 0.0 for r in ranks}
        rank_server_budget = hamilton_apportionment(raw_budget_weights, total=len(servers))

        # Prepare server bookkeeping
        assignment: Dict[str, Dict[str, float]] = defaultdict(dict)
        server_load_tps: Dict[str, float] = {sv.id: 0.0 for sv in servers}
        server_max_rank: Dict[str, int] = {sv.id: 0 for sv in servers}

        # Partition servers into disjoint groups by rank budget (simple, deterministic)
        # Here we allocate higher ranks first so lower-rank leftovers can be placed on high-rank servers later.
        servers_by_rank: Dict[int, List[Server]] = {r: [] for r in ranks}
        free_servers = servers[:]
        ranks_desc = sorted(ranks, reverse=True)

        idx = 0
        for r in ranks_desc:
            b = rank_server_budget.get(r, 0)
            servers_by_rank[r] = free_servers[idx: idx + b]
            idx += b

        # ---------- Step 3: pack ranks with non-zero budget using fractional bin packing ----------
        leftovers: List[Adapter] = []
        for r in ranks_desc:
            b = rank_server_budget.get(r, 0)
            if b <= 0:
                continue

            group = servers_by_rank[r]
            cap = target_util * operating_points[r]  # server TPS "soft target" for this rank group

            rank_adapters = [Adapter(ad.id, ad.rank) for ad in adapters if ad.rank == r]
            rank_leftovers = fractional_bin_packing(
                rank_adapters, group, cap, assignment, server_load_tps, server_max_rank, demand_tps
            )
            leftovers.extend(rank_leftovers)

        # Also include any adapters from ranks with zero server budget as leftovers
        budgeted_ranks = {r for r, b in rank_server_budget.items() if b > 0}
        for ad in adapters:
            if ad.rank not in budgeted_ranks:
                # ensure demand used (extrapolated)
                leftovers.append(Adapter(ad.id, ad.rank))

        # ---------- Step 4: allocate leftovers (descending rank), prefer higher-max-rank servers ----------
        leftovers_sorted = sorted(
            {ad.id: ad for ad in leftovers}.values(),  # de-dup by id
            key=lambda a: a.rank,
            reverse=True
        )
        for ad in leftovers_sorted:
            allocate_highest_max_rank(ad, servers, assignment, server_load_tps, server_max_rank, demand_tps)

        # Normalize adapter distributions (sum phi = 1)
        normalize_adapter_distributions(assignment)

        # ---------- Step 5: permute placement across servers to reduce churn ----------
        perm = permute_assignment_to_match_previous(servers, assignment, prev_primary)
        assignment, server_load_tps, server_max_rank = apply_server_permutation(
            assignment, server_load_tps, server_max_rank, perm
        )
        normalize_adapter_distributions(assignment)

        # ---------- Step 6: update routing table + local metadata ----------
        routing_table: Dict[str, List[Tuple[str, float]]] = {}
        for a, dist in assignment.items():
            routing_table[a] = sorted(dist.items(), key=lambda x: (-x[1], x[0]))  # sort by phi desc

        server_adapter_phi = build_server_view(servers, assignment)

        return PlacementResult(
            routing_table=routing_table,
            server_adapter_phi=server_adapter_phi,
        )

