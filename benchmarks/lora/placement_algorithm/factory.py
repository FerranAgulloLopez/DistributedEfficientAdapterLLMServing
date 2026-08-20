from typing import Type
from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from benchmarks.lora.placement_algorithm.interface_dynamic_routing import PlacementAlgorithmInterfaceDynamicRouting
from benchmarks.lora.placement_algorithm.interface_dynamic_routing_with_probabilities import \
    PlacementAlgorithmInterfaceDynamicRoutingWithProbabilities
from benchmarks.lora.placement_algorithm.subclasses.lora_serve import PlacementAlgorithmLoRAServe
from benchmarks.lora.placement_algorithm.subclasses.lora_serve_half import PlacementAlgorithmLoRAServeHalf
from benchmarks.lora.placement_algorithm.subclasses.random import PlacementAlgorithmRandom
from benchmarks.lora.placement_algorithm.subclasses.solver import PlacementAlgorithmSolver
from benchmarks.lora.placement_algorithm.subclasses.baseline_1 import PlacementAlgorithmBASELINE1
from benchmarks.lora.placement_algorithm.subclasses.baseline_2 import PlacementAlgorithmBASELINE2
from benchmarks.lora.placement_algorithm.subclasses.baseline_3 import PlacementAlgorithmBASELINE3
from benchmarks.lora.placement_algorithm.subclasses.baseline_3_2 import PlacementAlgorithmBASELINE3_2
from benchmarks.lora.placement_algorithm.subclasses.baseline_4 import PlacementAlgorithmBASELINE4
from benchmarks.lora.placement_algorithm.subclasses.baseline_6 import PlacementAlgorithmBASELINE6
from benchmarks.lora.placement_algorithm.subclasses.baseline_4_with_proposal import PlacementAlgorithmBASELINE4WithProposal
from benchmarks.lora.placement_algorithm.subclasses.baseline_5 import PlacementAlgorithmBASELINE5
from benchmarks.lora.placement_algorithm.subclasses.proposal import PlacementAlgorithmProposal
from benchmarks.lora.placement_algorithm.subclasses.proposal_throughput import PlacementAlgorithmProposalThroughput
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation import PlacementAlgorithmProposalStarvation
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_2 import PlacementAlgorithmProposalStarvation2
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_2_fast import PlacementAlgorithmProposalStarvation2Fast
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_3 import PlacementAlgorithmProposalStarvation3
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_4 import PlacementAlgorithmProposalStarvation4
from benchmarks.lora.placement_algorithm.subclasses.dlora_proactive_mechanism import PlacementAlgorithmDLoRAProactiveMechanism


ACCEPTED_SUBCLASSES = {
    'random': PlacementAlgorithmRandom,
    'solver': PlacementAlgorithmSolver,
    'baseline-1': PlacementAlgorithmBASELINE1,
    'baseline-2': PlacementAlgorithmBASELINE2,
    'baseline-3': PlacementAlgorithmBASELINE3,
    'baseline-3-2': PlacementAlgorithmBASELINE3_2,
    'baseline-4': PlacementAlgorithmBASELINE4,
    'baseline-4-with-proposal': PlacementAlgorithmBASELINE4WithProposal,
    'baseline-5': PlacementAlgorithmBASELINE5,
    'baseline-6': PlacementAlgorithmBASELINE6,
    'proposal': PlacementAlgorithmProposal,
    'proposal-throughput': PlacementAlgorithmProposalThroughput,
    'proposal-starvation': PlacementAlgorithmProposalStarvation,
    'proposal-starvation-2': PlacementAlgorithmProposalStarvation2,
    'proposal-starvation-2-fast': PlacementAlgorithmProposalStarvation2Fast,
    'proposal-starvation-3': PlacementAlgorithmProposalStarvation3,
    'proposal-starvation-4': PlacementAlgorithmProposalStarvation4,
}


ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING = {
    'dlora-proactive-mechanism': PlacementAlgorithmDLoRAProactiveMechanism,
}


ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING_WITH_PROBABILITIES = {
    'lora-serve': PlacementAlgorithmLoRAServe,
    'lora-serve-half': PlacementAlgorithmLoRAServeHalf,
}


def check_subclass(subclass_type: str):
    if subclass_type in ACCEPTED_SUBCLASSES:
        return
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')


def get_subclass(subclass_type: str) -> Type[PlacementAlgorithmInterface]:
    if subclass_type in ACCEPTED_SUBCLASSES:
        return ACCEPTED_SUBCLASSES[subclass_type]
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')


def check_subclass_dynamic_routing(subclass_type: str):
    if subclass_type in ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING:
        return
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')


def get_subclass_dynamic_routing(subclass_type: str) -> Type[PlacementAlgorithmInterfaceDynamicRouting]:
    if subclass_type in ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING:
        return ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING[subclass_type]
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')


def check_subclass_dynamic_routing_with_probabilities(subclass_type: str):
    if subclass_type in ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING_WITH_PROBABILITIES:
        return
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')


def get_subclass_dynamic_routing_with_probabilities(subclass_type: str) -> Type[PlacementAlgorithmInterfaceDynamicRoutingWithProbabilities]:
    if subclass_type in ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING_WITH_PROBABILITIES:
        return ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING_WITH_PROBABILITIES[subclass_type]
    else:
        raise ValueError(f'Subclass {subclass_type} of placement algorithm does not exist')
