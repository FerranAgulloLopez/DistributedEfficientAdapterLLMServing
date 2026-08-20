
from __future__ import annotations

from typing import List, Tuple

from benchmarks.lora.placement_algorithm.subclasses.lora_serve import PlacementAlgorithmLoRAServe


class PlacementAlgorithmLoRAServeHalf(PlacementAlgorithmLoRAServe):

    def define_placement(self, *args, **kwargs) -> Tuple[List[int], List[List[str]], List[List[float]]]:
        servers_adapter_slots, adapters_servers, adapters_servers_probabilities = super().define_placement(*args, **kwargs)
        servers_adapter_slots = [max(1, round(slots / 3)) for slots in servers_adapter_slots]
        return servers_adapter_slots, adapters_servers, adapters_servers_probabilities
