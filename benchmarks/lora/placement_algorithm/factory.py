from typing import Type
from benchmarks.lora.placement_algorithm.interface import PlacementAlgorithmInterface
from benchmarks.lora.placement_algorithm.interface_dynamic_routing import PlacementAlgorithmInterfaceDynamicRouting
from benchmarks.lora.placement_algorithm.interface_dynamic_routing_with_probabilities import \
    PlacementAlgorithmInterfaceDynamicRoutingWithProbabilities
from benchmarks.lora.placement_algorithm.subclasses.lora_serve import PlacementAlgorithmLoRAServe
from benchmarks.lora.placement_algorithm.subclasses.random import PlacementAlgorithmRandom
from benchmarks.lora.placement_algorithm.subclasses.baseline_4_with_proposal import PlacementAlgorithmBASELINE4WithProposal
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_2 import PlacementAlgorithmProposalStarvation2
from benchmarks.lora.placement_algorithm.subclasses.proposal_starvation_2_fast import PlacementAlgorithmProposalStarvation2Fast
from benchmarks.lora.placement_algorithm.subclasses.dlora_proactive_mechanism import PlacementAlgorithmDLoRAProactiveMechanism


ACCEPTED_SUBCLASSES = {
    'random': PlacementAlgorithmRandom,
    'baseline-4-with-proposal': PlacementAlgorithmBASELINE4WithProposal,  # corresponds to paper ProposedLat algorithm
    'proposal-starvation-2': PlacementAlgorithmProposalStarvation2,  # corresponds to paper Proposed algorithm
    'proposal-starvation-2-fast': PlacementAlgorithmProposalStarvation2Fast,  # corresponds to paper ProposedFast algorithm
}


ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING = {
    'dlora-proactive-mechanism': PlacementAlgorithmDLoRAProactiveMechanism,  # corresponds to paper dLoRAProactive algorithm
}


ACCEPTED_SUBCLASSES_DYNAMIC_ROUTING_WITH_PROBABILITIES = {
    'lora-serve': PlacementAlgorithmLoRAServe,  # corresponds to paper LoRAServe algorithm
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
