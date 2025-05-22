import torch
from copy import deepcopy

import Tactics
from Types import *


def _do_nothing(attention: Attention) -> Tuple[Attention, None]:
    return attention , None


def _cut_threshold(attention: Attention, threshold: float) -> Tuple[Attention, Args]:
    attention = deepcopy(attention)
    extra_info = {
        'cut_edge_tot' : sum([(attn <= threshold).sum().item() for attn in attention])
    }
    for attn in attention:
        attn[attn < threshold] = 0
    return attention , extra_info

def _compact_with_compact_list(attention: Attention, 
                              compact_list: Optional[Compact_list] = None,
                              threshold: float = 0
                              ) -> Tuple[Attention, Args]:
    compacted_attention: Attention = []
    step = 0

    if compact_list == None:
        compact_list = []
    extra_info: Args = {'compact-list' : compact_list}
    for start_attn, end_attn in compact_list:
        while step < start_attn:
            compacted_attention.append(attention[step])
            step += 1
        
        max_ = attention[start_attn].max().item()
        compacted_attn = attention[start_attn]
        for step_ in range(start_attn + 1, end_attn + 1):
            compacted_attn = (torch.mm(compacted_attn, attention[step_]) > threshold).to(attention[0].dtype)
            max_ = max(max_, attention[step_].max().item())


        compacted_attn[compacted_attn <= threshold] = 0
        compacted_attn[compacted_attn > threshold] = max_
        compacted_attention.append(compacted_attn)
        step = end_attn + 1
    
    while step < len(attention):
        compacted_attention.append(attention[step])
        step += 1

    extra_info['layer-depth-after-compact'] = len(compacted_attention) + 1

    return compacted_attention, extra_info


def _compact_with_tactic(attention: Attention, 
                        threshold: float,
                        tactic: Callable[..., Tuple[Compact_list, Args]], 
                        tactic_args : Args,
                        )-> Tuple[Attention, Args]:
    compact_list, tactic_info = tactic(attention, **tactic_args)
    attention, compact_info = _compact_with_compact_list(attention, compact_list, threshold)
    return attention, {**tactic_info, **compact_info}


def _compact_to_four_layers(attention: Attention,
                            threshold: float = 0) -> Tuple[Attention, Args]:
    depth = len(attention)
    if depth > 3:
        compact_list = [(1, depth - 2)]
    else:
        compact_list = []
    return _compact_with_compact_list(attention, compact_list, threshold)

def _compact_with_heuristic_tactic_0(attention: Attention, 
                        threshold: float,
                        retain_rate: float,
                        trying_times: int
                        )-> Tuple[Attention, Args]:
    return _compact_with_tactic(attention, threshold, 
                                Tactics.heuristic_tactic_0, 
                                {
                                    'retain_rate' : retain_rate,
                                    'trying_times' : trying_times,
                                }
    )


def _compact_with_heuristic_tactic_1(attention: Attention, 
                        threshold: float,
                        retain_rate: float,
                        trying_times: int
                        )-> Tuple[Attention, Args]:
    return _compact_with_tactic(attention, threshold, 
                                Tactics.heuristic_tactic_1, 
                                {
                                    'retain_rate' : retain_rate,
                                    'trying_times' : trying_times,
                                }
    )

def _compact_with_heuristic_tactic_2(attention: Attention, 
                        threshold: float,
                        retain_rate: float,
                        trying_times: int
                        )-> Tuple[Attention, Args]:
    return _compact_with_tactic(attention, threshold,
                                Tactics.heuristic_tactic_2, 
                                {
                                    'retain_rate' : retain_rate,
                                    'trying_times' : trying_times,
                                }
    )

def _compact_with_heuristic_tactic_3(attention: Attention,
                        threshold: float,
                        high_threshold: float,
                        retain_rate: float
                        )-> Tuple[Attention, Args]:
    return _compact_with_tactic(attention, threshold,
                                Tactics.heuristic_tactic_3, 
                                {
                                    'high_threshold' : high_threshold,
                                    'retain_rate' : retain_rate,
                                }
    )

Attention_processor : Dict[str, Callable[..., Tuple[Attention, Optional[Args]]]] = {
    'do_nothing': _do_nothing,
    'cut_threshold': _cut_threshold,
    'compact_to_four_layers': _compact_to_four_layers,
    'compact_with_heuristic_tactic_0': _compact_with_heuristic_tactic_0,
    'compact_with_heuristic_tactic_1': _compact_with_heuristic_tactic_1,
    'compact_with_heuristic_tactic_2': _compact_with_heuristic_tactic_2,
    'compact_with_heuristic_tactic_3': _compact_with_heuristic_tactic_3,
}
