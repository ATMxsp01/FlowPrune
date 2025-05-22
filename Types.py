import torch
from typing import List, Tuple, Dict, Any, Callable, Optional, Generator

Attention = List[torch.Tensor]
Compact_list = List[Tuple[int, int]]
Args = Dict[str, Any]
Attention_ptr = Tuple[str, int]

class Attention_size:
    def __init__(self, depth: int, width: int) -> None:
        self.depth = depth
        self.width = width


class Setting:
    def __init__(self, name: str, 
                 atten_processor_name: str, 
                 atten_prcessor_args: Args
                 ) -> None:
        from Attention_processor import Attention_processor
        self.name = name
        self.atten_processor : Callable[..., Tuple[Attention, Optional[Args]]] = \
            Attention_processor[atten_processor_name]
        self.atten_processor_args = atten_prcessor_args
        return


class Attention_info:
    def __init__(self,
                 name: str,
                 attention: Attention,
                 ) -> None:
        self.name = name
        self.attention = attention
        self.layer_depth : int = len(attention) + 1
        self.layer_width : int = attention[0].size(0)
        return
