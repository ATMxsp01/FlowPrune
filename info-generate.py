import os
import sys
import multiprocessing
import time
from datetime import datetime 

import itertools

import torch
import networkx as nx
from omegaconf import OmegaConf

import pickle

from Types import *
from ij_selector import Ij_selector

if len(sys.argv) == 2:
    _conf_path = sys.argv[1]
else:
    _conf_path = './config.yaml'

conf = OmegaConf.load(_conf_path)

attention_head_list = conf.attention_head_list

settings: List[Setting] = []
for _setting in conf.settings:
    settings.append(Setting(*_setting))


result_root_dir = conf.result_root_dir
attention_info_dir = conf.attention_info_dir

if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)

if conf.save_id is not None and conf.save_id >= 0:
    __save_id = conf.save_id
else:
    __save_id = 0
    while os.path.exists(f"{result_root_dir}{__save_id}/"):
        __save_id += 1
result_root_dir = f"{result_root_dir}{__save_id}/"
if not os.path.exists(result_root_dir):
    os.makedirs(result_root_dir)

with open(f"{conf.result_root_dir}/task_record.log", "a") as f:
    f.write(f"|{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}| save-id: {__save_id}\nTask: {conf.task_name}\n\n")

def pkl_save(obj : Any, _filename : str, overwrite: int = -1) -> None:
    filename = f"{result_root_dir}{_filename}"
    if os.path.exists(f"{filename}.pkl") and overwrite != 1:
        if overwrite == 0:
            return

        trailing = 1
        while os.path.exists(f"{filename}_({trailing}).pkl"):
            trailing += 1
        filename = f"{filename}_({trailing})"

    with open(f"{filename}.pkl", 'wb') as f:
        pickle.dump(obj, f)
    return

def pkl_exist(_filename : str) -> bool:
    filename = f"{result_root_dir}{_filename}"
    return os.path.exists(f"{filename}.pkl") 

def pkl_load(_filename: str) -> Any:
    filename = f"{result_root_dir}{_filename}"
    with open(f"{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    return data



def get_attention_size(attention: Attention) -> Attention_size:
    return Attention_size(len(attention) + 1, attention[0].size(0))

def attention_ptr_gen(attention_info_dir:str, 
                  mode : str = 'avg'
                  ) -> Generator[Attention_ptr, None, None]:
    candidates = os.listdir(attention_info_dir)
    candidates = [c for c in candidates if c.endswith('.pt') or c.endswith('.pkl')]
    if mode == 'mean':
        for candidate in candidates:
            yield candidate, -1
    elif mode == 'head':
        for head in attention_head_list:
            for candidate in candidates:
                yield candidate, head
    elif mode == 'max':
        for candidate in candidates:
            yield candidate, -2
    

def attention_get_info(attention_ptr: Attention_ptr) -> Optional[Attention_info]:
    candidate, head = attention_ptr
    # print(f"Loading {candidate} with head {head}")
    if candidate.endswith('.pkl'):
        _raw_attentions = pickle.load(open(f"{attention_info_dir}{candidate}", 'rb'))
        raw_attentions: torch.Tensor = [x.to('cpu') for x in _raw_attentions]
        if head == -1:
            attention, head_related = [torch.mean(attn, dim = 2) for attn in raw_attentions], 'mean'
        elif head == -2:
            attention, head_related = [torch.max(attn, dim = 2)[0] for attn in raw_attentions], 'max'
        else:
            if head > raw_attentions[0].size(2):
                return None
            attention, head_related =[raw_attentions[l][:,:,head] for l in range(len(raw_attentions))], head
        return Attention_info(
            name=f"{candidate[:-4]}-{head_related}",
            attention=attention,
        )
    raise Exception(f"Unknown file type: {candidate}")



def mk_graph(attentions: Attention) -> nx.DiGraph:
    G = nx.DiGraph()
    for l in range(len(attentions)):
        attention = attentions[l]
        for i in range(attention.size(0)):
            for j in range(attention.size(1)):
                if attention[i][j] != 0:
                    G.add_edge(f"{l}-{i}", f"{l+1}-{j}", capacity=attention[i][j].item())
    return G




def generate_info(attention_ptr: Attention_ptr, cnt: Optional[int] = None) -> None:
    attention_info = attention_get_info(attention_ptr)
    if attention_info is None:
        return
    if pkl_exist(attention_info.name):
        ij_list = pkl_load(attention_info.name)['ij-list']
    else:
        if conf.ij_selector_info.args is None:
            conf.ij_selector_info.args = {}
        ij_list = Ij_selector[conf.ij_selector_info.selector](attention_info.layer_width, 
                                                                   **conf.ij_selector_info.args)
        pkl_save({
                'attention-info':attention_info,
                'ij-list': ij_list
            }, 
            attention_info.name)

    print(f"[ ] {attention_info.name} Start! {f'The serial number is {cnt}' if cnt is not None else ''} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    for setting in settings:
        result = []
        if pkl_exist(f"{attention_info.name}--{setting.name}"):
            print(f"[!] {attention_info.name}--{setting.name} already exist! {f'The serial number is {cnt}' if cnt is not None else ''} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
            continue
        start_time = time.perf_counter_ns()
        attention, extra_info = setting.atten_processor(attention_info.attention, **setting.atten_processor_args)
        attention_size = get_attention_size(attention)
        G = mk_graph(attention)
        for i, j in ij_list:
            if G.has_node(f"0-{i}") and G.has_node(f"{attention_size.depth - 1}-{j}"):
                max_flow = nx.maximum_flow_value(G, f"0-{i}", f"{attention_size.depth - 1}-{j}")
            else:
                max_flow = 0
            result.append((f'{i}-{j}',max_flow, time.perf_counter_ns() - start_time))
        
        if extra_info is None:
            extra_info = {}
        info_to_save = {
            'setting' : setting.name,
            'attention-after-process' : attention,
            'maxflow-result' : result,
            **extra_info,
        }
        pkl_save(info_to_save, f"{attention_info.name}--{setting.name}")
        print(f"[-] {attention_info.name}--{setting.name} completed! {f'number {cnt}' if cnt is not None else ''} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"[*] All {attention_info.name} completed! {f'number {cnt}' if cnt is not None else ''} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")




if __name__ == "__main__":
    # for attention_ptr in attention_ptr_gen(attention_info_dir, 'max'):
    #     generate_info(attention_ptr)
    #     break

    # '''
    my_pool = multiprocessing.Pool(processes=conf.pool_size, maxtasksperchild=1)
    data_gen_iter = zip(itertools.chain(attention_ptr_gen(attention_info_dir, 'mean'),
                                        attention_ptr_gen(attention_info_dir, 'head'),
                                        ),
                        itertools.count())  
    my_pool.starmap(generate_info, data_gen_iter)

    my_pool.close()
    my_pool.join()
    print("=== All the task finished! ===")
    # '''
