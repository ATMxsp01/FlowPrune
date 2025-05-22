import torch
from copy import deepcopy

from Types import *
import random
import numpy as np

def _compact_list_normalize(compact_list: Compact_list) -> Compact_list:
    if compact_list == []:
        return []
    compact_list.sort(key=lambda rg: (rg[0], rg[1]))
    ret = [compact_list[0]]
    for i in range(1, len(compact_list)):
        if compact_list[i][0] >= compact_list[i][1]:
            continue
        if ret[-1][1] >= compact_list[i][0]:
            ret[-1] = (ret[-1][0], compact_list[i][1])
        else:
            ret.append(compact_list[i])
    return ret



def heuristic_tactic_0(attention: Attention,
                       retain_rate: float, 
                       trying_times: int,
                       ) -> Tuple[Compact_list, Args]:
    layer_depth = len(attention) + 1
    layer_width = attention[0].size(0)
    routes = np.ndarray((layer_width, layer_depth), dtype=np.int32)
    min_count = np.zeros(layer_depth - 1, dtype=np.int32)
    tot_count = 0
    for _ in range(trying_times * 3):
        if tot_count >= trying_times * layer_width:
            break
        for i in range(routes.shape[1]):
            routes[:, i] = np.random.permutation(layer_width)
        for route in routes:
            min = attention[0][route[0]][route[1]]
            min_p = 0
            for i in range(1, route.shape[0] - 1):
                cand = attention[i][route[i]][route[i + 1]]
                if cand == 0:
                    break
                if cand < min:
                    min = cand
                    min_p = i
            min_count[min_p] += 1
            tot_count += 1

    retain_tot = max(3, round(retain_rate * (layer_depth - 1)))
    retain_tot -= 2
    f = np.ndarray((layer_depth, retain_tot + 2, 2), dtype=np.int32)
    f[:, :, :] = -32768
    f[:, 1, 0] = 0
    f[1, 2, 1] = min_count[1].item()
    for i in range(2, layer_depth - 1):
        for j in range(2, retain_tot + 1):
            f[i][j][0] = max(f[i - 1][j][0].item(), f[i - 1][j][1].item())
            f[i][j][1] = max(f[i - 1][j - 1][1].item(), f[i - 1][j - 2][0].item()) + min_count[i].item()

    uncompact_list = [layer_depth - 1]
    j = retain_tot
    i = layer_depth - 2
    k = 1 if f[i][j][1] > f[i][j][0] else 0
    
    while j > 1:
        if k:
            uncompact_list.append(i)
            if f[i - 1][j - 1][1] > f[i - 1] [j - 2][0]:
                j -= 1
            else:
                j -= 2
                k = 0
        else:
            k = 1 if f[i - 1][j][0] < f[i - 1][j][1] else 0
        i -= 1

    uncompact_list.append(0)
    uncompact_list.reverse()
    compact_list: Compact_list = [(a + 1, b - 1) for a, b in zip(uncompact_list[:-1], uncompact_list[1:]) if 2 < b - a]
    return compact_list, {'min_count': min_count, 'totcount_offset' : tot_count - layer_width * trying_times}
    






def heuristic_tactic_1(attention: Attention,
                       retain_rate: float, 
                       trying_times: int,
                       ) -> Tuple[Compact_list, Args]:
    layer_depth = len(attention) + 1
    layer_width = attention[0].size(0)
    min_count = np.zeros(layer_depth - 1, dtype=np.int32)
    tot_count = 0
    valid_road_counts = [(attn > 0).sum(dim=1).int() for attn in attention]
    for _ in range(trying_times * 2):
        if tot_count >= trying_times * layer_width:
            break
        for start in range(layer_width):
            min = 1.1
            min_p = -1
            pos = start
            for p, (road_count, attn) in enumerate(zip(valid_road_counts, attention)):
                if road_count[pos].item() == 0:
                    min = 0
                    break

                val, ind = attn[pos].topk(int(road_count[pos].item()))
                rnd = random.randint(0, int(road_count[pos].item()) - 1)
                if val[rnd] < min:
                    min = val[rnd]
                    min_p = p
                pos = ind[rnd]
            min_count[min_p] += 1
            tot_count += 1

    retain_tot = max(3, round(retain_rate * (layer_depth - 1)))
    retain_tot -= 2
    f = np.ndarray((layer_depth, retain_tot + 2, 2), dtype=np.int32)
    f[:, :, :] = -32768
    f[:, 1, 0] = 0
    f[1, 2, 1] = min_count[1].item()
    for i in range(2, layer_depth - 1):
        for j in range(2, retain_tot + 1):
            f[i][j][0] = max(f[i - 1][j][0].item(), f[i - 1][j][1].item())
            f[i][j][1] = max(f[i - 1][j - 1][1].item(), f[i - 1][j - 2][0].item()) + min_count[i].item()

    uncompact_list = [layer_depth - 1]
    j = retain_tot
    i = layer_depth - 2
    k = 1 if f[i][j][1] > f[i][j][0] else 0
    
    while j > 1:
        if k:
            uncompact_list.append(i)
            if f[i - 1][j - 1][1] > f[i - 1] [j - 2][0]:
                j -= 1
            else:
                j -= 2
                k = 0
        else:
            k = 1 if f[i - 1][j][0] < f[i - 1][j][1] else 0
        i -= 1

    uncompact_list.append(0)
    uncompact_list.reverse()
    compact_list: Compact_list = [(a + 1, b - 1) for a, b in zip(uncompact_list[:-1], uncompact_list[1:]) if 2 < b - a]
    return compact_list, {'min_count': min_count, 'totcount_offset' : tot_count - layer_width * trying_times}


def heuristic_tactic_2(attention: Attention,
                       retain_rate: float, 
                       trying_times: int,
                       ) -> Tuple[Compact_list, Args]:
    layer_depth = len(attention) + 1
    layer_width = attention[0].size(0)
    min_count = np.zeros(layer_depth - 1, dtype=np.int32)
    tot_count = 0
    valid_road_counts = [(attn > 0).sum(dim=1).int() for attn in attention]
    r_valid_road_counts = [(attn > 0).sum(dim=0).int() for attn in attention]
    for _ in range(trying_times * 2):
        if tot_count >= trying_times * layer_width * 2:
            break
        for start in range(layer_width):
            min = 1.1
            min_p = -1
            pos = start
            for p, (road_count, attn) in enumerate(zip(valid_road_counts, attention)):
                if road_count[pos].item() == 0:
                    min = 0
                    break

                val, ind = attn[pos].topk(int(road_count[pos].item()))
                rnd = random.randint(0, int(road_count[pos].item()) - 1)
                if val[rnd] < min:
                    min = val[rnd]
                    min_p = p
                pos = ind[rnd]
            min_count[min_p] += 1
            tot_count += 1
        for start in range(layer_width):
            min = 1.1
            min_p = -1
            pos = start
            L = [x for x in enumerate(zip(r_valid_road_counts, attention))]
            L.reverse()
            for p, (road_count, attn) in L:
                if road_count[pos].item() == 0:
                    min = 0
                    break

                val, ind = attn[:,pos].topk(int(road_count[pos].item()))
                rnd = random.randint(0, int(road_count[pos].item()) - 1)
                if val[rnd] < min:
                    min = val[rnd]
                    min_p = p
                pos = ind[rnd]
            min_count[min_p] += 1
            tot_count += 1

    retain_tot = max(3, round(retain_rate * (layer_depth - 1)))
    retain_tot -= 2
    f = np.ndarray((layer_depth, retain_tot + 2, 2), dtype=np.int32)
    f[:, :, :] = -32768
    f[:, 1, 0] = 0
    f[1, 2, 1] = min_count[1].item()
    for i in range(2, layer_depth - 1):
        for j in range(2, retain_tot + 1):
            f[i][j][0] = max(f[i - 1][j][0].item(), f[i - 1][j][1].item())
            f[i][j][1] = max(f[i - 1][j - 1][1].item(), f[i - 1][j - 2][0].item()) + min_count[i].item()

    uncompact_list = [layer_depth - 1]
    j = retain_tot
    i = layer_depth - 2
    k = 1 if f[i][j][1] > f[i][j][0] else 0
    
    while j > 1:
        if k:
            uncompact_list.append(i)
            if f[i - 1][j - 1][1] > f[i - 1] [j - 2][0]:
                j -= 1
            else:
                j -= 2
                k = 0
        else:
            k = 1 if f[i - 1][j][0] < f[i - 1][j][1] else 0
        i -= 1

    uncompact_list.append(0)
    uncompact_list.reverse()
    compact_list: Compact_list = [(a + 1, b - 1) for a, b in zip(uncompact_list[:-1], uncompact_list[1:]) if 2 < b - a]
    return compact_list, {'min_count': min_count, 'totcount_offset' : tot_count - layer_width * trying_times * 2}

def heuristic_tactic_3(attention: Attention, 
                       high_threshold: float,
                       retain_rate: float,
                       ) -> Tuple[Compact_list, Args]:
    layer_depth = len(attention) + 1
    layer_width = attention[0].size(0)
    min_count = np.zeros(layer_depth - 1, dtype=np.int32)
    tot_count = 0
    for i in range(layer_depth - 1):
        min_count[i] = ((attention[i] > 0) & (attention[i] <= high_threshold)).sum().item()
        tot_count += min_count[i]
    
    retain_tot = max(3, round(retain_rate * (layer_depth - 1)))
    retain_tot -= 2
    f = np.ndarray((layer_depth, retain_tot + 2, 2), dtype=np.int32)
    f[:, :, :] = -32768
    f[:, 1, 0] = 0
    f[1, 2, 1] = min_count[1].item()
    for i in range(2, layer_depth - 1):
        for j in range(2, retain_tot + 1):
            f[i][j][0] = max(f[i - 1][j][0].item(), f[i - 1][j][1].item())
            f[i][j][1] = max(f[i - 1][j - 1][1].item(), f[i - 1][j - 2][0].item()) + min_count[i].item()

    uncompact_list = [layer_depth - 1]
    j = retain_tot
    i = layer_depth - 2
    k = 1 if f[i][j][1] > f[i][j][0] else 0
    
    while j > 1:
        if k:
            uncompact_list.append(i)
            if f[i - 1][j - 1][1] > f[i - 1] [j - 2][0]:
                j -= 1
            else:
                j -= 2
                k = 0
        else:
            k = 1 if f[i - 1][j][0] < f[i - 1][j][1] else 0
        i -= 1

    uncompact_list.append(0)
    uncompact_list.reverse()
    compact_list: Compact_list = [(a + 1, b - 1) for a, b in zip(uncompact_list[:-1], uncompact_list[1:]) if 2 < b - a]
    return compact_list, {'totcount' : tot_count}