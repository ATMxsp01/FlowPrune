import random
from Types import *

def _all_ij(N : int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(N) for j in range(N)]

def _rand_ij(N : int, maxtot: int) -> list[tuple[int, int]]:
	candidate = [(i, j) for i in range(N) for j in range(N)]
	random.shuffle(candidate)
	return candidate[:min(maxtot, N * N)]


def _rand_i_all_j(N: int, itot: int) -> list[tuple[int, int]]:
    candidate = list(range(N))
    random.shuffle(candidate)
    return [(i, j) for j in range(N) for i in candidate[:min(N, itot)]]


def _rand_j_all_i(N: int, jtot: int) -> list[tuple[int, int]]:
    candidate = list(range(N))
    random.shuffle(candidate)
    return [(i, j) for i in range(N) for j in candidate[:min(N, jtot)]]

def _start_and_end(N: int, candidate: list[int]) -> list[tuple[int, int]]:
    _result = [(i, j) for i in range(N) for j in range(N)]
    return [x for x in _result if (x[0] in candidate) or (x[1] in candidate)]

Ij_selector: dict[str, Callable[..., list[tuple[int, int]]]] = {
    'all_ij': _all_ij,
    'rand_ij': _rand_ij,
    'rand_i_all_j': _rand_i_all_j,
    'rand_j_all_i': _rand_j_all_i,
    'start_and_end': _start_and_end,
}
