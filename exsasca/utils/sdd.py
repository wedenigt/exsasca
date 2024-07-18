from typing import Dict
from pysdd.sdd import SddManager, Vtree, SddNode
import numpy as np
from exsasca.utils.aes import xtime
import pickle


def load_sdd(vtree_filename: str, sdd_filename: str):
    vtree = Vtree.from_file(vtree_filename.encode())
    manager = SddManager.from_vtree(vtree)
    sdd = manager.read_sdd_file(sdd_filename.encode())

    return sdd, manager


def read_pickle_file(pickle_file) -> any:
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def get_mc_vars():
    return ['x1', 'x2', 'x3', 'x4',
            'xm1', 'xm2', 'xm3', 'xm4',
            'x12', 'x23', 'x34', 'x41',
            'g',
            'xx12', 'xx23', 'xx34', 'xx41',
            'xx12g', 'xx23g', 'xx34g', 'xx41g']


def xor_permute_pmf(pmf, xor_val):
    return pmf[:, np.bitwise_xor(np.arange(0, 256).astype(np.uint8), xor_val)]


def inv_xtime_permute_pmf(pmf):
    byte_seq = np.arange(0, 256).astype(np.uint8)
    forward_lut = xtime(byte_seq)

    # inv = np.array([np.argwhere(forward_lut == i) for i in byte_seq]).squeeze()
    return pmf[:, forward_lut]


def pre_merge_pmfs(pmfs, g, merge_x12_x34=True):
    if len(pmfs.shape) == 2:
        pmfs = pmfs[None, :, :]

    # precompute pmfs using the fact that g=g
    mc_vars = get_mc_vars()
    for var in ['xx12', 'xx23', 'xx34', 'xx41']:
        # this is "a-priori message passing"
        pmfs[:, :, mc_vars.index(var)] *= xor_permute_pmf(pmfs[:, :, mc_vars.index(var + 'g')], xor_val=g)
        pmfs[:, :, mc_vars.index(var + 'g')] = 1  # set to uniform

    for var in ['x12', 'x23', 'x34', 'x41']:
        # this is "a-priori message passing"
        pmfs[:, :, mc_vars.index(var)] *= inv_xtime_permute_pmf(pmfs[:, :, mc_vars.index('x' + var)])
        pmfs[:, :, mc_vars.index('x' + var)] = 1  # set to uniform

    if merge_x12_x34:
        pmfs[:, :, mc_vars.index('x12')] *= xor_permute_pmf(pmfs[:, :, mc_vars.index('x34')], xor_val=g)
        pmfs[:, :, mc_vars.index('x34')] = 1  # set to uniform

        pmfs[:, :, mc_vars.index('x23')] *= xor_permute_pmf(pmfs[:, :, mc_vars.index('x41')], xor_val=g)
        pmfs[:, :, mc_vars.index('x41')] = 1  # set to uniform

    return pmfs


def indicator_structure_to_weight_map(manager, indicator_structure, pmfs) -> Dict[SddNode, float]:
    weight_map = {}
    for byte_idx in indicator_structure.keys():
        for indicator_lit_int in indicator_structure[byte_idx].keys():
            bit_assignment = indicator_structure[byte_idx][indicator_lit_int]
            byte_val = bit_assignment_to_int(bit_assignment)
            weight_map[manager.literal(indicator_lit_int)] = pmfs[byte_val, byte_idx]

    return weight_map


def bit_assignment_to_int(bit_assignment: Dict[int, int]) -> int:
    bitstring = ''.join([str(bit_assignment[key]) for key in sorted(bit_assignment.keys())])
    return int(bitstring[::-1], 2)
