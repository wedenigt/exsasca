# import sys, os
# path = os.path.join(os.path.dirname(sys.path[0]),'pypsdd')
# print(path)
# sys.path.append(path)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysdd.sdd import Vtree as SddVtree, SddNode, SddManager
from pypsdd.vtree import Vtree
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import itertools
from tqdm import tqdm
from pysat.card import *
from pyeda.inter import *
import torch
#from pypsdd.tests.test_sasca_vs_psdd import create_mixcol_cnf, xor_cnf, xtimes, bit_to_byte_constraints, bit_to_byte_constraints_equivalence
# from generator_torch import mix_columns


def lca_of_literals_sdd_vtree(sdd_vtree, nodes, out_path=None):
    vtree = Vtree.from_sdd_vtree(sdd_vtree, out_path=out_path)
    return lca_of_literals(vtree, nodes)

def lca_of_literals(vtree, nodes):
    lits = set([node.literal for node in nodes])
    vars = vtree.variables()
    intersection = lits.intersection(vars)

    assert len(intersection) == len(lits)

    while True:
        left_intersection = vtree.left.variables().intersection(lits)
        right_intersection = vtree.right.variables().intersection(lits)

        if len(left_intersection) < len(lits) and len(right_intersection) < len(lits):
            return vtree

        if len(left_intersection) < len(lits):
            vtree = vtree.right
        else:
            vtree = vtree.left


def sort_bits_by_vtree_lca(sdd_vtree: SddVtree, bits: List[List[SddNode]], out_path=None) -> List[List[SddNode]]:
    """
    Sorts bits by the LCA of their literals in the vtree.
    """
    vtree = Vtree.from_sdd_vtree(sdd_vtree, out_path=out_path)

    heights = []
    for lits in bits:
        # lits is a list of literals
        lca = lca_of_literals(vtree, lits)
        heights.append(lca.height())

    bits = np.array(bits)
    return bits[np.argsort(heights)] # sort by height - first element has the lowest height

def conjoin_best_byte_greedily(sdd, bytes, alpha, P, manager):
    # bytes must be prefiltered with nasty bits
    sizes = []
    best_size = np.infty
    best_byte = None
    best_sdd = sdd

    for i, byte in tqdm(enumerate(bytes)):
        sdd_new, indicator_structure = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, byte,
                                                                          deref_old_sdd=False, last_call_for_byte=False)
        sizes.append(sdd.size())
        if best_size > sdd_new.size():
            best_size = sdd_new.size()
            best_byte = byte
            best_sdd = sdd_new
        else:
            sdd_new.deref()
            manager.garbage_collect()

    sdd.deref()
    manager.garbage_collect()

    print(f'best byte: {best_byte=}')
    return best_sdd

def conjoin_best_bit_to_sdd(sdd: SddNode, alpha: Optional[SddNode], P: List[SddNode], prev_P: List[SddNode],
                            manager: SddManager, bits_already_selected: List[SddNode],
                            bits: List[SddNode], exception_bit_ids=None,
                            skip_optimization=False, byte_indicators=False,
                            last_call_for_byte=False) -> tuple[SddNode, SddNode, Optional[Dict]]:
    # assert byte_indicators or len(bits_already_selected) >= 2
    indicator_structure = None

    bits_to_choose_from = [bit for bit in bits if bit not in bits_already_selected]
    if exception_bit_ids is not None:
        bits_to_choose_from = [bit for bit in bits_to_choose_from if bit not in exception_bit_ids]

    if len(bits_to_choose_from) == 0:
        return sdd, None, indicator_structure

    if len(bits_to_choose_from) == 1 or skip_optimization:
        best_bit = bits_to_choose_from[0]
        if alpha is None:
            sdd = equiv_bits_indicator_clause(sdd, prev_P, P, manager, best_bit, deref_old_sdd=True)
        else:
            if byte_indicators:
                sdd, indicator_structure = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, bits_to_choose_from,
                                                         deref_old_sdd=True, last_call_for_byte=True)
            else:
                sdd, indicator_structure = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, bits_already_selected + [best_bit],
                                                             deref_old_sdd=True, last_call_for_byte=last_call_for_byte)
        return sdd, best_bit, indicator_structure

    # manager.auto_gc_and_minimize_off()
    original_sdd = sdd
    sizes = []
    for i in tqdm(range(len(bits_to_choose_from))):
    # for i in tqdm(range(1)):
        sdd = original_sdd
        if alpha is None:
            sdd = equiv_bits_indicator_clause(sdd, prev_P, P, manager, bits_to_choose_from[i], deref_old_sdd=False)
        else:
            sdd, indicator_structure = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager,
                                                         bits_already_selected + [bits_to_choose_from[i]],
                                                         deref_old_sdd=False)
        sizes.append(sdd.size())
        sdd.deref()
        manager.garbage_collect()

    # manager.auto_gc_and_minimize_on()
    sdd = original_sdd
    best_bit = bits_to_choose_from[np.argmin(sizes)]
    if alpha is None:
        sdd = equiv_bits_indicator_clause(sdd, prev_P, P, manager, best_bit, deref_old_sdd=True)
    else:
        sdd, indicator_structure = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, bits_already_selected + [best_bit],
                                                     deref_old_sdd=True)

    return sdd, best_bit, indicator_structure



def conjoin_best_bit_pair_to_sdd(sdd: SddNode, alpha: Optional[SddNode], P: List[SddNode],
                                 manager: SddManager, bits: List[SddNode], exception_bits=None,
                                 skip_optimization=False) -> tuple[SddNode, list[SddNode]]:
    """
    Conjoins the best pair of bits to the SDD (such that its size grows the least).
    """
    assert len(bits) >= 2, 'bits should have at least 2 elements'
    assert len(P) == 4, 'P should have exactly 4 indicators'

    # manager.auto_gc_and_minimize_off()
    original_sdd = sdd
    sizes = np.zeros((len(bits), len(bits))) + np.infty
    bits = [b for b in bits if b not in exception_bits]
    # for i in tqdm(range(1)):
    #     for j in range(1, 2):
    if not skip_optimization:
        for i in tqdm(range(len(bits))):
            for j in range(i+1, len(bits)):
                # try all pairs of bits
                # new_mgr = manager.copy([original_sdd, alpha, *P, *bits])
                # sdd_new = original_sdd.copy(new_mgr)
                # alpha_new = alpha.copy(new_mgr)
                # P_new = []
                # for p in P:
                #     P_new.append(p.copy(new_mgr))
                #
                # bits_new = []
                # for bit in bits:
                #     bits_new.append(bit.copy(new_mgr))

                sdd = original_sdd
                bit1, bit2 = bits[i], bits[j]
                if alpha is None:
                    sdd = equiv_bits_indicator_clause(sdd, [-bit1, bit1], P, manager, bit2, deref_old_sdd=False)
                else:
                    sdd, _ = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, [bit1, bit2], deref_old_sdd=False)
                # manager.minimize()
                sizes[i,j] = sdd.size()
                sdd.deref()
                manager.garbage_collect()

        best_i, best_j = np.unravel_index(np.argmin(sizes), sizes.shape)
    else:
        best_i, best_j = 0, 1

    best_bit_pair = [bits[best_i], bits[best_j]]


    # manager.auto_gc_and_minimize_on()
    sdd = original_sdd
    if alpha is None:
        sdd = equiv_bits_indicator_clause(sdd, [-best_bit_pair[0], -best_bit_pair[0]], P, manager, best_bit_pair[1],
                                          deref_old_sdd=True)
    else:
        sdd, _ = equiv_bits_indicator_clause_with_alpha(sdd, alpha, P, manager, best_bit_pair, deref_old_sdd=True)

    return sdd, best_bit_pair

def equiv_bits_indicator_clause(sdd: SddNode, prev_P: List[SddNode], P: List[SddNode],
                                 manager: SddManager, new_bit: SddNode, deref_old_sdd: bool = True) -> SddNode:
    assert len(prev_P) * 2 == len(P), 'P should have double the indicators of prev_P'

    for i, prev_p in tqdm(enumerate(prev_P), total=len(prev_P)):
        P_neg, P_pos = P[i * 2], P[i * 2 + 1]
        for bit, indicator in zip([~new_bit, new_bit], [P_neg, P_pos]):
            cube = prev_p & bit
            cube.ref()
            sdd = conjoin_equivalence(sdd, indicator, cube, deref_old_sdd=deref_old_sdd)
            cube.deref()

    return sdd

def conjoin_equivalence(sdd: SddNode, a: SddNode, b: SddNode, deref_old_sdd: bool = True) -> SddNode:
    """
    Conjoins the equivalence a <=> b to the SDD.
    """
    old_sdd = sdd
    sdd = sdd & (~a | b)  # a => b
    if deref_old_sdd:
        old_sdd.deref()
    sdd.ref()

    old_sdd = sdd
    sdd = sdd & (a | ~b)  # b <= a
    old_sdd.deref(); sdd.ref()

    return sdd

def equiv_bits_indicator_clause_with_alpha(sdd: SddNode, alpha: SddNode, P: List[SddNode],
                                 manager: SddManager, bits: List[SddNode], deref_old_sdd: bool = True,
                                 last_call_for_byte=False, neg_bits: Optional[List[SddNode]]=None) -> SddNode:
    indicator_structure = {}
    dnf = manager.false()
    for k, bit_assignment in enumerate(itertools.product([0, 1], repeat=len(bits))):
        indicator_structure[P[k].literal] = {}
        clause = manager.true()
        for i, (bit, bit_val) in enumerate(zip(bits, bit_assignment)):
            indicator_structure[P[k].literal][bit.literal] = bit_val

            old_clause = clause
            if bit_val == 0:
                if neg_bits is not None:
                    clause = clause & neg_bits[i]
                else:
                    clause = clause & ~bit
            else:
                clause = clause & bit
            clause.ref(); old_clause.deref()

        # if last_call_for_byte:
        #     print(P[k].literal)

        cube = alpha.condition(~P[k]) & P[k]
        cube.ref()
        old_dnf = dnf
        dnf = dnf | (clause & cube)
        dnf.ref(); old_dnf.deref(); cube.deref(); clause.deref()

    old_sdd = sdd
    sdd = sdd & dnf  # clause and cube
    if deref_old_sdd:
        old_sdd.deref()
    sdd.ref(); dnf.deref()

    return sdd, indicator_structure

def pyeda_equiv_bits_hw_indicators_pysat_cardenc(bits, hamming_weights, aux_vars, k):
    k_cnf = CardEnc.equals(lits=list(range(1, 9)), bound=k, encoding=EncType.seqcounter)
    # print(k_cnf.clauses)
    card_cnf = expr(True)
    for clause in k_cnf.clauses:
        hw_clause = expr(False)
        for c in clause:
            sign = 1 if c > 0 else -1
            # print(abs(c))
            if abs(c) <= 8:
                var = bits[abs(c) - 1] if sign == 1 else ~bits[abs(c) - 1]
            else:
                var = aux_vars[abs(c) - 9] if sign == 1 else ~aux_vars[abs(c) - 9]

            hw_clause = hw_clause | var

        card_cnf = (card_cnf & hw_clause.simplify()).simplify()

    clause_right = ~hamming_weights[k] | card_cnf
    clause_right = clause_right.simplify()
    clause_left = hamming_weights[k] | ~card_cnf
    clause_left = clause_left.simplify()

    return clause_left & clause_right

def pyeda_equiv_bits_hw_indicators_naive_enum(bits, hamming_weights, k):
    # print(k_cnf.clauses)
    possible_combinations = expr(False)
    for byte, bit_assignment in enumerate(itertools.product([0, 1], repeat=len(bits))):
        if sum(bit_assignment) != k:
            continue

        bits_and = expr(True)
        for bit, bit_val in zip(bits, bit_assignment):
            if bit_val == 0:
                bits_and = bits_and & ~bit
            else:
                bits_and = bits_and & bit

        possible_combinations = possible_combinations | bits_and.simplify()
        possible_combinations = possible_combinations.simplify()

    clause_right = ~hamming_weights[k] | possible_combinations
    clause_right = clause_right.simplify()
    clause_left = hamming_weights[k] | ~possible_combinations
    clause_left = clause_left.simplify()

    return clause_left & clause_right

def mix_single_column_own(var_list_in, var_list_out, var_list_intermediate):
    assert len(var_list_in) == 4  # there must be 4 variables in a column
    assert len(var_list_out) == 4
    assert len(var_list_intermediate) >= 13  # each column needs exactly 13 intermediates

    x, x_mix, v = var_list_in, var_list_out, var_list_intermediate

    alpha = xor_cnf(x[0], x[1], v[0]) # x01 = x0 ^ x1
    alpha = alpha & xor_cnf(x[1], x[2], v[1]) # x12 = x1 ^ x2
    alpha = alpha & xor_cnf(x[2], x[3], v[2]) # x23 = x2 ^ x3
    alpha = alpha & xor_cnf(x[3], x[0], v[3]) # x30 = x3 ^ x0

    alpha = alpha & xor_cnf(v[0], v[2], v[4]) # v[4] is the global xor
    # v[2] == x[0] ^ x[1] ^ x[2] ^ x[3]
    # v[2] is `Tmp` in tinyaes.c
    tmp = v[4]
    alpha = alpha.simplify()

    alpha = alpha & xtimes(v[0], v[5]) # v[5] = xtime[x01]
    alpha = alpha & xtimes(v[1], v[6]) # v[6] = xtime[x12]
    alpha = alpha & xtimes(v[2], v[7]) # v[7] = xtime[x23]
    alpha = alpha & xtimes(v[3], v[8]) # v[8] = xtime[x30]

    alpha = alpha & xor_cnf(tmp, v[5], v[9]) # v[9] = xtime[x01] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[6], v[10]) # v[10] = xtime[x12] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[7], v[11]) # v[11] = xtime[x23] ^ Tmp
    alpha = alpha & xor_cnf(tmp, v[8], v[12]) # v[12] = xtime[x30] ^ Tmp

    alpha = alpha & xor_cnf(x[0], v[9], x_mix[0]) # x_mix[0] = x0 ^ v[9]
    alpha = alpha & xor_cnf(x[1], v[10], x_mix[1]) # x_mix[1] = x1 ^ v[10]
    alpha = alpha & xor_cnf(x[2], v[11], x_mix[2]) # x_mix[2] = x2 ^ v[11]
    alpha = alpha & xor_cnf(x[3], v[12], x_mix[3]) # x_mix[3] = x3 ^ v[12]

    return alpha

def sdd_equiv_bits_hw_indicators_naive_enum(sdd: SddNode, alpha: SddNode, P: List[SddNode],
                                 manager: SddManager, bits: List[SddNode], deref_old_sdd: bool = True) -> SddNode:

    dnf = manager.false()
    for k in tqdm(range(0, 9)):
        possible_combinations = manager.false()
        for byte, bit_assignment in enumerate(itertools.product([0, 1], repeat=len(bits))):
            if sum(bit_assignment) != k:
                continue

            bits_and = manager.true()
            for bit, bit_val in zip(bits, bit_assignment):
                old_bits_and = bits_and
                if bit_val == 0:
                    bits_and = bits_and & ~bit
                else:
                    bits_and = bits_and & bit

                old_bits_and.deref(); bits_and.ref()

            old_sdd = possible_combinations
            possible_combinations = possible_combinations | bits_and
            old_sdd.deref(); possible_combinations.ref()

        # print(possible_combinations.size())
        cube = alpha.condition(~P[k]) & P[k]
        cube.ref()
        old_dnf = dnf
        dnf = dnf | (possible_combinations & cube)
        dnf.ref(); old_dnf.deref(); cube.deref()

    old_sdd = sdd
    sdd = sdd & dnf
    if deref_old_sdd:
        old_sdd.deref()
    sdd.ref(); dnf.deref()

    return sdd

def check_sdd_against_mixcol(sdd, models_to_check=5):
    model_gen = sdd.models()
    for _ in range(models_to_check):
        mod = next(model_gen) # get the next model from the sdd
        model_bits = [mod[k] for k in range(1, 169)]
        model_bytes = np.packbits(np.array(model_bits, dtype=np.uint8).reshape((21, 8)), bitorder='little')
        out, out_dict = mix_columns(model_bytes[:4, None])
        true_out, sdd_out = out.reshape(-1), model_bytes.reshape(-1)
        assert np.array_equal(true_out, sdd_out), f'model does not match mixcol: true {true_out} vs sdd {sdd_out}'

def check_indicator_structure(sdd, indicator_structure, models_to_check=2000):
    merged_indicator_structure = {}
    for byte_id, indicator_map in indicator_structure.items():
        for indicator_id, bit_assignment in indicator_map.items():
            merged_indicator_structure[indicator_id] = bit_assignment

    model_gen = sdd.models()
    for _ in range(models_to_check):
        mod = next(model_gen) # get the next model from the sdd
        model_bits = [mod[k] for k in range(1, max(mod.keys()) + 1)]
        for bit_id, model_bit in enumerate(model_bits):
            if bit_id < 168:
                continue # actual bits, not indicators
            if model_bit == 1:
                if (bit_id + 1) not in merged_indicator_structure.keys():
                    continue # we might model more indicators than we have in the indicator structure

                for bit_ass_id, bit_ass_val in merged_indicator_structure[bit_id + 1].items():
                    assert model_bits[bit_ass_id - 1] == bit_ass_val, (f'Indicator {bit_id + 1} is set to 1, '
                                                                   f'but the corresponding bit combination is violated: '
                                                                   f'found {model_bits=} but expected {bit_ass_val=}')

def check_sdd_indicators(sdd, byte_id, lits, bit_order, indicator_structure, models_to_check=20_000):
    lits = [x.literal for x in lits] # convert to int names
    bit_order = np.array(bit_order) - min(bit_order)
    # print(f'{lits=}')
    # print(f'{bit_order=}')

    model_gen = sdd.models()
    for i in range(models_to_check):
        mod = next(model_gen) # get the next model from the sdd
        keys = [int(k) for k in mod.keys()]
        byte_bits = [mod[k] for k in range(1, 169)][byte_id*8:byte_id*8+8]
        indicator_bits = np.array([mod[k] for k in range(1, max(keys) + 1)])[np.array(lits) - 1] # lits is 1-indexed
        assert indicator_bits.sum() == 1, 'only one indicator bit should be set'
        indicator_idx = np.argwhere(indicator_bits == 1)[0][0]
        global_indicator_idx = indicator_idx + min(lits) - 1

        if byte_id in [4, 5, 6, 7]:
            truncated_byte_bits = byte_bits[1:] # remove the first bit
        else:
            truncated_byte_bits = byte_bits[1:-1] # remove the first and last bit

        assert np.isclose(len(truncated_byte_bits), np.log(len(lits))/np.log(2))
        found_combination = False
        for k, bit_assignment in enumerate(itertools.product([0, 1], repeat=len(truncated_byte_bits))):
            if np.all(np.array(bit_assignment)[bit_order] == np.array(truncated_byte_bits)):
                found_combination = True
                if k != indicator_idx:
                    print({order: ass for order, ass in zip(bit_order, bit_assignment)})
                    print(f'Indicator Structure: {indicator_structure[byte_id][global_indicator_idx+1]}')
                    print(f'{byte_id} | {k=} != {indicator_idx=}')
                    return None
                # assert k == indicator_idx, f'{k=} != {indicator_idx=}'
                break

        assert found_combination, f'could not find combination for {byte_bits=}'

        # if i % 1000 == 0:
        #     print(f'{byte_bits=}')
        # print(f'{indicator_bits=}')

def mix_columns(x: np.ndarray, use_torch=False) -> Tuple[np.ndarray, dict]:
    # x: (4, batch_size)
    # full_mixcol: (21, batch_size)

    x1, x2, x3, x4 = x[0, :], x[1, :], x[2, :], x[3, :]
    x12 = x1 ^ x2
    x23 = x2 ^ x3
    x34 = x3 ^ x4
    x41 = x4 ^ x1

    g = x12 ^ x34

    xx12 = xtime(x12, use_torch=use_torch)
    xx23 = xtime(x23, use_torch=use_torch)
    xx34 = xtime(x34, use_torch=use_torch)
    xx41 = xtime(x41, use_torch=use_torch)

    xx12g = xx12 ^ g
    xx23g = xx23 ^ g
    xx34g = xx34 ^ g
    xx41g = xx41 ^ g

    xm1 = x1 ^ xx12g
    xm2 = x2 ^ xx23g
    xm3 = x3 ^ xx34g
    xm4 = x4 ^ xx41g

    if use_torch:
        output = torch.stack([x1, x2, x3, x4,
                           xm1, xm2, xm3, xm4,
                           x12, x23, x34, x41,
                           g,
                           xx12, xx23, xx34, xx41,
                           xx12g, xx23g, xx34g, xx41g], dim=0)
    else:
        output = np.stack([x1, x2, x3, x4,
                            xm1, xm2, xm3, xm4,
                            x12, x23, x34, x41,
                            g,
                            xx12, xx23, xx34, xx41,
                            xx12g, xx23g, xx34g, xx41g], axis=0)

    arg_dict = {
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "x12": x12, "x23": x23, "x34": x34, "x41": x41,
        "g": g,
        "xx12": xx12, "xx23": xx23, "xx34": xx34, "xx41": xx41,
        "xx12g": xx12g, "xx23g": xx23g, "xx34g": xx34g, "xx41g": xx41g,
        "xm1": xm1, "xm2": xm2, "xm3": xm3, "xm4": xm4
    }

    assert output.shape == (21, x.shape[1])
    return output, arg_dict

def xtime(x, use_torch=False):
    if use_torch:
        res = (torch.bitwise_xor(x << 1, ((x >> 7) & 1) * 0x1b)) & 0xff
    else:
        res = (np.bitwise_xor(x << 1, ((x >> 7) & 1) * 0x1b)) & 0xff

    return res

