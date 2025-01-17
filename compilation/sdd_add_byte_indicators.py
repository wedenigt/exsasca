from pathlib import Path
import argparse
import pickle
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm
import itertools
from pysdd.sdd import SddManager, Vtree, SddNode
from graphviz import Source
import utils
import os
import matplotlib.pyplot as plt
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compile SDD for inference in MixColumns factor graph (WMC).'
    )
    parser.add_argument(
        "--g",
        type=int,
        default=0,
        help="Condition g on a particular byte value (default 0). Must be in [0, 255].",
    )

    parser.add_argument(
        "--first-last",
        type=int,
        default=0,
        help="Condition first and last bits of all 4 (or 3, if --g) inputs. "
             "Must be in [0, 63] if --g and in [0, 255] otherwise.",
    )

    parser.add_argument(
        "--sdd-name",
        type=str,
        default="out",
        help="Sdd name (without .sdd). Vtree is assumed to have the same name with extension .vtree (default: %(default)s).",
    )

    return parser.parse_args()

def compute_alpha(manager, P, k):
    # we will reuse alpha
    alpha = manager.true()
    for p in P:
        old_alpha = alpha
        alpha = alpha & ~p
        alpha.ref()
        old_alpha.deref()

    return alpha

def write_indicator_structure(indicator_structure, base_path='final_compilation'):
    with open(f'{base_path}/indicator_structure.pkl', 'wb') as f:
        pickle.dump(indicator_structure, f)

def write_indicators_per_byte(indicator_dict, curr_byte_idx, base_path='final_compilation'):
    serialized_indicator_dict = {}
    for byte_idx, P_dict in indicator_dict.items():
        serialized_P_dict = {}
        for k, P in P_dict.items():
            serialized_P_dict[k] = [p.literal for p in P]
        serialized_indicator_dict[byte_idx] = serialized_P_dict

    with open(f'{base_path}/indicator_to_lit_map_{curr_byte_idx}.pkl', 'wb') as f:
        pickle.dump(serialized_indicator_dict, f)

def write_selected_bits_per_byte(selected_bits, curr_byte_idx, base_path='final_compilation'):
    serialized_selected_bits = {}
    for byte_idx, bits in selected_bits.items():
        serialized_selected_bits[byte_idx] = [var.literal for var in bits]

    with open(f'{base_path}/selected_bits_{curr_byte_idx}.pkl', 'wb') as f:
        pickle.dump(serialized_selected_bits, f)

def print_gc(manager):
    print(f"  live sdd size = {manager.live_size()}")
    print(f"  dead sdd size = {manager.dead_size()}")
    print('----------')

def m(sdd, n=1):
    gen = sdd.models()
    for _ in range(n):
        d = next(gen)
        l = [k for k in d.keys() if d[k] == 1]
        l.sort()
        print(l)

def conjoin_xor(sdd, a_list, b_list, c_list):
    for i in tqdm(range(8)):
        # print_gc()
        a, b, c = a_list[i], b_list[i], c_list[i]
        c1, c2, c3, c4 = (~a | ~b | ~c), (~a | b | c), (a | ~b | c), (a | b | ~c)
        c1.ref(); c2.ref(); c3.ref(); c4.ref()
        for c in [c1, c2, c3, c4]:
            old_sdd = sdd
            sdd = sdd & c
            sdd.ref()
            old_sdd.deref()

        c1.deref(); c2.deref(); c3.deref(); c4.deref()

    return sdd

def load_sdd(vtree_filename: str, sdd_filename: str, num_bytes: int, all_bits=None,
             custom_vtree_for_sorting=None, use_lca_sorting=True, fixed_vtree_name=None):
    vtree = Vtree.from_file(vtree_filename.encode())
    manager = SddManager.from_vtree(vtree)
    manager.auto_gc_and_minimize_on()

    sdd = manager.read_sdd_file(sdd_filename.encode())
    sdd.ref()

    if all_bits is None:
        manager.minimize() # initial minimization
        all_bits_new = [[manager.literal(n * 8 + i) for i in range(1, 9)] for n in range(num_bytes)]
        vtree_for_sorting = manager.vtree() if custom_vtree_for_sorting is None else custom_vtree_for_sorting
        if use_lca_sorting:
            all_bits_new = utils.sort_bits_by_vtree_lca(vtree_for_sorting, all_bits_new)
    else:
        # copy old bits to new manager
        all_bits_new = [[manager.literal(var.literal) for var in all_bits[byte_idx]] for byte_idx in range(num_bytes)]

    # assert sdd.model_count() == 2 ** 32
    utils.check_sdd_against_mixcol(sdd)
    return sdd, manager, all_bits_new

def materialize_byte_result(sdd: SddNode, manager: SddManager, byte_idx, selected_bits_per_byte, indicators_per_byte):
    sdd_name = f'final_compilation/byte_{byte_idx}.sdd'
    vtree_name = f'final_compilation/byte_{byte_idx}.vtree'

    sdd.save(sdd_name.encode())
    manager.vtree().save(vtree_name.encode())
    write_indicators_per_byte(indicators_per_byte, byte_idx)
    write_selected_bits_per_byte(selected_bits_per_byte, byte_idx)

    return sdd_name, vtree_name

def determine_nasty_bits(sdd, all_bits, num_bits=8, at_most_one_bit_per_byte=False, plot_min_sizes=False):
    nasty_bits = []
    nasty_bit_ids_per_byte_id = defaultdict(list)
    nasty_bits_per_byte_id = defaultdict(list)
    min_sizes = []

    for round_idx in range(num_bits):
        all_sizes = []
        for byte_idx, byte in enumerate(all_bits):

            sizes = []
            for bit in byte:
                if at_most_one_bit_per_byte and byte_idx in nasty_bits_per_byte_id.keys():
                    sizes.append(np.infty)  # skip bits in bytes where we have already conditioned on a bit
                else:
                    sizes.append(sdd.condition(~bit).size())
            all_sizes.append(sizes)

        all_sizes = np.array(all_sizes)
        best_byte_idx, best_bit_idx = np.unravel_index(np.argmin(all_sizes), all_sizes.shape)
        bit_to_condition_on = all_bits[best_byte_idx][best_bit_idx]

        old_sdd = sdd
        sdd = sdd.condition(~bit_to_condition_on)
        sdd.ref()
        if round_idx > 0:
            old_sdd.deref()  # don't deref original one

        nasty_bits.append(bit_to_condition_on)
        nasty_bit_ids_per_byte_id[best_byte_idx].append(best_bit_idx)
        nasty_bits_per_byte_id[best_byte_idx].append(bit_to_condition_on)
        min_size = np.min(all_sizes)
        print('min size', min_size)
        min_sizes.append(min_size)

    if plot_min_sizes:
        plt.plot(min_sizes)
        plt.show()

    return nasty_bits, nasty_bit_ids_per_byte_id, nasty_bits_per_byte_id

def load_empty_sdd(num_bytes: int, vtree_type='balanced'):
    # try var order
    vtree = Vtree(var_count=num_bytes*8, vtree_type=vtree_type)
    manager = SddManager.from_vtree(vtree)
    manager.auto_gc_and_minimize_on()

    sdd = manager.true()
    sdd.ref()

    all_bits = [[manager.literal(n * 8 + i) for i in range(1, 9)] for n in range(num_bytes)]
    return sdd, manager, all_bits

def main(experiment_folder='compiled_psdds', num_bytes=21, num_nasty_bits=8, hw_indicators=False,
         condition_on_g=False, condition_first_last=False, skip_optimization=False,
         use_lca_sorting=True, g_conditioning=None, first_last_conditioning=None, byte_indicators=False,
         is_load_empty_sdd=False, sdd_name='out', fixed_byte_order=None, pick_bytes_greedily=False,
         fixed_vtree_name=None):

    assert not (use_lca_sorting and fixed_byte_order), 'use_lca_sorting and fixed_byte_order cannot be used together'
    assert not ((use_lca_sorting and pick_bytes_greedily) or (fixed_byte_order and pick_bytes_greedily)), \
        'if picked bytes greedily, then we cannot use lca sorting or fixed byte order'

    # create folder 'final_compilation' if it doesn't exist
    if not os.path.exists('final_compilation'):
        os.mkdir('final_compilation')

    if is_load_empty_sdd:
        print('WARNING: loading empty sdd, this is just for testing!')
        sdd, manager, all_bits = load_empty_sdd(num_bytes=num_bytes, vtree_type='balanced')
    else:
        sdd, manager, all_bits = load_sdd(f'{sdd_name}.vtree',
                                          f'{sdd_name}.sdd',
                                          num_bytes=num_bytes, all_bits=None,
                                          use_lca_sorting=use_lca_sorting and not condition_first_last and not condition_on_g,
                                          fixed_vtree_name=fixed_vtree_name)

    sdd.ref() # ref the original sdd twice such that we don't delete it during conditioning

    original_sdd = sdd
    nasty_bits, nasty_bit_ids_per_byte_id, nasty_bits_per_byte_id = determine_nasty_bits(sdd, all_bits,
                                                                                         num_bits=num_nasty_bits,
                                                                                         at_most_one_bit_per_byte=True,
                                                                                         plot_min_sizes=False)

    base_path = f'final_compilation/cond_g/g_{g_conditioning}'
    os.makedirs(base_path, exist_ok=True)

    print(nasty_bit_ids_per_byte_id)
    print('original size', original_sdd.size())
    for nasty_bit_assignment in itertools.product([0, 1], repeat=num_nasty_bits):
        sdd = original_sdd
        for lit, assignment in zip(nasty_bits, nasty_bit_assignment):
            old_sdd = sdd
            sdd = sdd.condition(lit) if assignment == 1 else sdd.condition(-lit)
            sdd.ref(); old_sdd.deref()

        # custom_vtree_for_sorting = manager.vtree()
        # sdd, manager, all_bits = load_sdd('final_compilation/byte_20.vtree',
        #                                   'final_compilation/byte_20.sdd', num_bytes=NUM_BYTES, all_bits=None,
        #                                   custom_vtree_for_sorting=custom_vtree_for_sorting)

        # print(f'START: size: {sdd.size()}, ref count: {sdd.ref_count()}')

        use_alpha = True
        selected_bits_per_byte = {}
        indicators_per_byte = {}

        if condition_on_g:
            # assert not use_lca_sorting
            g_cond_bits = np.unpackbits(np.array(g_conditioning).astype(np.uint8), bitorder='little')
            # condition on g
            g_bits = all_bits[12]
            assert g_bits[0].literal == 97, 'first bit of g is not 97'
            print(f'g_bit ids: {[b.literal for b in g_bits]}')
            for i, bit in enumerate(g_bits):  # [-4:]:
                old_sdd = sdd
                if g_cond_bits[i] == 1:
                    sdd = sdd.condition(bit)
                else:
                    sdd = sdd.condition(~bit)
                sdd.ref(); old_sdd.deref()
                manager.minimize()

        if condition_first_last:
            # assert not use_lca_sorting
            first_last_cond_bits = np.unpackbits(np.array(first_last_conditioning).astype(np.uint8), bitorder='little')
            num_bits = 3 if condition_on_g else 4
            # num_bits = 4
            first_last_bits = [all_bits[i][0] for i in range(num_bits)] + [all_bits[i][-1] for i in range(num_bits)]
            # first_last_bits += [all_bits[4+i][-1] for i in range(4)]
            # first_last_bits += [all_bits[8+i][0] for i in range(4)] + [all_bits[8+i][-1] for i in range(4)]
            # first_last_bits += [all_bits[13+i][0] for i in range(4)] + [all_bits[13+i][1] for i in range(4)]
            # first_last_bits += [all_bits[17+i][0] for i in range(4)] + [all_bits[17+i][1] for i in range(4)]
            # first_last_bits = [all_bits[i][0] for i in range(num_bits)]
            print(f'first_last bit ids: {[b.literal for b in first_last_bits]}')
            for i, bit in enumerate(first_last_bits):
                old_sdd = sdd
                if (len(first_last_cond_bits) > i and first_last_cond_bits[i] == 1):
                    sdd = sdd.condition(bit)
                else:
                    sdd = sdd.condition(~bit)
                sdd.ref(); old_sdd.deref()
                manager.minimize()

        print('conditioned sdd size', sdd.size())
        # utils.check_sdd_against_mixcol(sdd)


        # for byte_idx in range(2, 21):  # test
        # all_bits = [all_bits[0], all_bits[1], all_bits[2], all_bits[3],
        #             all_bits[8], all_bits[9],
        #             all_bits[4], all_bits[5], all_bits[6], all_bits[7]]

        if fixed_byte_order is not None:
            print('using fixed byte order')
            all_bits = [all_bits[i] for i in fixed_byte_order]

        byte_count = 0
        indicator_structure = {}
        for _ in range(21):
            if use_lca_sorting:
                all_bits = utils.sort_bits_by_vtree_lca(manager.vtree(), all_bits, out_path=Path(base_path))

            bits = all_bits[0]  # take best byte to conjoin
            all_bits = all_bits[1:]  # remove best byte

            bits_literals = [b.literal for b in bits]
            assert (min(bits_literals) - 1) % 8 == 0, 'bits are not byte aligned'
            true_byte_idx = int((min(bits_literals) - 1) // 8)

            nasty_bit_ids_for_byte = nasty_bit_ids_per_byte_id[byte_count] if byte_count in nasty_bit_ids_per_byte_id.keys() \
                else []

            nasty_bits_for_byte = nasty_bits_per_byte_id[byte_count] if byte_count in nasty_bits_per_byte_id.keys() \
                else []

            # if bits[0].literal == 97:
            #     print('current byte is g, condition bits on all 0')
            #     # print('condition last four bit to 0')
            #     for bit in bits:#[-4:]:
            #         old_sdd = sdd
            #         sdd = sdd.condition(~bit)
            #         sdd.ref(); old_sdd.deref()
            #         manager.minimize()

                # max_num_bits_in_combination = 4 - len(nasty_bits_for_byte)
            # else:

            # set bits that follow deterministically from the conditional bits (aka manual unit clause propagation)
            if condition_on_g:
                if bits[0].literal >= 97 or bits[0].literal == 81 or bits[0].literal == 89:
                # if bits[0].literal >= (97 - 16):
                    continue # global byte we condition on or unused bytes (also x34 and x41 are unused now)

            if condition_first_last:
                if bits[0].literal <= 32 or (bits[0].literal > 64 and bits[0].literal <= 64+32): # input bytes or x12, x23, x34, x41
                    nasty_bits_for_byte = [bits[0], bits[-1]]
                    nasty_bit_ids_for_byte = [bits[0].literal, bits[-1].literal]
                else:
                    nasty_bits_for_byte = [bits[0]]
                    nasty_bit_ids_for_byte = [bits[0].literal]

            max_num_bits_in_combination = 8 - len(nasty_bits_for_byte)

            print('-------------------------------------------------------')
            print(f'BYTE IDX: {byte_count} | nasty bits to condition on: {nasty_bit_ids_for_byte}')
            print(f'bit ids: {[b.literal for b in bits]}')
            selected_bits = []
            bits_lca = utils.lca_of_literals_sdd_vtree(manager.vtree(), [b for b in bits if b.literal not in nasty_bit_ids_for_byte])
            P_dict = {}
            if hw_indicators:
                iter_range = [np.log(9)/np.log(2)]
            elif byte_indicators:
                iter_range = [max_num_bits_in_combination]
            else:
                iter_range = range(2, max_num_bits_in_combination+1)

            indicator_structure_for_byte = None
            for k in iter_range:  # we investigate combinations of k bits
                # (we skip k=1 because we can use the weight of the bit itself to encode this)
                print(k)
                P = []  # new variables
                print(f'lca: {bits_lca.id}')
                for i in range(int(2 ** k)):
                    manager.add_var_after(bits_lca.id) # TODO: after or before?
                    # manager.add_var_after(bits[0].literal)  # TODO: after or before?
                    # manager.add_var_before(bits[0].literal)  # TODO: after or before?
                    P.append(manager.literal(manager.var_count()))

                alpha = compute_alpha(manager, P, k) if use_alpha else None

                if hw_indicators:
                    sdd = utils.sdd_equiv_bits_hw_indicators_naive_enum(sdd, alpha, P, manager, bits, deref_old_sdd=True)
                else:
                    if k == 2:
                        sdd, best_bit_pair = utils.conjoin_best_bit_pair_to_sdd(sdd, alpha, P, manager, bits,
                                                                                exception_bits=nasty_bits_for_byte,
                                                                                skip_optimization=skip_optimization)
                        selected_bits += best_bit_pair
                        P_dict[1] = [-best_bit_pair[0],
                                     best_bit_pair[0]]  # first indicator is the first bit (we get that for free)
                    else:
                        if byte_indicators:
                            # let's pretend like we have already selected the first 7 bits and want to add the 8th
                            assert skip_optimization, 'byte indicators only work with skip_optimization'
                            sdd, best_bit, indicator_structure_for_byte = utils.conjoin_best_bit_to_sdd(sdd, alpha, P, None, manager, bits_already_selected=[],
                                                                          bits=bits, exception_bit_ids=nasty_bits_for_byte,
                                                                          skip_optimization=skip_optimization, byte_indicators=True)
                        else:
                            last_call_for_byte = (k == max_num_bits_in_combination)
                            sdd, best_bit, indicator_structure_for_byte  = utils.conjoin_best_bit_to_sdd(sdd, alpha, P,
                                                                          P_dict[k - 1] if not skip_optimization else None,
                                                                          manager, selected_bits,
                                                                          bits, exception_bit_ids=nasty_bits_for_byte,
                                                                          skip_optimization=skip_optimization,
                                                                          last_call_for_byte=last_call_for_byte)
                            if best_bit is not None:
                                selected_bits.append(best_bit)

                P_dict[k] = P

                # alpha.deref()
                # manager.garbage_collect()
                # manager.minimize()  # minimize
                # utils.check_sdd_against_mixcol(sdd)
                print(f'{sdd.size():,}')
                # assert sdd.model_count() == 2 ** 32
                # m(sdd, 5)

            assert indicator_structure_for_byte is not None, 'indicator_structure_for_byte is None'
            indicator_structure[true_byte_idx] = indicator_structure_for_byte

            selected_bits_per_byte[true_byte_idx] = selected_bits
            indicators_per_byte[true_byte_idx] = P_dict
            # assert sdd.model_count() == 2 ** 32
            # print_gc()
            print(f'FINAL: size: {sdd.size():,}, ref count: {sdd.ref_count()}')
            # Source(manager.vtree().dot()).render('vtree_cond_opt', view=False)
            sdd_name, vtree_name = materialize_byte_result(sdd, manager, true_byte_idx, selected_bits_per_byte,
                                                           indicators_per_byte)
            byte_count += 1
            # sdd, manager, all_bits = load_sdd(vtree_name, sdd_name, num_bytes=num_bytes, all_bits=all_bits)

        # save sdd
        save_name = 'sdd_with_byte_indicators'
        sdd.save(f'{save_name}.sdd'.encode())
        # save vtree
        manager.vtree().save(f'{save_name}.vtree'.encode())

        if condition_on_g or condition_first_last:
            sdd.save(f'{base_path}/out.sdd'.encode())
            manager.vtree().save(f'{base_path}/out.vtree'.encode())
            write_indicator_structure(indicator_structure, base_path=base_path)
            for byte_count in selected_bits_per_byte.keys():
                write_indicators_per_byte(indicators_per_byte, byte_count, base_path=base_path)
                write_selected_bits_per_byte(selected_bits_per_byte, byte_count, base_path=base_path)


        wmc = sdd.wmc(log_mode=True)
        w = wmc.propagate()
        print(f"Model count: {np.exp(w)}")

        # for byte_idx in range(21):
        #     for k, P in indicators_per_byte[byte_idx].items():
        #         for p in P:
        #             wmc.set_literal_weight(p.literal, np.log(0.5))
        #
        #         if k == 1:
        #             wmc.set_literal_weight(-P[0].literal, np.log(0.5))
        #
        # w = wmc.propagate()
        # print(f"Weighted model count: {np.exp(w)}")


if __name__ == '__main__':
    np.random.seed(1337)

    settings = parse_args()

    CONDITION_FIRST_LAST = False
    CONDITION_ON_G = True
    SKIP_OPTIMIZATION = True
    if CONDITION_FIRST_LAST and CONDITION_ON_G:
        print('WMC must be divided by 2**14 (since 14 bits are actually conditioned on in the SDD)')

    # EXPERIMENT_FOLDER = 'cond_g_and_first_last'
    EXPERIMENT_FOLDER = 'no_cond'
    USE_LCA_SORTING = True
    BYTE_INDICATORS = True

    LOAD_EMPTY_SDD = False # True is just for testing

    # FIXED_BYTE_ORDER = [2, 5, 0, 7, 1, 9, 8, 3, 4, 6] # extracted from balanced order
    # FIXED_BYTE_ORDER = [0, 1, 8, 4, 2, 9, 5, 3, 6, 7] # whiteboard order
    FIXED_BYTE_ORDER = None

    # FIXED_VTREE_NAME = 'out.vtree'
    FIXED_VTREE_NAME = None

    PICK_BYTES_GREEDILY = False

    start = time.time()

    main(experiment_folder=EXPERIMENT_FOLDER,
         num_nasty_bits=0,
         hw_indicators=False,
         condition_first_last=CONDITION_FIRST_LAST,
         condition_on_g=CONDITION_ON_G,
         skip_optimization=SKIP_OPTIMIZATION,
         use_lca_sorting=USE_LCA_SORTING,
         g_conditioning=settings.g,
         first_last_conditioning=settings.first_last,
         byte_indicators=BYTE_INDICATORS,
         is_load_empty_sdd=LOAD_EMPTY_SDD,
         sdd_name=settings.sdd_name,
         fixed_byte_order=FIXED_BYTE_ORDER,
         pick_bytes_greedily=PICK_BYTES_GREEDILY,
         fixed_vtree_name=FIXED_VTREE_NAME)

    end = time.time()
    print(f'compilation took {end - start} seconds')


