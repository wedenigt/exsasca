import time
import numpy as np
from exsasca.exhaustive_inference import exhaustive_inference
from exsasca.utils.aes import xtime
from exsasca.utils.sdd import load_sdd, read_pickle_file, indicator_structure_to_weight_map, xor_permute_pmf, \
    get_mc_vars, pre_merge_pmfs
import torch

device = 'cpu' if torch.cuda.is_available() else 'cpu'


def check_sdd_correct(g=255, base_path=None):
    base_path = f'../compilation/compiled_sdd_g_{g}' if base_path is None else base_path

    # Load SDD from disk -- this takes about 1 minute since the SDD is large
    print('Loading SDD from disk...\nThis will take a minute.')
    sdd, manager = load_sdd(sdd_filename=f'{base_path}/out.sdd',
                            vtree_filename=f'{base_path}/out.vtree')
    sdd.ref()
    manager.auto_gc_and_minimize_off()
    print(f'SDD Size: {sdd.size()}')

    np.random.seed(0)
    pmfs = np.random.rand(1, 256, 21)  # randomly generate 21 PMFs (one for each byte)
    pmfs /= np.sum(pmfs, axis=1, keepdims=True)

    new_g = 1  # the new g value (coming from e.g. g=255)
    pmfs = pre_merge_pmfs(pmfs, g=new_g, merge_x12_x34=True)
    pmfs = change_weights_old_g_to_new_g(pmfs, old_g=g, new_g=new_g)
    pmfs = pmfs[0, :, :]
    p_g = pmfs[new_g, 12]  # probability of g=1, which we do not take into account during WMC

    exhaustive_res, _ = exhaustive_inference(pmfs, g_range=[new_g])  # the true result

    print('Computing weighted model counting (WMC) solutions for random byte values...')
    for x_idx in [0, 1, 2, 3]:
        x_value = np.random.randint(0, 256) # check random byte value
        sdd_prob = p_g * wmc(pmfs, sdd, manager, x_idx=x_idx, xi=x_value, old_g=g, new_g=new_g, base_path=base_path)
        true_prob = exhaustive_res[0, x_idx, x_value]
        assert np.isclose(sdd_prob, true_prob,
                          atol=1e-5), f'x_{x_idx}={x_value} does not match: {sdd_prob} vs {true_prob}'

    return True


def wmc(pmfs, sdd, manager, x_idx, xi, old_g=None, new_g=None, base_path='../compilation/compiled_sdd_g_255'):
    assert x_idx in [0, 1, 2, 3], 'x_idx must be in [0, 1, 2, 3]'
    if x_idx == 3:
        assert old_g is not None and new_g is not None, 'old_g and new_g must be provided for x3'
        xi = np.bitwise_xor(xi, np.bitwise_xor(old_g, new_g))

    xi_bits = np.unpackbits(np.array([xi]).astype(np.uint8), bitorder='little')

    # check model
    indicator_structure = read_pickle_file(f'{base_path}/indicator_structure.pkl')
    weight_map = indicator_structure_to_weight_map(manager, indicator_structure, pmfs)

    start = time.time()
    wmc = sdd.wmc(log_mode=True)
    for sdd_indicator_node, weight in weight_map.items():
        wmc.set_literal_weight(sdd_indicator_node, np.log(weight))

    # condition on x1
    for i in range(8):
        xi_bit_literal = manager.literal(i + 1 + x_idx * 8)
        positive_weight = -np.inf if xi_bits[i] == 0 else np.log(1)
        negative_weight = np.log(1) if xi_bits[i] == 0 else -np.inf

        wmc.set_literal_weight(xi_bit_literal, positive_weight)
        wmc.set_literal_weight(-xi_bit_literal, negative_weight)

    w = wmc.propagate()
    end = time.time()

    prob = np.exp(w) / 2 ** 8
    return prob


def change_weights_old_g_to_new_g(pmfs, old_g, new_g):
    xor = np.bitwise_xor
    mc_vars = get_mc_vars()

    for var in ['xm3']:
        xor_val = xor(xtime(xor(old_g, new_g)), xor(old_g, new_g))
        pmfs[:, :, mc_vars.index(var)] = xor_permute_pmf(pmfs[:, :, mc_vars.index(var)], xor_val=xor_val)

    for var in ['xm4']:
        xor_val = xtime(xor(old_g, new_g))
        pmfs[:, :, mc_vars.index(var)] = xor_permute_pmf(pmfs[:, :, mc_vars.index(var)], xor_val=xor_val)

    for var in ['x4', 'xm1', 'xm2']:
        xor_val = xor(old_g, new_g)
        pmfs[:, :, mc_vars.index(var)] = xor_permute_pmf(pmfs[:, :, mc_vars.index(var)], xor_val=xor_val)

    return pmfs


if __name__ == '__main__':
    check_sdd_correct(
        base_path='../compilation/compiled_sdd_g_255')  # if this does not raise an exception, the SDD and WMC procedure are correct
    print('Passed SDD correctness check.')
