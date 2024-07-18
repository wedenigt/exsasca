import os
import pickle
from enum import Enum

import numpy as np
from scipy.special import logsumexp
from pathlib import Path
import torch
from tqdm import tqdm

from exsasca.sasca import sasca_mixcol_inference
from exsasca.utils.aes import mix_columns, sbox
from exsasca.utils.pmf import add_noise
from exsasca.utils.sdd import pre_merge_pmfs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class InferenceMethod(Enum):
    EXHAUSTIVE = 1
    SASCA = 2


def exact_loop_solution_conditioned_on_g(pmfs, g, first_last=None,
                                         load_input_matrix_from_file=True, device='cpu', just_max=False):
    assert 0 <= g and g < 256
    assert len(pmfs.shape) == 3, 'pmfs must be of shape (B, 256, 21)'
    batch_size = pmfs.shape[0]

    condition_first_last = first_last is not None
    base = Path('../compilation/np_matrices')
    input_matrix_filename = base / f'mixcol_inputs_g_{g}.npz' if not condition_first_last else (
            base / f'mixcol_inputs_g_{g}_fl_{first_last}.npz')
    first_bits, last_bits = None, None

    if condition_first_last:
        first_bits, last_bits = compute_first_last_bits(g, first_last)
        mixcol_inputs = np.zeros((4, 2 ** 18), dtype=np.uint8)
    else:
        mixcol_inputs = np.zeros((4, 2 ** 24), dtype=np.uint8)
    # x4_indices = np.zeros(2**16, dtype=np.uint32)

    if load_input_matrix_from_file:
        print('Loading input matrix from file...')
        mixcol_inputs = np.load(input_matrix_filename)['mixcol_inputs']
        mixcol_inputs = torch.from_numpy(mixcol_inputs).to(device)
    else:
        print('Building input matrix...')
        count = 0
        # for x1 in tqdm(range(256)):
        for x1 in range(256):
            for x2 in range(256):
                for x3 in range(256):
                    x4 = x1 ^ x2 ^ x3 ^ g  # x4 is uniquely determined by x1, x2, x3, g

                    if condition_first_last:
                        x = [x1, x2, x3, x4]
                        incompatible = False
                        for i in range(4):
                            x_bits = np.unpackbits(np.array([x[i]]).astype(np.uint8), bitorder='little')
                            if x_bits[0] != first_bits[i] or x_bits[-1] != last_bits[i]:
                                incompatible = True
                                break

                        if incompatible:
                            continue

                    # idx = x1 * 2**16 + x2 * 2**8 + x3
                    # x4_indices[x4].append(idx)
                    mixcol_inputs[:, count] = [x1, x2, x3, x4]
                    count += 1
                    # break
        np.savez(input_matrix_filename, mixcol_inputs=mixcol_inputs)
        mixcol_inputs = torch.from_numpy(mixcol_inputs).to(device)

    final_dist = torch.zeros((batch_size, 4, 256), device=device).double() - float('inf')

    print('Computing MixColumns...')
    out, _ = mix_columns(mixcol_inputs, use_torch=True)  # out is (21, 2**24)
    print('done')
    probs = torch.zeros((batch_size, 10, out.shape[1])).float()  # probs need to be floats

    print('Indexing PMFs...')
    pmfs = torch.log(pmfs)  # pmfs is expected to be (B, 256, 21) where B is the batch size
    for i in tqdm(range(10)):
        probs[:, i, :] = pmfs[:, out[i].long(), i]  # replace outputs x with pmf(x) for the entire batch

    # res = np.prod(probs, axis=0) # this includes p(g=g), you have to divide by it to get the conditional
    res = torch.sum(probs, dim=1)  # product is sum in log space, output: (B, 2**24)
    print('Compute argmax p(x1, x2, x3, x4)...')
    max_logprobs, argmax = torch.max(res, dim=1)
    # max_probs = torch.exp(max_logprobs) # (B,)
    argmax_xs = mixcol_inputs[:, argmax].T  # (B, 4)

    if just_max:
        return None, max_logprobs, argmax_xs

    print('Computing p(x_4)...')
    # final_dist[:, 3, i] = torch.logsumexp(res[:, mixcol_inputs[-1, :] == i], dim=1)

    # #------
    # # Create a tensor of indices from 0 to 255
    # indices = torch.arange(256, device=res.device)
    #
    # # Expand dimensions of 'indices' to match 'mixcol_inputs' for broadcasting
    # expanded_indices = indices.view(1, 1, -1)
    #
    # # Compare 'mixcol_inputs' with 'expanded_indices' and create a boolean mask
    # mask = mixcol_inputs[-1, :].unsqueeze(-1) == expanded_indices
    #
    # # Use this mask to select the relevant elements from 'res' and compute logsumexp
    # final_dist[:, 3, :] = torch.logsumexp(res[:, mask], dim=1)
    # #------
    #
    for i in tqdm(range(256)):
        if condition_first_last:
            for j in range(4):
                sliced = res[np.argwhere(mixcol_inputs[j, :] == i)]
                if len(sliced) > 0:
                    final_dist[j, i] = logsumexp(sliced)
        else:
            final_dist[:, 3, i] = torch.logsumexp(res[:, mixcol_inputs[-1, :] == i], dim=1)

    if not condition_first_last:
        res = res.reshape((batch_size, 256, 256, 256))
        print('Computing p(x_1),...,p(x_3)')
        for byte_idx in tqdm(range(3)):
            axis_to_sum = tuple([i + 1 for i in range(3) if i != byte_idx])
            # final_dist[byte_idx, :] = np.sum(res, axis=axis_to_sum).squeeze()
            final_dist[:, byte_idx, :] = torch.logsumexp(res, dim=axis_to_sum)  # .squeeze()

    # final_dist -= logsumexp(final_dist, axis=1, keepdims=True) # do not normalize

    dist_probs = torch.exp(final_dist)
    # np.savez(f'naive_loop_dist_g_{g}.npz', dist_probs=dist_probs)

    return dist_probs, max_logprobs, argmax_xs


def compute_first_last_bits(g, first_last):
    g_bits = np.unpackbits(np.array([g]).astype(np.uint8), bitorder='little')
    # first/last conditioning
    first_last_cond_bits = np.unpackbits(np.array(first_last).astype(np.uint8), bitorder='little')
    first_bits = first_last_cond_bits[:3]
    first_bit_4 = np.bitwise_xor(np.bitwise_xor(np.bitwise_xor(first_bits[0], first_bits[1]), first_bits[2]), g_bits[0])
    first_bits = np.concatenate((first_bits, [first_bit_4]))

    last_bits = first_last_cond_bits[3:6]
    last_bit_4 = np.bitwise_xor(np.bitwise_xor(np.bitwise_xor(last_bits[0], last_bits[1]), last_bits[2]), g_bits[-1])
    last_bits = np.concatenate((last_bits, [last_bit_4]))

    return first_bits, last_bits


def exhaustive_inference(pmfs: np.ndarray, g_range=range(256), just_max=False):
    """
    pmfs: (B, 256, 21)
    g_range: range(256), anything else is for testing purposes
    """
    original_pmfs = pmfs.copy()
    dists = []
    max_probs_all, argmax_xs_all = [], []
    for g in tqdm(g_range):
        pmfs = original_pmfs.copy()
        pmfs = pre_merge_pmfs(pmfs, g=g, merge_x12_x34=True)
        pmfs = torch.from_numpy(pmfs).to(device)
        dist, max_logprobs, argmax_xs = exact_loop_solution_conditioned_on_g(pmfs, g=g,
                                                                             load_input_matrix_from_file=True,
                                                                             device=device, just_max=just_max)
        max_probs_all.append((pmfs[:, g, 12].log() + max_logprobs).cpu().numpy())
        argmax_xs_all.append(argmax_xs.cpu().numpy())
        if not just_max:
            # dist is (B, 4, 256)
            dist = (pmfs[:, g, 12] * dist.permute(2, 1, 0)).permute(2, 1,
                                                                    0)  # * p(G=g), broadcasted over batch dimension
            dists.append(dist.cpu().numpy())

    argmax_xs_all = np.stack(argmax_xs_all)  # (256, B, 4)
    argmax_g = np.stack(max_probs_all).argmax(axis=0)  # (B,)
    final_argmax_xs = []
    for batch_idx, g in enumerate(argmax_g):
        final_argmax_xs.append(argmax_xs_all[g, batch_idx, :])

    final_argmax_xs = np.stack(final_argmax_xs)  # (B, 4)
    if not just_max:
        final_dists = np.stack(dists).sum(axis=0)
    else:
        final_dists = None

    return final_dists, final_argmax_xs

def exhaustive_inference_mixcol_pmfs(inference_method: InferenceMethod, mc_block, num_traces=32, testing=True,
                                     g_range=range(256), noise_alpha=0.0, num_bp_iterations=50, load_sasca=False,
                                     batch_idx=0, load_exact=False, just_max=False):
    if 255 not in g_range:
        print('WARNING: not using full g_range. This should only happen when testing this code.')

    assert mc_block in [0,1,2,3], 'mc_block must be in [0, 1, 2, 3]'

    pmfs, y_pmfs, labels = load_lda_pmfs(num_traces, batch_idx, testing, mc_block)
    # plot_pmf(pmfs[0, :, 0])
    pmfs = add_noise(pmfs=pmfs.transpose((0, 2, 1)), alpha=noise_alpha).transpose((0, 2, 1))
    y_pmfs = add_noise(pmfs=y_pmfs.transpose((0, 2, 1)), alpha=noise_alpha)

    # eps = 1e-2
    # print(f'DEBUG: Sparsifying PMFs: {eps=}')
    # for i in range(pmfs.shape[0]):
    #     for row_idx in range(pmfs.shape[1]):
    #         if row_idx > 3:  # sparsify all input dists
    #             continue
    #         pmfs[i, :, row_idx], top_k = sparsify_pmf(pmfs[i, :, row_idx], p=1 - eps)

    # plot_pmf(pmfs[0, :, 0])
    pmfs = pmfs[0:1] # batch size 1
    if inference_method == InferenceMethod.EXHAUSTIVE:
        out_folder = f'../experiments/saved_pmfs/{str.lower(inference_method.name)}'
        merged_pmfs = message_pass_to_x(pmfs[:, :, :4].transpose((0, 2, 1)), y_pmfs) # first pass in messages from y to x (s.t. we can compute the argmax)
        pmfs[:, :, :4] = merged_pmfs.transpose((0, 2, 1))
        if load_exact:
            x1_4_pmfs = np.load(f'{out_folder}/val/mc_{mc_block}_x1_4_pmfs_testing_{testing}_alpha_{noise_alpha}_batch_{batch_idx}.npz')['x1_4_pmfs']
        else:
            x1_4_pmfs, argmax_xs = exhaustive_inference(pmfs, g_range=g_range, just_max=just_max) # (B, 4, 256) and (B, 4)
    elif inference_method == InferenceMethod.SASCA:
        out_folder = f'../experiments/saved_pmfs/{str.lower(inference_method.name)}_bp_iters_{num_bp_iterations}'
        if load_sasca:
            x1_4_pmfs = np.load(f'{out_folder}/mc_{mc_block}_x1_4_pmfs_testing_{testing}_alpha_{noise_alpha}.npz')['x1_4_pmfs']
        else:
            x1_4_pmfs = []
            for i in range(num_traces):
                x1_dist, x2_dist, x3_dist, x4_dist = sasca_mixcol_inference(torch.tensor(pmfs[i].T).float(),
                                                                            num_bp_iterations=num_bp_iterations,
                                                                            is_log=False)
                probs = np.concatenate([x1_dist, x2_dist, x3_dist, x4_dist], axis=0)
                x1_4_pmfs.append(probs)

            x1_4_pmfs = np.stack(x1_4_pmfs)

    os.makedirs(out_folder, exist_ok=True)
    if not just_max:
        # save to saved_pmfs
        np.savez(f'{out_folder}/mc_{mc_block}_x1_4_pmfs_testing_{testing}_alpha_{noise_alpha}_batch_{batch_idx}.npz',
                 x1_4_pmfs=x1_4_pmfs)

    if inference_method == InferenceMethod.EXHAUSTIVE:
        if not just_max:
            key_pmfs = x1_4_pmfs # we have already passed messages
            # normalize pmfs:
            key_pmfs /= np.sum(key_pmfs, axis=2, keepdims=True)

        if not load_exact:
            np.savez(f'{out_folder}/mc_{mc_block}_argmax_xs_testing_{testing}_alpha_{noise_alpha}_batch_{batch_idx}.npz',
                 argmax_xs=argmax_xs)
            argmax_successes = compute_argmax_successes(argmax_xs, labels, mc_block)
            np.savez(f'{out_folder}/mc_{mc_block}_argmax_successes_testing_{testing}_alpha_{noise_alpha}_batch_{batch_idx}.npz',
                     argmax_successes=argmax_successes)
            print(argmax_successes)
    else:
        key_pmfs = message_pass_to_x(x1_4_pmfs, y_pmfs)

    if not just_max:
        ranks = compute_exhaustive_ranks(key_pmfs, labels, mc_block)
        np.savez(f'{out_folder}/mc_{mc_block}_ranks_alpha_{noise_alpha}_batch_{batch_idx}.npz', ranks=ranks)
        return ranks
    else:
        return np.array([])


def compute_exhaustive_ranks(key_pmfs, labels, mc_block):
    num_traces = key_pmfs.shape[0]
    ranks = []
    x_names = [f'x{i}_b{mc_block+1}^r1' for i in range(1, 5)]

    for i in range(num_traces):
        true_x = np.array([labels[x][i:i+1] for x in x_names], dtype=np.uint8).squeeze()
        rank = joint_key_dist_rank(key_pmfs[i, :, :], true_x)
        # rank = exhaustive_rank_in_joint_distribution(torch.from_numpy(key_pmfs[i, :, :]), torch.from_numpy(true_x))
        ranks.append(rank)

    return np.array(ranks)


def compute_argmax_successes(argmax_xs, labels, mc_block):
    """
    We assume argmax_xs is of shape (B, 4)
    """
    num_traces = argmax_xs.shape[0]
    successes = []
    x_names = [f'x{i}_b{mc_block+1}^r1' for i in range(1, 5)]
    for i in range(num_traces):
        true_x = np.array([labels[x][i:i+1] for x in x_names], dtype=np.uint8).squeeze()
        success = np.all(argmax_xs[i, :] == true_x)
        successes.append(success)

    return np.array(successes)


def load_lda_pmfs(num_traces, batch_idx, testing=True, mc_block=0):
    # assert num_traces % 4 == 0, 'num_traces must be a multiple of 4'

    # assert num_traces <= 102, 'we have at most 102 traces for testing'
    # n = num_traces // 4
    # batch_idx runs from 0 to 72*2
    batch_idx, half = batch_idx // 2, batch_idx % 2
    npz = np.load(f'../experiments/saved_pmfs/pmfs_fullkey_testing_{testing}.npz', allow_pickle=True)
    pmfs, k, pts = npz['pmfs'], npz['k'], npz['pts']
    with open(f'../experiments/saved_pmfs/labels_fullkey_testing_{testing}.pkl', 'rb') as f:
        labels = pickle.load(f)

    if half == 0:
        pmfs = (pmfs[batch_idx:(batch_idx+1), :128, mc_block*25:(mc_block+1)*25, :]
                .reshape(-1, 25, 256).transpose((0, 2, 1))) # (B, 256, 21)
    else:
        pmfs = (pmfs[batch_idx:(batch_idx+1), 128:, mc_block*25:(mc_block+1)*25, :]
                .reshape(-1, 25, 256).transpose((0, 2, 1))) # (B, 256, 21)

    y_pmfs, pmfs = pmfs[:, :, :4], pmfs[:, :, 4:]
    # y_pmfs = y_pmfs[batch_idx*4:(batch_idx+1)*4, :n, :, :].reshape(-1, 4, 256).transpose((0, 2, 1)) # (B, 4, 256)

    for var in labels.keys():
        l = labels[var].reshape(-1, 256)
        if half == 0:
            labels[var] = l[batch_idx:(batch_idx+1), :128].reshape(-1)
        else:
            labels[var] = l[batch_idx:(batch_idx + 1), 128:].reshape(-1)
        # test
        # labels[var] = labels[var][93:94]

    # pmfs = pmfs[:num_traces, 0, :, :].transpose((0, 2, 1)) # (B, 256, 21)
    # y_pmfs= y_pmfs[:num_traces, 0, :, :] # (B, 4, 256)

    # test
    # pmfs = pmfs[93:94, :, :]
    # y_pmfs = y_pmfs[93:94, :, :]

    return pmfs, y_pmfs, labels


def message_pass_to_x(x1_4_pmfs, y_pmfs):
    num_traces = x1_4_pmfs.shape[0]
    key_pmfs = []
    for i in range(num_traces):
        x_probs = x1_4_pmfs[i, :, :]
        y_probs = y_pmfs[i, :, :]
        x_probs_from_y_probs = np.zeros((4, 256))
        for idx in range(4):
            for i in range(256):
                x_probs_from_y_probs[idx, sbox[i]] = y_probs[idx, i]

        probs = x_probs * x_probs_from_y_probs
        # probs /= probs.sum(axis=1, keepdims=True) # don't normalize - this is only allowed if this is called at the end of the inference procedure
        key_pmfs.append(probs)

    key_pmfs = np.stack(key_pmfs) # (B, 4, 256)
    return key_pmfs
