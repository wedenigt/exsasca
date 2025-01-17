import numpy as np
from tqdm import tqdm
import torch


def joint_key_dist_rank(key_dists: np.ndarray, true_k: np.ndarray, max_count=256) -> int:
    true_k_prob = np.prod([key_dists[i, true_k[i]] for i in range(4)])
    key_dists_sorted = np.flip(np.sort(key_dists, axis=1), axis=1)
    # argsort = np.flip(np.argsort(key_dists, axis=1), axis=1)

    pointers = np.zeros(4, dtype=np.uint8)
    rank = -2
    curr_prob = float('inf')
    next_probs = []
    next_pointers = []
    while curr_prob >= true_k_prob and rank < max_count:
        rank = rank + 1
        curr_prob = np.prod(key_dists_sorted[[0,1,2,3], pointers])

        # update pointers
        for j in range(4):
            p = np.copy(pointers)
            if p[j] == 255:
                # next_probs.append(float('-inf'))
                continue

            p[j] = p[j] + 1
            next_probs.append(np.prod(key_dists_sorted[[0,1,2,3], p]))
            next_pointers.append(p)

        argmax_probs = np.argmax(next_probs)
        best_pointer = next_pointers[argmax_probs]
        # remove best_pointer and best_probs
        next_probs[argmax_probs] = float('-inf')
        pointers = best_pointer

    return rank

def exhaustive_rank_in_joint_distribution(pmfs: torch.tensor, true_k: torch.tensor, device='cpu') -> int:
    """Computes the rank of the true key in the joint distribution."""
    assert pmfs.shape == (4, 256), "pmfs must be of shape (4, 256)"
    assert true_k.shape == (4,), "true_k must be of shape (4,)"

    pmfs = torch.log(pmfs.to(device))
    true_k = true_k.to(device).int()  # we need full ints here (no uint8, since this is interpreted as a binary mask)
    true_k_logprob = sum(pmfs[i, true_k[i]] for i in range(4))

    x_range = torch.arange(0, 256, dtype=torch.uint8, device=device)
    selector = torch.cartesian_prod(x_range, x_range, x_range, x_range).T  # 16GB RAM
    # true_k_idx = 256 ** 3 * true_k[0] + 256 ** 2 * true_k[1] + 256 * true_k[2] + true_k[3]
    # assert true_k_idx < 2 ** 32

    prod = pmfs[0, selector[0].long()]
    for i in tqdm(range(1, 4)):
        prod += pmfs[i, selector[i].long()]

    del selector  # free up memory

    rank = torch.sum(prod >= true_k_logprob)
    del prod

    return rank


if __name__ == '__main__':
    torch.manual_seed(0)

    pmfs = torch.rand(4, 256)
    pmfs /= pmfs.sum(dim=1, keepdims=True)
    true_k = torch.randint(0, 256, size=(4,))
    exhaustive_rank_in_joint_distribution(pmfs, true_k)
