import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from pypsdd import Vtree, SddManager, io, SddNode


def count_products_and_sums(sdd_path):
    vtree_path = sdd_path.replace('.sdd', '.vtree')

    vtree = Vtree.read(vtree_path)
    manager = SddManager(vtree)
    print('Reading SDD from disk...\nThis takes about 4 minutes because we use a naive Python loop to parse the SDD.')
    alpha = io.sdd_read(sdd_path, manager, show_progress=True)  # reading does not work for fully factorized sdds

    non_indicator_lits = list(range(1, 168 + 1))
    non_indicator_lits += [-x for x in non_indicator_lits]
    negative_indicators = [-x for x in range(168 + 1, manager.var_count + 1)]

    # Find out how many (prime, sub) pairs are repeated throughout the sdd (i.e., decomp nodes can share (p,s) pairs)
    print('Looping over SDD...\nThis takes about 3 minutes.')
    ps_map = {}
    for node in tqdm(alpha):
        if node.is_decomposition():
            for p, s in node.elements:
                if s.node_type in [SddNode.TRUE, SddNode.FALSE] or p.node_type in [SddNode.TRUE, SddNode.FALSE]:
                    continue  # skip true/false nodes, these do not need multiplications

                if p.node_type == SddNode.LITERAL and p.literal in non_indicator_lits + negative_indicators:
                    continue  # skip non-indicator primes or negative indicator subs (these are constant 1)

                if s.node_type == SddNode.LITERAL and s.literal in non_indicator_lits + negative_indicators:
                    continue  # skip non-indicator subs or negative indicator subs (these are constant 1)

                if (p, s) in ps_map:
                    ps_map[(p, s)] += 1
                else:
                    ps_map[(p, s)] = 1

    num_multiplications = len(ps_map)
    num_additions = sum(len([(p, s) for p, s in n.elements if not s.is_false_sdd]) - 1 for n in alpha.positive_iter()
                        if n.is_decomposition())

    return num_multiplications, num_additions


def main():
    num_multiplications, num_additions = count_products_and_sums(sdd_path='../compilation/compiled_sdd_g_254/out.sdd')

    # We have to evaluate this circuit 256 times for one forward pass (once for each g in [0,...,255])
    num_multiplications *= 256
    num_additions *= 256
    num_ops = num_multiplications + num_additions

    # A naive loop needs this many operations (since we have 21 bytes)
    naive_num_multiplications = 2 ** 32 * (21 - 1)
    naive_num_additions = 4 * 2 ** 8 * (2 ** 24 - 1)
    naive_num_ops = naive_num_multiplications + naive_num_additions

    mpe_speedup = naive_num_ops / num_ops
    # To compute marginals, we have to compute a forward and backward pass (approx. 2x the number of ops)
    marginal_speedup = naive_num_ops / (num_ops * 2)

    print('SDD Ops')
    print(f'{num_multiplications=}, {num_additions=}, {num_ops=}')
    print('-' * 20)
    print('Naive Loop Ops')
    print(f'{naive_num_multiplications=}, {naive_num_additions=}, {naive_num_ops=}')
    print('-' * 20)
    print(f'MPE Speedup: {mpe_speedup}')
    print(f'Marginal Speedup: {marginal_speedup}')


if __name__ == '__main__':
    main()
