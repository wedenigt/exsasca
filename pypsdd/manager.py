#!/usr/bin/env python
import os
from collections import defaultdict

import numpy as np
import torch

from .sdd import SddNode
from .psdd import PSddNode
import functools


class ComputationCache:
    def __init__(self):
        # self.variable_size = variable_size
        self.cache = defaultdict(dict)

    def lookup(self, first_node: SddNode, second_node: SddNode):
        cache_at_vtree = self.cache[first_node.vtree.id]
        if (first_node, second_node) not in cache_at_vtree:
            return None, 0

        return cache_at_vtree[(first_node, second_node)]

    def update(self, first_node: SddNode, second_node: SddNode, result):
        self.cache[first_node.vtree.id][(first_node, second_node)] = result


def cmp(x, y):
    return (x > y) - (x < y)


class SddManager:
    """SDD Manager"""

    Node = SddNode  # native node class

    def __init__(self, vtree):
        """Constructor

        Initialize with vtree"""

        self.vtree = vtree
        self.var_count = vtree.var_count
        self.unique = {}
        self.id_counter = 0

        self._setup_var_to_vtree(vtree)
        self._setup_terminal_sdds()

    @staticmethod
    def from_pysdd(pysdd, sdd_manager, filename='./tmp.sdd'):
        from pypsdd import io
        pysdd.save(str.encode(filename))
        sdd = io.sdd_read(filename, sdd_manager)  # reading does not work for fully factorized sdds
        os.remove(filename)
        return sdd

    def new_id(self):
        index = self.id_counter
        self.id_counter += 1
        return index

    def _setup_var_to_vtree(self, vtree):
        self.var_to_vtree = [None] * (self.var_count + 1)
        for node in vtree:
            if node.is_leaf():
                self.var_to_vtree[node.var] = node

    def _setup_terminal_sdds(self):
        """Create FALSE, TRUE and LITERAL SDDs"""
        self.false = self.Node(SddNode.FALSE, None, None, self)
        self.true = self.Node(SddNode.TRUE, None, None, self)

        lit_type = SddNode.LITERAL
        self.literals = [None] * (2 * self.var_count + 1)
        for var in range(1, self.var_count + 1):
            vtree = self.var_to_vtree[var]
            self.literals[var] = self.Node(lit_type, var, vtree, self)
            self.literals[-var] = self.Node(lit_type, -var, vtree, self)

    def _canonical_elements(self, elements):
        """Given a list of elements, canonicalize them"""
        cmpf = lambda x, y: cmp(x[0].id, y[0].id)
        elf = lambda x: tuple(sorted(x, key=functools.cmp_to_key(cmpf)))
        return elf(elements)

    def lookup_node(self, elements, vtree_node):
        """Unique table lookup for DECOMPOSITION nodes.

        Elements is a list of prime,sub pairs:
            [ (p1,s1),(p2,s2),...,(pn,sn) ]"""
        elements = self._canonical_elements(elements)
        if elements not in self.unique:
            node_type = SddNode.DECOMPOSITION
            node = self.Node(node_type, elements, vtree_node, self)
            self.unique[elements] = node

        return self.unique[elements]

    def create_fully_factorized_sdd(self):
        """
        Uses vtree stored in manager to construct a fully-factorized PSDD that respects this vtree.
        """
        vtree_to_sdd_node = {}
        for node in self.vtree.post_order():
            if node.is_leaf():
                true_node = SddNode(node_type=SddNode.TRUE, alpha=node.var, vtree=node, manager=self)
                vtree_to_sdd_node[node.id] = true_node
            else:
                prime = vtree_to_sdd_node[node.left.id]
                sub = vtree_to_sdd_node[node.right.id]

                vtree_to_sdd_node[node.id] = SddNode(node_type=SddNode.DECOMPOSITION, alpha=[(prime, sub)], vtree=node,
                                                     manager=self)

        root = vtree_to_sdd_node[self.vtree.id]

        return root


class PSddManager(SddManager):
    Node = PSddNode  # native node class

    def __init__(self, vtree):
        SddManager.__init__(self, vtree)
        self._setup_true_false_sdds()

    def _setup_true_false_sdds(self):
        """PSDDs are normalized, so we create a unique true/false SDD/PSDD
        node for each vtree node."""
        node_count = 2 * self.var_count - 1
        self.true_sdds = [None] * node_count
        self.false_sdds = [None] * node_count

        for vtree_node in self.vtree.post_order():
            # setup true SDDs
            if vtree_node.is_leaf():
                node_type = SddNode.TRUE
                true_node = self.Node(node_type, None, vtree_node, self)
            else:
                left_true = self.true_sdds[vtree_node.left.id]
                right_true = self.true_sdds[vtree_node.right.id]
                elements = [(left_true, right_true)]
                true_node = self.lookup_node(elements, vtree_node)
            self.true_sdds[vtree_node.id] = true_node

            # setup false SDDs
            if vtree_node.is_leaf():
                node_type = SddNode.FALSE
                false_node = self.Node(node_type, None, vtree_node, self)
            else:
                left_true = self.true_sdds[vtree_node.left.id]
                right_false = self.false_sdds[vtree_node.right.id]
                elements = [(left_true, right_false)]
                false_node = self.lookup_node(elements, vtree_node)
            self.false_sdds[vtree_node.id] = false_node
            false_node.is_false_sdd = True

            true_node.negation = false_node
            false_node.negation = true_node

    def negate(self, node, vtree):
        """Negate a normalized SDD node"""

        if node.is_false():
            return self.true_sdds[vtree.id]
        elif node.is_true():
            return self.false_sdds[vtree.id]
        elif node.is_literal():
            return self.literals[-node.literal]
        elif node.negation is not None:
            return node.negation
        else:  # node.is_decomposition()
            right = node.vtree.right
            elements = [(p, self.negate(s, right)) for p, s in node.elements]
            neg = self.lookup_node(elements, vtree)
            neg.negation = node
            node.negation = neg
            return neg

    def copy_and_normalize_sdd(self, alpha, vtree) -> PSddNode:
        """Copy an SDD alpha from another manager to the self manager, and
        normalize it with respect to the given vtree."""

        for node in alpha.post_order(clear_data=True):
            if node.is_false():
                copy_node = self.false
            elif node.is_true():
                copy_node = self.true
            elif node.is_literal():
                copy_node = self.literals[node.literal]
            else:  # node.is_decomposition()
                elements = []
                left, right = node.vtree.left, node.vtree.right
                for prime, sub in node.elements:
                    copy_prime = self._normalize_sdd(prime.data, left)
                    copy_sub = self._normalize_sdd(sub.data, right)
                    elements.append((copy_prime, copy_sub))
                copy_node = self.lookup_node(elements, node.vtree)
            node.data = copy_node
        root_sdd = self._normalize_sdd(copy_node, vtree)
        return root_sdd

    def _normalize_sdd(self, alpha, vtree):
        """Normalize a given sdd for a given vtree"""
        if alpha.is_false():
            return self.false_sdds[vtree.id]
        elif alpha.is_true():
            return self.true_sdds[vtree.id]
        elif alpha.vtree.id == vtree.id:
            return alpha

        if alpha.vtree.id < vtree.id:
            left = self._normalize_sdd(alpha, vtree.left)
            right = self.true_sdds[vtree.right.id]
            neg_left = self.negate(left, vtree.left)
            false_right = self.false_sdds[vtree.right.id]
            elements = [(left, right), (neg_left, false_right)]
        elif alpha.vtree.id > vtree.id:
            left = self.true_sdds[vtree.left.id]
            right = self._normalize_sdd(alpha, vtree.right)
            elements = [(left, right)]
        return self.lookup_node(elements, vtree)

    def get_byte_marginal_evidence(self, num_vars=None):
        """
        Returns evidence matrix of shape (num_bytes*256, num_vars).
        e.g., first row is 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, ..., -1
             second row is 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, ..., -1
               256 row is -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, ..., 0, -1, ...
               257 row is -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, ..., 1, -1, ...
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_vars = num_vars or self.vtree.var_count
        assert num_vars % 8 == 0, 'PSDD needs to model a distribution over a multiple of 8 bits'
        num_bytes = int(num_vars // 8)
        e = torch.zeros((num_bytes * 256, num_vars), dtype=torch.float32, device=device) - 1.0

        for byte_idx in range(num_bytes):
            for byte_value in range(256):
                i = (byte_idx * 256) + byte_value
                np_value = np.array([byte_value], dtype=np.uint8)
                e[i, byte_idx * 8:(byte_idx + 1) * 8] = torch.tensor(np.unpackbits(np_value, bitorder='little'),
                                                                     device=device).float()

        return e

    def multiply(self, first: Node, second: Node, batch_size: int, num_reps: int):
        assert (first.mixing.shape[-1] > 1 and second.mixing.shape[-1] == 1) or \
               (first.mixing.shape[-1] == 1 and second.mixing.shape[-1] > 1) or \
               (first.mixing.shape[-1] == 1 and second.mixing.shape[-1] == 1), \
            'Exactly one of the two PSDDs must be a mixture (can be a degenerate one with only one weight)'

        mixing_weights = first.mixing if second.mixing.shape[-1] == 1 else second.mixing
        assert mixing_weights.shape[-1] == num_reps

        cache = ComputationCache()
        result_psdd, partition = self.multiply_with_cache(first, second, cache, batch_size, num_reps)
        result_psdd.mixing = mixing_weights
        partition = (partition + mixing_weights).logsumexp(dim=-1,
                                                           keepdim=True)  # partition of the mixture is the weighted sum of the partitions (in log-space)

        return result_psdd, partition

    def multiply_with_cache(self, first: Node, second: Node, cache: ComputationCache, batch_size: int, num_reps: int):
        assert first.vtree.id == second.vtree.id  # must be normalized w.r.t. the same vtree

        result, param = cache.lookup(first, second)
        if result is not None:
            return result, param

        if first.is_decomposition():
            assert second.is_decomposition()
            first_primes, first_subs = [e[0] for e in first.positive_elements], [e[1] for e in first.positive_elements]
            second_primes, second_subs = [e[0] for e in second.positive_elements], [e[1] for e in
                                                                                    second.positive_elements]

            next_primes, next_subs, next_parameters = [], [], []
            for i in range(len(first_primes)):
                for j in range(len(second_primes)):
                    mult_prime_result = self.multiply_with_cache(first_primes[i], second_primes[j], cache, batch_size,
                                                                 num_reps)
                    if mult_prime_result[0] is None:
                        continue

                    mult_sub_result = self.multiply_with_cache(first_subs[i], second_subs[j], cache, batch_size,
                                                               num_reps)
                    if mult_sub_result[0] is None:
                        continue
                    next_primes.append(mult_prime_result[0]);
                    next_subs.append(mult_sub_result[0])

                    assert first.theta.shape[0] == len(first_primes), (first.theta.shape, len(first_primes))
                    assert second.theta.shape[0] == len(second_primes), (second.theta.shape, len(second_primes))

                    param = second.theta[j] + first.theta[i] + mult_prime_result[1][0] + mult_sub_result[1][0]
                    next_parameters.append(param)
                    # partition = next_parameters[-1] if partition is None else torch.logaddexp(partition, next_parameters[-1])

            if len(next_primes) == 0:
                comp_result = (None, torch.zeros(
                    (1, batch_size, num_reps)) - 300)  # TODO: find out which shape to use here (zeros_like?)
                cache.update(first, second, comp_result)
                return comp_result

            next_parameters = torch.stack(next_parameters, dim=0)
            partition = torch.logsumexp(next_parameters, dim=0, keepdim=True)

            next_parameters = next_parameters - partition  # divide by partition in log-space
            # in C++, they call `GetConformedPsddDecisionNode` here, where they do some LCA magic (I guess to compress the result)
            new_node = PSddNode(PSddNode.DECOMPOSITION, alpha=list(zip(next_primes, next_subs)), vtree=first.vtree,
                                manager=self)
            new_node.theta = next_parameters
            cache.update(first, second, (new_node, partition))
            return new_node, partition
        elif first.is_literal():
            if second.is_literal():
                if first.literal == second.literal:
                    new_node = PSddNode(PSddNode.LITERAL, alpha=first.literal, vtree=first.vtree, manager=self)
                    result = (new_node, torch.zeros(
                        (1, batch_size, num_reps)))  # TODO: find out which shape to use here (zeros_like?)
                else:
                    assert first.vtree.var == second.vtree.var
                    result = (None, torch.zeros(
                        (1, batch_size, num_reps)) - 300)  # TODO: find out which shape to use here (zeros_like?)

                cache.update(first, second, result)
                return result
            else:
                assert second.is_true(), 'Second Node must be True here. (called TOP_NODE in C++ code)'
                assert first.vtree.var == second.vtree.var
                assert second.theta.shape[0] == 2, 'Theta should only contain one positive and one negative value'

                new_node = PSddNode(PSddNode.LITERAL, alpha=first.literal, vtree=first.vtree, manager=self)
                if first.literal > 0:
                    result = (new_node, second.theta[1:, :, :])  # slice out positive parameters
                else:
                    result = (new_node, second.theta[:1, :, :])  # slice out negative parameters

                cache.update(first, second, result)
                return result
        else:
            assert first.is_true(), first
            assert first.vtree.var == second.vtree.var

            if second.is_literal():
                new_node = PSddNode(PSddNode.LITERAL, alpha=second.literal, vtree=second.vtree, manager=self)
                if first.literal > 0:
                    result = (new_node, first.theta[1:, :, :])  # slice out positive parameters
                else:
                    result = (new_node, first.theta[:1, :, :])  # slice out negative parameters

                cache.update(first, second, result)
                return result
            else:
                assert second.is_true()
                param_mult = first.theta + second.theta  # multiplies negative and positive params in log-space
                partition = torch.logsumexp(param_mult, dim=0, keepdim=True)
                new_node = PSddNode(PSddNode.TRUE, alpha=None, vtree=second.vtree, manager=self)
                normalized_theta = param_mult - partition  # divide param_mult by partition in log-space (will braodcast over dimension 0)
                batch_size = normalized_theta.shape[1]
                assert torch.allclose(normalized_theta.logsumexp(dim=0, keepdim=True),
                                      torch.zeros((1, batch_size, num_reps)), atol=1e-06)

                new_node.theta = normalized_theta

                result = (new_node, partition)
                cache.update(first, second, result)
                return result
