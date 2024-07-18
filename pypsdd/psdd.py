import math
import random
import heapq
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm

from . import Vtree
from .sdd import SddNode, NormalizedSddNode
from .data import DataSet, Inst, InstMap, WeightedInstMap
from .sdd import SddEnumerator, SddTerminalEnumerator
from .prior import Prior
import torch
import torch.nn.functional as F


class PSddNode(NormalizedSddNode):
    """Probabilistic Sentential Decision Diagram (PSDD)

    See https://github.com/hahaXD/psdd for PSDD multiply."""

    _brute_force_limit = 10

    ########################################
    # CONSTRUCTOR + BASIC FUNCTIONS
    ########################################

    def __init__(self, node_type, alpha, vtree, manager):
        """Constructor

        node_type is FALSE, TRUE, LITERAL or DECOMPOSITION
        alpha is a literal if node_type is LITERAL
        alpha is a list of elements if node_type is DECOMPOSITION
        vtree is the vtree that the node is normalized for
        manager is the PSDD manager"""
        NormalizedSddNode.__init__(self, node_type, alpha, vtree, manager)
        self.mixing = None
        self.num_branches = None

    def get_num_branches(self):
        num_branches = defaultdict(list)
        for node in self.positive_iter():
            if node.is_decomposition() and len(node.positive_elements) > 1:
                num_branches[len(node.positive_elements)].append(node)

            # elif node.is_mixing():
            #    num_branches[len(node.elements)].append(node)
            elif node.is_true():
                num_branches[2].append(node)

        return num_branches

    def check_if_respects_vtree(self, vtree: Vtree):
        vtree_nodes = {}
        for node in vtree:
            vtree_nodes[node.id] = node

        for psdd_node in self.positive_iter():
            vtree_node = vtree_nodes[psdd_node.vtree.id]
            if psdd_node.is_true() or psdd_node.is_literal():
                assert vtree_node.is_leaf(), "PSDD node is leaf but vtree node is not"
            elif psdd_node.is_decomposition():
                assert not vtree_node.is_leaf(), "PSDD node is decomp but vtree node is leaf"

                for prime, sub in psdd_node.positive_elements:
                    assert vtree_node.left.id == prime.vtree.id, "PSDD prime does not respect vtree"
                    assert vtree_node.right.id == sub.vtree.id, "PSDD sub does not respect vtree"
            else:
                raise Exception("Unknown PSDD node type")

    def prune_zero_weight_decomp_nodes(self, log_threshold=-300):
        for psdd_node in self.positive_iter():
            if psdd_node.is_decomposition():
                el_to_keep = []
                for i, el in enumerate(psdd_node.positive_elements):
                    if psdd_node.theta[i, 0, 0].item() > log_threshold:
                        el_to_keep.append(el)

                psdd_node.positive_elements = el_to_keep

    def set_uniform_params(self, batch_size, num_reps, device):
        """
        Sets theta for each node such that the output distribution is uniform over all valid states.
        Returns list of trainable parameters (tensors).
        """
        params = []
        for node in self.positive_iter():
            if node.is_decomposition():
                if len(node.positive_elements) == 1:
                    node.theta = torch.zeros((1, batch_size, num_reps), device=device).float()
                else:
                    node.theta = torch.zeros((len(node.positive_elements), batch_size, num_reps), device=device).float() \
                                 - torch.log(
                        torch.tensor(len(node.positive_elements), device=device))  # (1/len(elements) in log-space)
                    params.append(node.theta)
            elif node.is_true():
                node.theta = torch.zeros((2, batch_size, num_reps), device=device).float() \
                             - torch.log(torch.tensor(2, device=device))  # (1/2 in log-space)
                params.append(node.theta)

        self.mixing = torch.zeros((batch_size, num_reps), device=device).float() \
                      - torch.log(torch.tensor(num_reps, device=device))  # (1/num_reps in log-space)
        params.append(self.mixing)

        return params

    def set_fully_factorized_params(self, thetas, log_space=True, detach=False):
        assert log_space, "Fully factorized parameterization only implemented in log-space"

        # thetas: (batch_size, 168)
        batch_size = thetas.shape[0]
        if self.num_branches is None:
            self.num_branches = self.get_num_branches()

        assert len(self.num_branches[2]) == self.vtree.var_count  # only true nodes to parameterize

        # set theta for decomposition nodes with only one (p,s) pair (which must have weight 1, no free parameter here)
        for node in self.positive_iter():
            if node.is_decomposition() and len(node.positive_elements) == 1:
                node.theta = torch.zeros((1, batch_size, 1), device=thetas.device).float()

        self.mixing = torch.zeros((batch_size, 1), device=thetas.device).float()  # no mixing

        for node in self.num_branches[2]:
            pos_param = F.logsigmoid(thetas[:, node.vtree.var - 1])
            neg_param = torch.log(1 - torch.exp(pos_param))
            param = torch.stack((neg_param, pos_param), dim=1)  # (batch_size, 2)

            node.theta = param.T.unsqueeze(-1)  # shape: (2, batch_size, 1)

    def set_fully_factorized_params_with_dense_gating_function(self, thetas, log_space=True, detach=False):
        assert log_space, "Fully factorized parameterization only implemented in log-space"

        # thetas: (batch_size, num_vars_to_set, 2, num_reps)
        batch_size, num_vars_to_set, _, num_reps = thetas.shape
        self.num_branches = self.get_num_branches()

        assert len(self.num_branches[2]) == self.vtree.var_count  # only true nodes to parameterize

        # set theta for decomposition nodes with only one (p,s) pair (which must have weight 1, no free parameter here)
        for node in self.positive_iter():
            if node.is_decomposition() and len(node.positive_elements) == 1:
                node.theta = torch.zeros((1, batch_size, 1), device=thetas.device).float()

        self.mixing = torch.zeros((batch_size, num_reps), device=thetas.device).float() \
                      - torch.log(torch.tensor(num_reps, device=thetas.device).float())  # uniform mixing

        set_var_count = 0
        for node in self.num_branches[2]:
            if node.vtree.var > num_vars_to_set:
                node.theta = torch.zeros((2, batch_size, num_reps), device=thetas.device).float() + \
                             torch.log(torch.tensor(1 / 2, device=thetas.device).float())  # uniform distribution
                continue  # assumes dense set of variables to set; TODO: make this more general
            else:
                param = F.log_softmax(thetas[:, node.vtree.var - 1, :, :], dim=1)
                node.theta = param.permute(1, 0, 2)  # shape: (2, batch_size, num_reps)
                set_var_count += 1

        assert set_var_count == num_vars_to_set

    def set_params(self, thetas, log_space=True, detach=False, only_leafs=False):
        params = []
        batch_size = thetas[-1].shape[0]
        # if self.num_branches is None:
        self.num_branches = self.get_num_branches()

        # set theta for decomposition nodes with only one (p,s) pair (which must have weight 1, no free parameter here)
        for node in self.positive_iter():
            if node.is_decomposition() and len(node.positive_elements) == 1:
                node.theta = torch.zeros((1, batch_size, 1), device=thetas[-1].device).float()

        self.mixing = thetas[-2].log_softmax(dim=1) if log_space else thetas[-2].softmax(dim=1)
        self.root_mixing = thetas[-1].log_softmax(dim=1) if log_space else thetas[-1].softmax(dim=1)
        self.theta = self.root_mixing.permute(1, 0, 2)

        if detach:
            self.mixing = self.mixing.detach();
            self.mixing.requires_grad_()
            self.root_mixing = self.root_mixing.detach();
            self.root_mixing.requires_grad_()

        params.append(self.mixing);
        params.append(self.root_mixing)

        for theta, grouping in zip(thetas[:-2], self.num_branches.items()):
            # (batch_size x num_sum_nodes x num_children, K)
            # -> (num_sum_nodes x batch_size x num_children, K)
            # assert(theta.size(1) == len(grouping[1]) and theta.size(2) == grouping[0])

            theta = theta.permute(1, 2, 0, 3)  # (num_sum_node x num_children x batch_size x K)
            if log_space:
                theta = F.log_softmax(theta, dim=1)
            else:
                theta = theta.softmax(dim=1)

            for param, node in zip(theta, grouping[1]):
                # shape: (batch_size x num_children, K)
                # if transpose_dims_true_nodes and node.is_true():
                #     node.theta = param.transpose(0, 1) # transpose the first two dims if we have a true node (constant). many computations seem to assume this
                # else:
                if detach:
                    param = param.detach();
                    param.requires_grad_()

                # if node.is_true():
                #     print('debug')

                node.theta = param
                params.append(param)

        return params

    def as_table(self, global_var_count=0):
        """Returns a string representing the full-joint probability
        distribution, e.g., a table over two variables would yield:

        AB   Pr
        ---------
        00 0.1000
        01 0.2000
        10 0.3000
        11 0.4000

        Maximum 10 (global) variables.

        If global_var_count is set to the global PSDD var count, then
        print sub-nodes will try to use global variable names.  Otherwise,
        the variable indices start with A.

        If positive is set to True, then only positive rows are printed."""
        var_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        var_count = max(self.vtree.var_count, global_var_count)
        assert var_count <= PSddNode._brute_force_limit

        st = [var_names[:var_count] + "   Pr  "]
        st.append(("-" * var_count) + "+------")
        for model in self.models(self.vtree, lexical=True):
            pr = self.pr_model(model)
            if global_var_count:
                model[global_var_count] = model[global_var_count]  # hack
                st.append("%s %.4f" % (model, pr))
            else:
                st.append("%s %.4f" % (model.shrink(), pr))
        return "\n".join(st)

    def get_top_k_products_from_maxes(maxes: List[np.ndarray], argmaxes: List[List[Dict]], k=256) \
            -> Union[np.ndarray, List[Dict]]:
        num_children = len(maxes)
        # true_k_prob = np.prod([key_dists[i, true_k[i]] for i in range(4)])
        # key_dists_sorted = np.flip(np.sort(key_dists, axis=1), axis=1)
        # argsort = np.flip(np.argsort(key_dists, axis=1), axis=1)

        pointers = [0] * num_children  # np.zeros(num_children, dtype=np.uint16)
        curr_prob = float('inf')
        l = list(range(num_children))
        next_probs = []
        next_pointers = []
        final_maxes, final_argmaxes = [], []
        for counter in range(k):
            curr_max = np.max(maxes[child][pointer] for child, pointer in zip(l, pointers))
            final_maxes.append(curr_max)
            final_argmaxes.append([argmaxes[child][pointer] for child, pointer in zip(l, pointers)])

            # update pointers
            for j in range(4):
                p = list(np.copy(pointers))
                if p[j] == len(maxes[j]) - 1:
                    # next_probs.append(float('-inf'))
                    continue

                p[j] = p[j] + 1
                next_probs.append(np.sum(maxes[child][pointer] for child, pointer in zip(l, p)))
                next_pointers.append(p)

            argmax_probs = np.argmax(next_probs)
            best_pointer = next_pointers[argmax_probs]
            # remove best_pointer and best_probs
            next_probs[argmax_probs] = float('-inf')
            pointers = best_pointer

        return counter

    def top_k_mpe(self, k=256, show_tqdm=False, clear_data=True, num_nodes_in_psdd=None, device='cpu') -> torch.tensor:
        if show_tqdm:
            iterator = tqdm(self.as_positive_list(clear_data=clear_data), total=num_nodes_in_psdd)
        else:
            iterator = self.as_positive_list(clear_data=clear_data)

        for node in iterator:
            if node.is_literal():
                node.maxes = np.array([0])  # LOG_ONE
                # node.argmaxes = [{abs(node.literal): 1}] if node.literal > 0 else [{abs(node.literal): 0}]
            elif node.is_true():
                node.maxes = np.array([node.theta[:, 0, 0].max().item(), node.theta[:, 0, 0].min().item()])
                # node.argmaxes = [{abs(node.literal): node.theta[:, 0, 0].argmax().item()},
                #                  {abs(node.literal): node.theta[:, 0, 0].argmin().item()}]
            elif node.is_decomposition():
                node.children_maxes = []
                # node.children_argmaxes = []

                # find top-k maxes and argmaxes for each (p,s) pair
                for j, (p, s) in enumerate(node.positive_elements):
                    x = np.add.outer(p.maxes, s.maxes)
                    maxes = np.sort(x.flatten())[::-1][:k]
                    node.children_maxes.append(maxes + node.theta[j, 0, 0].cpu().numpy())

                    # argmaxes = np.argsort(x.flatten())[::-1][:k]
                    # argmaxes_p = argmaxes // x.shape[1]
                    # argmaxes_s = argmaxes % x.shape[1]

                    # argmaxes_p, argmaxes_s = np.where(np.isin(x, maxes))
                    # argmaxes = []
                    # for ass_p, ass_s in zip(argmaxes_p, argmaxes_s):
                    #     ass = {**p.argmaxes[ass_p], **s.argmaxes[ass_s]} # merge dicts
                    #     argmaxes.append(ass)
                    # node.children_argmaxes.append(argmaxes)

                # find top-k maxes among all children_maxes
                node.children_maxes = np.array(node.children_maxes)
                maxes = np.sort(node.children_maxes.flatten())[::-1][:k]

                # argmaxes = np.argsort(node.children_maxes.flatten())[::-1][:k]
                # argmax_child_idx = argmaxes // node.children_maxes.shape[1]
                # argmax_ass = argmaxes % node.children_maxes.shape[1]

                # argmax_child_idx, argmax_ass = np.where(np.isin(node.children_maxes, maxes))
                # argmaxes = []
                # for child_idx, ass in zip(argmax_child_idx, argmax_ass):
                #     argmaxes.append(node.children_argmaxes[child_idx][ass])

                node.maxes = maxes
                # node.argmaxes = argmaxes
            else:
                raise ValueError('Unknown node')

        return self.maxes  # , self.argmaxes

    def prob(self, evidence: torch.tensor, clear_data=True, starting_node=None,
             custom_data=None, log_w=True, num_nodes_in_psdd=None, show_tqdm=False) -> torch.tensor:
        """
        Compute the probability p(evidence[i]) w.r.t. the PSDD with parameters theta[i]
        """
        assert evidence.min() >= -1 and evidence.max() <= 1, "The evidence tensor should only contain -1, 0, and 1"

        beta = starting_node if starting_node is not None else self

        ONE = torch.tensor(1.0, device=evidence.device)
        LOG_ONE = torch.tensor([0.0], device=evidence.device)  # 0 is "1" in logspace
        LOG_ZERO = torch.tensor([-300.0], device=evidence.device)  # essentially 0 in logspace (e^{-300})

        if beta.mixing is not None:
            num_reps = beta.mixing.shape[-1]

        if show_tqdm:
            iterator = tqdm(beta.as_positive_list(clear_data=clear_data), total=num_nodes_in_psdd)
        else:
            iterator = beta.as_positive_list(clear_data=clear_data)

        for node in iterator:
            if node.is_false():
                value = LOG_ZERO
            elif node.is_true():
                num_reps, batch_size = node.theta.shape[-1], evidence.shape[0]
                val = evidence[..., node.vtree.var - 1].long()
                # value = node.theta[val, :]

                theta = node.theta.permute(1, 0, 2)
                if num_reps > 1:
                    pos = theta[:, 1, :] * val
                    neg = theta[:, 0, :] * (1 - val)
                else:
                    pos = theta[:, 1:, :] * val
                    neg = theta[:, 0:1, :] * (1 - val)

                # value = pos + neg

                value = torch.nan_to_num(pos, posinf=float('inf'), neginf=-float('inf')) + \
                        torch.nan_to_num(neg, posinf=float('inf'),
                                         neginf=-float('inf'))  # 0 * (-inf) = nan, so we need to replace nan with 0

                # value = torch.cat([node.theta[val[k], k].T for k in range(batch_size)], dim=0) # TODO: figure out way to vectorize
                # assert torch.allclose(value, value_old)

                val = val.repeat(1, 1, num_reps)  # .reshape(num_reps, 1, batch_size).T
                value = torch.where(val == -1, LOG_ONE, value)
                # value = value.squeeze().unsqueeze(dim=0)
            elif node.is_literal():
                val = evidence[..., abs(node.literal) - 1]
                if node.literal < 0:
                    val = 1 - val  # flip the value if the literal is negative
                    val = torch.where(val == 2, ONE,
                                      val)  # marginalization: if -1, set literal to 1 (which we convert to LOG_ONE later)
                else:
                    val = torch.where(val == -1, ONE,
                                      val)  # marginalization: if -1, set literal to 1 (which we convert to LOG_ONE later)

                val = val.repeat(1, 1, 1)
                value = torch.where(val == 1, LOG_ONE, LOG_ZERO)  # if val == 1, set to LOG_ONE, else set to LOG_ZERO
                # assert value.shape[0] == 1
                # value = value.squeeze(dim=0)
            elif node.is_decomposition():
                num_reps = node.theta.shape[-1]

                if len(node.positive_elements) == 1:
                    p, s = node.positive_elements[0]
                    value = (p.data.T + s.data.T).T
                else:
                    primes, subs = zip(*node.positive_elements)
                    primes = torch.stack([p.data for p in primes])  # .transpose(0, 1)
                    subs = torch.stack([s.data for s in subs])  # .transpose(0, 1)
                    # assert len(subs.shape) == 3
                    # print(primes.device, subs.device, node.theta.device)
                    theta = node.theta.T.unsqueeze(dim=1) if num_reps > 1 else node.theta.T
                    value = (primes.T + subs.T + theta).T.logsumexp(dim=0)  # .logsumexp(dim=1)
            else:
                raise ValueError('Unknown node type')

            node.data = value
            if custom_data is not None and node.id in custom_data.keys():
                node.data = custom_data[node.id]
            # print(value.shape)

        if beta.mixing is not None and beta.mixing.shape[-1] > 1:
            if log_w:
                value = (value + beta.mixing).logsumexp(dim=-1, keepdim=True)
            else:
                assert torch.isclose(torch.sum(beta.mixing.exp()), torch.tensor(1.0, device=beta.mixing.device))
                value = (beta.mixing.exp() * value).sum(dim=-1).unsqueeze(dim=-1)  # compute \sum_k w_k * log(q_k)

        # assert value.shape[-1] == 1
        return value

    ########################################
    # STATS
    ########################################

    def theta_count(self):
        """Counts the number of free parameters in a PSDD.

        Only 'live' nodes are considered. i.e., we do not count
        (sub)-nodes of primes with false subs."""
        count = 0
        for node in self.positive_iter():
            if node.is_literal():  # or node.is_false_sdd
                pass
            elif node.is_true():
                count += 1
            else:  # node.is_decomposition()
                count += len(node.positive_elements) - 1
        return count

    def zero_count(self):
        """Counts the number of (live) zero parameters in a PSDD"""
        count = 0
        for node in self.positive_iter():
            if node.is_literal():  # or node.is_false_sdd
                val = int(node.literal > 0)
                count += node.theta[val] == 0
            elif node.is_true():
                count += node.theta[0] == 0
                count += node.theta[1] == 0
            else:  # node.is_decomposition()
                for element in node.positive_elements:
                    count += node.theta[element] == 0
        return count

    def true_count(self):
        """Counts the number of (live) true nodes in a PSDD"""
        return sum(1 for n in self.positive_iter() if n.is_true())

    def vtree_counts(self, manager):
        """Counts the number of nodes for each vtree node, indexed by
        vtree.id"""
        counts = [0] * (2 * manager.var_count - 1)
        for node in self.positive_iter():
            counts[node.vtree.id] += 1
        return counts

    ########################################
    # INFERENCE
    ########################################

    def pr_model(self, inst):
        """Returns Pr(inst) for a complete instantiation inst (where inst is
        an Inst or InstMap).

        Performs recursive test, which can be faster than linear
        traversal as in PSddNode.value."""
        self.is_model_marker(inst, clear_bits=False, clear_data=False)

        if self.data is None:
            pr = 0.0
        else:
            pr = 1.0
            queue = [self] if self.data is not None else []
            while queue:
                node = queue.pop()
                assert node.data is not None
                pr *= node.theta[node.data] / node.theta_sum
                if node.is_decomposition():
                    queue.append(node.data[0])  # prime
                    queue.append(node.data[1])  # sub

        self.clear_bits(clear_data=True)
        return pr

    def value(self, evidence=InstMap(), clear_data=True):
        """Compute the (un-normalized) value of a PSDD given evidence"""
        if self.is_false_sdd: return 0.0
        for node in self.as_positive_list(clear_data=clear_data):
            if node.is_false():
                value = 0.0
            elif node.is_true():
                if node.vtree.var in evidence:
                    val = evidence[node.vtree.var]
                    value = node.theta[val]
                else:
                    value = node.theta_sum
            elif node.is_literal():
                sim = evidence.is_compatible(node.literal)
                value = node.theta_sum if sim else 0.0
            else:  # node.is_decomposition()
                value = 0.0
                for p, s in node.positive_elements:
                    theta = node.theta[(p, s)]
                    # print(theta.shape, p.data.shape, s.data.shape)
                    value += (p.data / p.theta_sum) * (s.data / s.theta_sum) * theta
            node.data = value

        return value

    def probability(self, evidence=InstMap(), clear_data=True):
        """Compute the probability of evidence in a PSDD"""
        value = self.value(evidence=evidence, clear_data=clear_data)
        return value / self.theta_sum

    def marginals(self, evidence=InstMap(), clear_data=True, do_bottom_up=True):
        """Evaluate a PSDD top-down for its marginals.

        Returns a list var_marginals where:
        = var_marginals[lit] = value(lit,e)
        = var_marginals[0]   = value(e)

        Populates a field on each node:
        = node.pr_context has probability of context
        = node.pr_node has probability of node"""
        var_marginals = [0.0] * (2 * self.vtree.var_count + 1)
        if self.is_false_sdd: return var_marginals

        if do_bottom_up:  # do not call value if done already
            self.value(evidence=evidence, clear_data=False)
        for node in self.as_positive_list(clear_data=False):  # init field
            node.pr_context = 0.0

        value = self.data
        self.pr_context = 1.0
        for node in self.as_positive_list(reverse=True, clear_data=clear_data):
            if node.is_true() or node.is_literal():
                # accumulate variable marginals
                var = node.vtree.var
                pr_pos = node.theta[1] / node.theta_sum
                pr_neg = node.theta[0] / node.theta_sum
                if var in evidence:
                    val = evidence[var]
                    if val:
                        var_marginals[var] += pr_pos * node.pr_context
                    else:
                        var_marginals[-var] += pr_neg * node.pr_context
                else:
                    var_marginals[var] += pr_pos * node.pr_context
                    var_marginals[-var] += pr_neg * node.pr_context
            else:  # node.is_decomposition()
                # accumulate node marginals
                for p, s in node.positive_elements:
                    theta = node.theta[(p, s)] / node.theta_sum
                    pr_p = p.data / p.theta_sum
                    pr_s = s.data / s.theta_sum
                    p.pr_context += theta * pr_s * node.pr_context
                    s.pr_context += theta * pr_p * node.pr_context
            node.pr_node = node.pr_context * (node.data / node.theta_sum)

        var_marginals[0] = value
        return var_marginals

    def mpe(self, evidence=InstMap()):
        """Compute the most probable explanation (MPE) given evidence.
        Returns (un-normalized) MPE value and instantiation.

        If evidence is inconsistent with the PSDD, will return arbitrary
        instanatiation consistent with evidence."""
        if self.is_false_sdd:
            inst = InstMap.from_bitset(0, self.vtree.var_count).concat(evidence)
            return 0.0, inst
        for node in self.as_positive_list(clear_data=False):
            if node.is_false():
                var = node.vtree.var
                mpe_val = 0.0
                mpe_ind = evidence[var] if var in evidence else 0  # arbitrary index
            elif node.is_false_sdd:
                mpe_val = 0.0
                mpe_ind = node.elements[0]  # arbitrary index
            elif node.is_true() or node.is_literal():
                var, theta = node.vtree.var, list(node.theta)
                if var in evidence:
                    mpe_val = theta[evidence[var]]
                    mpe_ind = evidence[var]
                else:
                    mpe_val = max(theta)
                    mpe_ind = theta.index(mpe_val)
            else:  # node.is_decomposition()
                pels = node.positive_elements
                pvals = [p.data[0] / p.theta_sum for p, s in pels]
                svals = [s.data[0] / s.theta_sum for p, s in pels]
                vals = [pval * sval * node.theta[el] for pval, sval, el \
                        in zip(pvals, svals, pels)]
                mpe_val, mpe_ind = max(list(zip(vals, pels)))
            node.data = (mpe_val, mpe_ind)

        mpe_inst = InstMap()
        queue = [self] if self.data is not None else []
        while queue:
            node = queue.pop()
            if node.is_decomposition():
                prime, sub = node.data[1]
                queue.append(prime)
                queue.append(sub)
            else:
                mpe_inst[node.vtree.var] = node.data[1]

        # clear_data
        for node in self._positive_array: node.data = None
        return mpe_val, mpe_inst

    def enumerate_mpe(self, pmanager, evidence=InstMap()):
        """Enumerate the top-k MPE's of a PSDD
        AC: TODO evidence."""

        enum = PSddEnumerator(self.vtree)
        return enum.enumerator(self)

    ########################################
    # KL-DIVERGENCE
    ########################################

    @staticmethod
    def kl(pr1, pr2):
        """Compute KL-divergence between two (list) distributions pr1 and pr2"""
        kl = 0.0
        for p1, p2 in zip(pr1, pr2):
            if p1 == 0.0: continue
            kl += p1 * (math.log(p1) - math.log(p2))
        if kl < 0.0: kl = 0.0
        return kl

    def kl_psdd_brute_force(self, other):
        """Brute-force (enumerative) computation of KL-divergence between two
        PSDDs"""
        assert self.vtree.var_count <= PSddNode._brute_force_limit
        if self.vtree.var_count != other.vtree.var_count:
            raise ValueError("PSDDs have different # of variables")
        kl = 0.0
        for model in self.models(self.vtree):
            pr1 = self.pr_model(model)
            if pr1 == 0.0: continue
            pr2 = other.pr_model(model)
            kl += pr1 * (math.log(pr1) - math.log(pr2))
        return kl

    def kl_psdd(self, other):
        """Compute KL-divergence between two PSDDs, recursively.  The PSDDs
        must have the same structure, but may have different parameters."""
        if self.is_false_sdd: return 0.0
        for n1, n2 in zip(self.as_positive_list(), other.as_positive_list()):
            assert n1.id == n2.id
            if n1.is_false_sdd:
                kl = 0.0
            elif n1.vtree.is_leaf():
                pr1 = [p / n1.theta_sum for p in n1.theta]
                pr2 = [p / n2.theta_sum for p in n2.theta]
                kl = PSddNode.kl(pr1, pr2)
            else:  # decomposition
                pels1, pels2 = n1.positive_elements, n2.positive_elements
                pr1 = [n1.theta[el] / n1.theta_sum for el in pels1]
                pr2 = [n2.theta[el] / n2.theta_sum for el in pels2]
                kl = sum(p1 * (p.data + s.data) for p1, (p, s) in zip(pr1, pels1))
                kl += PSddNode.kl(pr1, pr2)
            n1.data = kl
        return kl

    def kl_psdd_alt(self, other):
        """Alternative computation of the KL-divergence between two PSDDs.
        The PSDDs must have the same structure, but may have different
        parameters.  This one uses node marginals to compute the KL."""
        self.marginals()
        kl = 0.0
        for n1, n2 in zip(self.as_positive_list(), other.as_positive_list()):
            assert n1.id == n2.id
            if n1.is_false_sdd or n1.pr_node == 0.0:
                continue
            elif n1.vtree.is_leaf():
                pr1 = [p / n1.theta_sum for p in n1.theta]
                pr2 = [p / n2.theta_sum for p in n2.theta]
                kl += n1.pr_node * PSddNode.kl(pr1, pr2)
            else:  # decomposition
                pr1 = [n1.theta[el] / n1.theta_sum for el in n1.positive_elements]
                pr2 = [n2.theta[el] / n2.theta_sum for el in n2.positive_elements]
                kl += n1.pr_node * PSddNode.kl(pr1, pr2)
        return kl

    ########################################
    # SIMULATE A PSDD
    ########################################

    @staticmethod
    def sample(pr, z=1.0):
        """If input is a list of tuples (item,probability), randomly return an
        item based according to their probability"""
        q = random.random()
        cur = 0.0
        for item, p in pr:
            cur += p / z
            if q <= cur:
                return item
        return item

    def simulate(self, inst=None, seed=None):
        """Draw a model from the distribution induced by the PSDD"""
        assert not self.is_false()
        if seed is not None: random.seed(seed)
        if inst is None: inst = InstMap()

        if self.is_true():
            p = self.theta[0] / self.theta_sum
            val = 0 if random.random() < p else 1
            inst[self.vtree.var] = val
        elif self.is_literal():
            val = 0 if self.literal < 0 else 1
            inst[self.vtree.var] = val
        else:
            pr = iter(self.theta.items())
            p, s = PSddNode.sample(pr, z=self.theta_sum)
            p.simulate(inst=inst)
            s.simulate(inst=inst)

        return inst

    ########################################
    # LEARN (COMPLETE DATA)
    ########################################

    def log_likelihood(self, data):
        """Computes the log likelihood

            log Pr(data | theta)
        """
        return sum(cnt * math.log(self.pr_model(inst)) for inst, cnt in data)

    def log_posterior(self, data, prior):
        """Computes the (unnormalized) log posterior:

            log Pr(theta | data) 
              = log Pr(data | theta) + log Pr(theta) - log Pr(data)

        but we leave out the - log Pr(data) term.
        """
        return self.log_likelihood(data) + prior.log_prior(self)

    def learn(self, data, prior, verbose=False):
        """Given a complete dataset and a prior, learn the parameters of a
        PSDD"""
        prior.initialize_psdd(self)
        n = len(data)
        for i, (inst, count) in enumerate(data):
            if verbose and (n - i - 1) % max(1, (n / 10)) == 0:
                print("%3.0f%% done" % (100.0 * (i + 1) / n))
            # mark satisfying sub-circuit
            self.is_model_marker(inst, clear_bits=False, clear_data=False)
            self._increment_follow_marker(float(count))
            self.clear_bits(clear_data=True)

    def _increment_follow_marker(self, count):
        """Increment the PSDD parameters by following sub-circuit markers"""
        assert self.data is not None
        queue = [self]
        while queue:
            node = queue.pop()
            node.theta[node.data] += count
            node.theta_sum += count
            if node.is_decomposition():
                queue.append(node.data[0])  # prime
                queue.append(node.data[1])  # sub


########################################
# SUB-CIRCUIT
########################################

class SubCircuit:
    """Sub-Circuit's of PSDD models"""

    def __init__(self, node, element, left, right):
        self.node = node
        self.element = element
        self.left = left
        self.right = right

    def __repr__(self):
        pr = self.node.theta[self.element] / self.node.theta_sum
        vt_id = self.node.vtree.id
        node_id = self.node.id
        if self.node.is_decomposition():
            p_id = self.element[0].id
            s_id = self.element[1].id
            return "vt_%d: %d (%d/%d) %.4f" % (vt_id, node_id, p_id, s_id, pr)
        else:
            return "vt_%d: %d (%d) %.4f" % (vt_id, node_id, self.element, pr)

    def print_subcircuit(self):
        print(self)
        if self.node.is_decomposition():
            self.left.print_subcircuit()
            self.right.print_subcircuit()

    @staticmethod
    def induce_sub_circuit(inst, node):
        node.is_model_marker(inst, clear_bits=False, clear_data=False)
        subcircuit = SubCircuit._induce_sub_circuit(node)
        node.clear_bits(clear_data=True)
        return subcircuit

    @staticmethod
    def _induce_sub_circuit(node):
        left, right = None, None
        element = node.data
        if element is not None:
            if node.is_decomposition():
                prime, sub = element
                left = SubCircuit._induce_sub_circuit(prime)
                right = SubCircuit._induce_sub_circuit(sub)
        return SubCircuit(node, element, left, right)

    def probability(self):
        pr = self.node.theta[self.element] / self.node.theta_sum
        if self.left is not None:
            pr *= self.left.probability()
            pr *= self.right.probability()
        self.pr = pr
        return pr

    def node_of_vtree(self, vtree):
        my_id = self.node.vtree.id
        target_id = vtree.id
        if my_id == target_id:
            return self
        elif target_id < my_id:
            return self.left.node_of_vtree(vtree)
        else:  # elif my_id > target_id:
            return self.right.node_of_vtree(vtree)


########################################
# k-BEST ENUMERATION
########################################

class PSddEnumerator(SddEnumerator):
    """Manager for k-best MPE enumeration."""

    @staticmethod
    def _element_update(element_enum, inst):
        """This is invoked after inst.concat(other)"""
        element = (element_enum.prime, element_enum.sub)
        parent = element_enum.parent
        theta = parent.theta[element] / parent.theta_sum
        inst.mult_weight(theta)

    def __init__(self, vtree):
        SddEnumerator.__init__(self, vtree)
        self.terminal_enumerator = PSddTerminalEnumerator


class PSddTerminalEnumerator(SddTerminalEnumerator):
    """Enumerator for terminal PSDD nodes"""

    def __init__(self, node, vtree):
        self.heap = []

        if node.is_false():
            pass
        elif node.is_literal():
            # weight = node.theta[node.literal > 0]/node.theta_sum
            inst = WeightedInstMap.from_literal(node.literal, weight=1.0)
            heapq.heappush(self.heap, inst)
        if node.is_true():
            weight = node.theta[0] / node.theta_sum
            inst = WeightedInstMap.from_literal(-vtree.var, weight=weight)
            heapq.heappush(self.heap, inst)

            weight = node.theta[1] / node.theta_sum
            inst = WeightedInstMap.from_literal(vtree.var, weight=weight)
            heapq.heappush(self.heap, inst)
