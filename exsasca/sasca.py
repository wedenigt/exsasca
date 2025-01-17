import torch
import numpy as np
from exsasca.utils.aes import xtime_lut
from exsasca.utils.sdd import get_mc_vars
from scalib.attacks import FactorGraph, BPState

def sasca_mixcol_inference(leakage_pmfs: torch.tensor, num_bp_iterations=10,
                           is_log=True, only_parameterize_first_n_vars=False):
    """
    leakage_pmfs: a matrix with log-pmfs on the rows
    """
    # assert leakage_pmfs.exp().sum(dim=-1).allclose(torch.ones(leakage_pmfs.shape[0], device=leakage_pmfs.device))
    vars = get_mc_vars()

    if only_parameterize_first_n_vars:
        assert leakage_pmfs.shape[0] == only_parameterize_first_n_vars
    else:
        assert leakage_pmfs.shape[0] == len(vars)

    # convert to float64 probabilities (and to numpy array)
    leakage_pmfs = leakage_pmfs.double().detach().cpu().numpy()
    if is_log:
        leakage_pmfs = np.exp(leakage_pmfs)

    graph_desc = f"""
                NC 256
                TABLE xtime

                VAR MULTI x1
                VAR MULTI x2
                VAR MULTI x3
                VAR MULTI x4

                VAR MULTI x12
                VAR MULTI x23
                VAR MULTI x34
                VAR MULTI x41

                VAR MULTI g

                VAR MULTI xx12
                VAR MULTI xx23
                VAR MULTI xx34
                VAR MULTI xx41

                VAR MULTI xx12g
                VAR MULTI xx23g
                VAR MULTI xx34g
                VAR MULTI xx41g

                VAR MULTI xm1
                VAR MULTI xm2
                VAR MULTI xm3
                VAR MULTI xm4

                # ------------------------

                PROPERTY x12 = x1 ^ x2
                PROPERTY x23 = x2 ^ x3
                PROPERTY x34 = x3 ^ x4
                PROPERTY x41 = x4 ^ x1

                PROPERTY g = x12 ^ x34

                PROPERTY xx12 = xtime[x12]
                PROPERTY xx23 = xtime[x23]
                PROPERTY xx34 = xtime[x34]
                PROPERTY xx41 = xtime[x41]

                PROPERTY xx12g = xx12 ^ g
                PROPERTY xx23g = xx23 ^ g
                PROPERTY xx34g = xx34 ^ g
                PROPERTY xx41g = xx41 ^ g

                PROPERTY xm1 = x1 ^ xx12g
                PROPERTY xm2 = x2 ^ xx23g
                PROPERTY xm3 = x3 ^ xx34g
                PROPERTY xm4 = x4 ^ xx41g
                """

    graph = FactorGraph(graph_desc, {'xtime': xtime_lut})
    num_traces = 1
    bp = BPState(graph, num_traces)

    for i, var in enumerate(vars):
        bp.set_evidence(var, leakage_pmfs[i:i + 1, :])
        if only_parameterize_first_n_vars and i == only_parameterize_first_n_vars - 1:
            break

    bp.bp_loopy(it=num_bp_iterations, initialize_states=True)
    x1_dist, x2_dist, x3_dist, x4_dist = bp.get_distribution("x1"), bp.get_distribution("x2"), \
        bp.get_distribution("x3"), bp.get_distribution("x4")

    assert bp.is_cyclic(), 'Graph is supposed to be cyclic when modeling MixColumns (due to x41 = x4 ^ x1)'

    return x1_dist, x2_dist, x3_dist, x4_dist
