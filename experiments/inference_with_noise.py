import argparse
import numpy as np
from exsasca.exhaustive_inference import InferenceMethod, exhaustive_inference_mixcol_pmfs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference in AES with noise on input PMFs."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="exhaustive",
        help="Inference method. Can be 'exhaustive' or 'sasca' (default: %(default)s).",
    )
    parser.add_argument(
        "--noise-alpha",
        type=float,
        default=0.0,
        help="Interpolation coefficient for adding uniform distributions to PMFs (default is no noise, i.e. alpha=0.0).",
    )
    parser.add_argument(
        "--noise-alpha-idx",
        type=int,
        help="[int] Interpolation coefficient for adding uniform distributions to PMFs (default is no noise, i.e. alpha=0.0).",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=128,
        help="Number of traces to compute PMFs (default: %(default)s).",
    )
    parser.add_argument(
        "--num-bp-iterations",
        type=int,
        default=50,
        help="Number of (loopy) belief propagation iterations (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-idx",
        type=int,
        default=0,
        help="Batch idx (default: %(default)s).",
    )
    parser.add_argument(
        "--mc-block",
        type=int,
        default=0,
        help="MixColumns Block idx (0,1,2,3) (default: %(default)s).",
    )
    parser.add_argument(
        "--just-max",
        action='store_true',
        help="Just compute the max (default: %(default)s).",
    )

    return parser.parse_args()


def main(method: InferenceMethod, noise_alpha: float, num_traces: int, num_bp_iterations: int,
         batch_idx: int, just_max: bool, mc_block: int):
    print(f'noise_alpha: {noise_alpha} | mc_block: {mc_block}')
    ranks = exhaustive_inference_mixcol_pmfs(inference_method=method,
                                             mc_block=mc_block,
                                             num_traces=num_traces,
                                             noise_alpha=noise_alpha,
                                             num_bp_iterations=num_bp_iterations,
                                             load_sasca=False,
                                             load_exact=False,
                                             g_range=range(256),
                                             testing=False,
                                             batch_idx=batch_idx,
                                             just_max=just_max)
    print(np.mean(ranks == 0))
    print(ranks)


if __name__ == '__main__':
    args = parse_args()
    if args.method == 'exhaustive':
        method = InferenceMethod.EXHAUSTIVE
    elif args.method == 'sasca':
        method = InferenceMethod.SASCA
    else:
        raise ValueError(f"Unknown inference method: {args.method}")

    if args.just_max:
        print('Warning: just_max is set to True. This will only compute the max and not the marginal PMFs.')

    if args.noise_alpha_idx is not None:
        ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        args.noise_alpha = ALPHAS[args.noise_alpha_idx]

    main(method=method,
         noise_alpha=args.noise_alpha,
         num_traces=args.num_traces,
         num_bp_iterations=args.num_bp_iterations,
         batch_idx=args.batch_idx,
         just_max=args.just_max,
         mc_block=args.mc_block)
