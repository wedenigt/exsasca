# ExSASCA: Exact Soft Analytical Side-Channel Attacks using Tractable Circuits

This is the code repository that contains the experimental code for our paper "Exact Soft Analytical Side-Channel Attacks using Tractable Circuits".

## Setup

We recommend that you create a conda environment with Python 3.9.7:
```bash
conda create -n exsasca python=3.9.7
conda activate exsasca
```
and install the required packages:
```bash
pip install -r requirements.txt
```

To download the compiled SDD (about 1GB in size), you can run the following command:
```bash
cd compilation
python ./download_sdd.py
```

To compile the SDD yourself, you can run the following command:
```bash
cd compilation
./run_compilation.sh
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{wedenig2024exsasca,
  title     = {Exact Soft Analytical Side-Channel Attacks using Tractable Circuits},
  author    = {Wedenig, Thomas and Nagpal, Rishub and Cassiers, Ga{\"e}tan and
               Mangard, Stefan and Peharz, Robert},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2501.13748}
}
```

Machine-readable metadata for this repository is in [`CITATION.cff`](CITATION.cff);
metadata for the archived Zenodo release is in [`.zenodo.json`](.zenodo.json).

## Funding

This project has received funding from the European Union's EIC Pathfinder Challenges
2022 programme under grant agreement No 101115317
([NEO](https://cordis.europa.eu/project/id/101115317), *Next Generation Molecular Data
Storage*). Views and opinions expressed are however those of the author(s) only and do
not necessarily reflect those of the European Union or European Innovation Council.
Neither the European Union nor the European Innovation Council can be held responsible
for them.
