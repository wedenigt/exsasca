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