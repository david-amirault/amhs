# amhs
This repository includes models, scripts, and utilities related to the automated material handling system (AMHS) of a modern semiconductor Fab. The pipeline in this repository allows the user to simulate traffic in an AMHS, create datasets based on the simulated traffic, and train models to on these created datasets.

This repository includes the following:

 - `archive/`: depricated simulations. Ignore.
 - `fig/`: schematic of the original Graph WaveNet architecture.
 - `reference/`: `.txt` documents explaining acronyms, open-ended research questions, and the software requirements for this dataset. The requirements may be installed using Python 3 Pip via `python3 -m pip install <package name>`.
 - `AMHS_realistic_sim.ipynb`: the Jupyter notebook that simulates traffic in an AMHS. This notebook includes utilities for creating visualizations of the traffic that is simulated in the AMHS.
 - `AMHS_preprocess.ipynb`: dataset preprocessing utilities. This Jupyter notebook serves as the intermediary between `AMHS_realistic_sim.ipynb` and the https://github.com/sshleifer/Graph-WaveNet codebase on which the models in this repository are built.
 - `gen_adj_mx.py`: utilities for Graph WaveNet adjacency matrix creation.
 - `generate_training_data.py`: utilities to create Graph WaveNet datasets for training in PyTorch.
 - `exp_results.py`: utilities for summarizing the performance of trained models with a concise set of metrics.
 - `train.py`, `test.py`, `engine.py`, `model.py`, `util.py`: the Graph WaveNet model and derived modifications for very large graphs.
 - `regression.py`: a post-processing utility for Graph WaveNet models. A model trained on a graph preprocessed with a Graph partition (https://en.wikipedia.org/wiki/Graph_partition) can use this utility to make predictions on the original graph using a linear model.
 - `makefile`: for GNU Make to execute the long command-line arguments used by `gen_adj_mx.py`, `generate_training_data.py`, `train.py`, and `regression.py`.

We would like to credit the original Graph WaveNet authors (https://arxiv.org/abs/1906.00121) and the authors of "Incrementally Improving Graph WaveNet Performance on Traffic Prediction" (https://arxiv.org/abs/1912.07390) for their model and code base respectively.
