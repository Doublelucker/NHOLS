# NHOLS
The Nonlinear Higher Order Label Spreading algorithm Julia package

This library does semi-supervised learning algorithm **NHOLS** or **LS** (Label Spreading) on a number of prepared datasets or on datasets provided by a user. The code is supplementary to the paper on [NHOLS](https://arxiv.org/abs/2006.04762)

If you use this code, please cite our work
@inproceedings{tudisco2020nonlinear,
  title={Nonlinear higher-order label spreading},
  author={Tudisco, Francesco and Benson, Austin R and Prokopchik, Konstantin},
  booktitle={Proceedings of The Web Conference (WWW)},
  year={2021}
}


# Requirements

You need [Julia](https://julialang.org/downloads/) installed.  
Everything else will be installed with the package itself.

To start working execute  `activate .` inside the repository.  
Then switch to `pkg` mode and execute `build` command.

# Functonality

The main script is `train.jl`, it picks up the information from `config.yaml`.   

Here is the detailed description of what's in the config file:

> **mode**: `string`: `LS`, `HOLS` or `both`, depends on what algorithm you want to execute. the last option executes both LS and HOLS algorithms at the same time.

> **α**: a `float` value in (0, 1) interval,\
    &nbsp;&nbsp;&nbsp; range as a list of size 3: `[start: float, stop: float, length: int]` (equivalent of a Python `numpy.linspace`)\
    &nbsp;&nbsp;&nbsp; a list of values of any size `list[float]`

> **β** (not for **LS**): same as α

Note: Both α and β are relaxation parameters mentioned in the paper.

> **ε**: `float` value, should be picked quite small, small perturbation to the input vector

> **kn**: `int` the number of neighbors in the graph built around the data

> **noise**: a `number` value - the amount of added noise to the weight matrix

> **tolerance**: a `number` value - stopping criteria

> **dataset**: 'string' - the name of dataset, detailed explanation in the next section below

> **distance** (not for **LS**): `string` - the function defined in the `src/similarity_knn.jl` file, it's possible to add custom functions, if there's a need.

> **binary**: `bool` - if `true` - adjacency matrix and triangle tensor, weight matrix and custom tensor with `distance` function specified otherwise.

> **percentage_of_known_labels** - can be specified in the same way as α and β parameters,


> **balanced**: `bool` - if `true` - the amount of known labels from classes is taken equally between all the  classes (**percentage_of_known_labels** / **number_of_classes**), and differently otherwise

> **data_type**: `string`: `matrix` or `points`. Choose according to your data: adjacency matrix or a features matrix

> **num_trials**: `int`: the known labels are randomized at each run, this parameter allows to run the algorithm several times to get an averaged prediction across all the runs.

> **mixing_functions** (not for **LS**): 'string': functions `f` from the algorithm in the paper, that are separated with a comma\
&nbsp;&nbsp;&nbsp; the defined functions are in the 'src/functions.jl' file.\
&nbsp;&nbsp;&nbsp; user may add new functions to that file


# Datasets

Some are loaded via the package, the rest is in the **data** folder.

Available:

> datasets from [UCI](https://archive.ics.uci.edu/ml/datasets.php) website (the `Julia` package loads them)

> fb100 folder with two datasets: Caltech36 and Rice31

> optdigits and pendigits in a CSV format (first n-1 columns comprise a feature matrix, the last one is labels)

> mnist and fashion-mnist from [MLDatasets](https://github.com/JuliaML/MLDatasets.jl) package

Custom:

If you wish to use your dataset you have to put them in a **custom** folder in a folder that has the name of your dataset (you can see the examples in a **data** folder).

These formats are available:

> .xlsx, .xls, .csv are accepted, the file should be structured as, e.g. **optdigits.csv**

> .npy format is accepted, it requires two files: X.npy and y.npy. X can be either a feature matrix or a adjacency matrix (if you pass a weight matrix it will be converted into an adjacency matrix).

This is it for now, I will be adding more formats in the next days.


# Notes

This is not a final version of the project. Apart from more options on datasets, I will be adding more kinds of experiments for the algorithm (e.g. **Stochastic Block Modeling**) and **Cross-Validation** features for tuning.
