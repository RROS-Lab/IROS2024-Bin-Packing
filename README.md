# Simulation-Assisted Learning for Efficient Bin-Packing of Deformable Packages in a Bimanual Robotic Cell

[Omey M. Manyar<sup>1,*</sup>](https://omey-manyar.com/), [Hantao Ye<sup>1,*</sup>](https://hantao-ye.github.io/), [Meghana Sagare<sup>1</sup>](https://www.linkedin.com/in/msagare), [Siddharth Mayya<sup>2</sup>](https://www.amazon.science/author/siddharth-mayya), [Fan Wang<sup>2</sup>](https://www.amazon.science/author/fan-wang), [Satyandra K. Gupta<sup>1</sup>](https://sites.usc.edu/skgupta/)

<sup>1</sup>University of Southern California, <sup>2</sup>Amazon Robotics, <sup>*</sup>Equal contribution

IROS 2024

**[Project Page](https://sites.google.com/usc.edu/bimanual-binpacking/home) | [Video](https://www.youtube.com/watch?v=l6VeTOpoE5A) | [Paper](https://www.amazon.science/publications/simulation-assisted-learning-for-efficient-bin-packing-of-deformable-packages-in-a-bimanual-robotic-cell)**

![Bimanual-BinPacking](./assets/bi-manual-binpacking.gif)

This repository is the official implementation of the paper. This code is intended for reproduction purposes only. Current implementation does not support extensions. The objective of this repository is to provide the reader with the implementation details of the learning framework proposed in the IROS 2024 paper.

- [Simulation-Assisted Learning for Efficient Bin-Packing of Deformable Packages in a Bimanual Robotic Cell](#simulation-assisted-learning-for-efficient-bin-packing-of-deformable-packages-in-a-bimanual-robotic-cell)
  - [Environment Setup](#environment-setup)
    - [Pre-requirements](#pre-requirements)
    - [Step 1: Install PDM](#step-1-install-pdm)
    - [Step 2: Clone the Repository](#step-2-clone-the-repository)
    - [Step 3: Create a Virtual Environment](#step-3-create-a-virtual-environment)
    - [Step 4: Activate the Virtual Environment](#step-4-activate-the-virtual-environment)
    - [Optional Step: Manage Dependencies](#optional-step-manage-dependencies)
  - [Simulation](#simulation)
    - [Visualization](#visualization)
    - [Data generation](#data-generation)
  - [Failure Classification Model](#failure-classification-model)
  - [Packing Score Prediction Model](#packing-score-prediction-model)
  - [Action Prediction Module (Optimizer)](#action-prediction-module-optimizer)
  - [Citation](#citation)

## Environment Setup

To simplify the process of setting up the development environment, we use **[PDM](https://pdm-project.org/en/latest/)** for Python package management and virtual environment management.

### Pre-requirements

- Ubuntu >= 20.04 (Didn't test on other platforms)
- Python >= 3.8 (For PDM installation)
- Anaconda/Miniconda (For virtualenv creation)

### Step 1: Install PDM

To install PDM, run the following command:

```shell
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

### Step 2: Clone the Repository

Clone the project repository and navigate into the project folder:

```shell
git clone https://github.com/RROS-Lab/IROS2024-Bin-Packing.git
cd IROS2024-Bin-Packing
```

### Step 3: Create a Virtual Environment

Next, create a Python 3.10 virtual environment using PDM and select conda as backend:

```shell
pdm venv create --with conda 3.10
```

To verify the virtual environment was created successfully, use:

```shell
pdm venv list
```

You should see output like:

```shell
Virtualenvs created with this project:

*  in-project: /path/to/IROS2024-Bin-Packing/.venv
```

Here, `in-project` is the default name of the virtual environment. If you'd like to specify a custom name for the environment, use:

```shell
pdm venv create --with conda --name my-env-name 3.10
```

### Step 4: Activate the Virtual Environment

To activate the virtual environment and install dependencies, run:

```shell
eval $(pdm venv activate in-project)
pdm install
```

All necessary dependencies will be installed after running the command above.

### Optional Step: Manage Dependencies

```shell
pdm add requests   # add requests
pdm add requests==2.25.1   # add requests with version constraint
pdm add requests[socks]   # add requests with extra dependency
pdm add "flask>=1.0" flask-sqlalchemy   # add multiple dependencies with different specifiers
```

Please follow [PDM documentation](https://pdm-project.org/en/latest/usage/dependency/) for details

## Simulation

### Visualization

```shell
$ pwd 
# ensure your current working dir is the repo
path/to/IROS2024-Bin-Packing
$ python -m src.sim_env.env -h environment
# The help information for running simulation 
usage: env.py [-h] [--data_filename DATA_FILENAME] [--visualize VISUALIZE] [--num_threads NUM_THREADS]
              [--num_scenes NUM_SCENES]

options:
  -h, --help            show this help message and exit
  --data_filename DATA_FILENAME
                        Filename to save data to (default: data.csv)
  --visualize VISUALIZE
                        Visualize the scene (default: False)
  --num_threads NUM_THREADS
                        Number of threads to use, not working when visualize enabled (default: 12)
  --num_scenes NUM_SCENES
                        Number of scenes to generate (default: 400)
$ python -m src.sim_env.env --visualize True --num_scenes 10
# will create and visualize 10 scenes using MuJoCo passive_launcher and 
# store data in src/sim_env/data/sim_exp/data.csv
```

### Data generation

For boosting data generation, we also provide a shell script to execute

```shell

sh ./src/sim_env/scripts/multiple_run.sh
# will create 25 data files generated by env.py in its default config and 
# store data files in src/sim_env/data/sim_exp/data_{idx}.csv
```

## Failure Classification Model

The role of this model is identify cases that are considered unrecoverable in terms of packing score.

An example dataset is also provided for reproduction of our studies ([google drive link](https://drive.google.com/file/d/1uVUyZfa5tIXsbdR-5V-E6-_dTNmQGyZw/view?usp=drive_link)). Place the two csv files in the following directory: `./src/packing_score_predictor/dataset/processed_data/`.

In order to perform training kindly run the following command in terminal:

```shell
cd src/packing_score_predictor
python -f train_classifier.py -f training_params_classifier.json
```

This will generate the model for detecting failure cases. Running the script generates a confusion matrix representative of the final classification performance.

## Packing Score Prediction Model

The packing score model is the model 1 (Suction Robot) and model 2 (Paddle Robot/In-bin Robot) state-action model that predicts the packing score for given state and actions.

An example dataset is also provided for reproduction of our studies.

In order to perform training of the model with the given dataset, run the following command:

```shell
cd src/packing_score_predictor
python train_score_predictor.py -f training_params_score.json
```

## Action Prediction Module (Optimizer)

The action prediction module is the online optimizer that computes the actions for the robots.

We've provided a pre-trained checkpoint for the packing score predictor. The action prediction module takes this pre-trained checkpoint as the input and predicts the delta actions for the bi-manual robots.

In order to perform inference. You can update the state values, based on the dataset values and perform inference. This can be done by varying line 160 in [action_predictor.py](./src/action_predictor/action_predictor.py). For state definitions refer to lines 73-92. in [action_predictor.py](./src/action_predictor/action_predictor.py).

Run the following command for inference:

```shell
cd src/action_predictor
python action_predictor.py -f predictor_config.json
```

## Citation

```BibTeX
@INPROCEEDINGS{manyar_iros_2024,
  author={Manyar, Omey M. and Ye, Hantao. and Sagare, Meghana and Mayya, Siddharth and Wang, Fan and Gupta, Satyandra K.},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Simulation-Assisted Learning for Efficient Bin-Packing of Deformable Packages in a Bimanual Robotic Cell}, 
  year={2024},
  month={Oct},
  address="Abu Dhabi, UAE", 
  volume={},
  number={},
  pages={},
  doi={}}
```
