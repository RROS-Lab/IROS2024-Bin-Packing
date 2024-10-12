# IROS2024-Bin-Packing
This repository is the official implementation of [Simulation-Assisted Learning for Efficient Bin-Packing of Deformable Packages in a Bimanual Robotic Cell](https://sites.google.com/usc.edu/bimanual-binpacking/home).

<img src="./assets/bi-manual-binpacking.gif" width="500px"></img>

This code is intended for reproduction purposes only. Current implementation does not support extensions. The objective of this repository is to provide the reader with the implementation details of the learning framework proposed in the IROS 2024 paper.
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

*  in-project: /path/to/repo/IROS2024-Bin-Packing/.venv
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

TODO

## Failure Classification Model
The role of this model is identify cases that are considered unrecoverable in terms of packing score.
An example dataset is also provided for reproduction of our studies (You can download it from [https://drive.google.com/file/d/1uVUyZfa5tIXsbdR-5V-E6-_dTNmQGyZw/view?usp=drive_link](https://drive.google.com/file/d/1uVUyZfa5tIXsbdR-5V-E6-_dTNmQGyZw/view?usp=drive_link)).
Place the two csv files in the following directory: /src/packing_score_predictor/dataset/processed_data/ .

In order to perform training kindly run the following command in terminal:
```Failure Classifier
cd src/packing_score_predictor
python -f train_classifier.py -f training_params_classifier.json
```
This will generate the model for detecting failure cases. Running the script generates a confusion matrix representative of the final classification performance.

## Packing Score Prediction Model
The packing score model is the model 1 (Suction Robot) and model 2 (Paddle Robot/In-bin Robot) state-action model that predicts the packing score for given state and actions.
An example dataset is also provided for reproduction of our studies.

In order to perform training of the model with the given dataset, run the following command:
```Inferencing Packing Score Prediction Model
python train_score_predictor.py -f training_params_score.json
```

## Action Prediction Module (Optimizer)
The action prediction module is the online optimizer that computes the actions for the robots. 
We've provided a pre-trained checkpoint for the packing score predictor.
The action prediction module takes this pre-trained checkpoint as the input and predicts the delta actions for the bi-manual robots.

In order to perform inference. You can update the state values, based on the dataset values and perform inference. This can be done by varying line 160 in action_predictor.py. For state definitions refer to lines 73-92. in action_predictor.py. 

Run the following command for inference:
```Action Prediction
cd src/action_predictor
python action_predictor.py -f predictor_config.json
```


## Citation

```
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