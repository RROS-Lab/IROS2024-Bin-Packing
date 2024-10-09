# IROS2024-Bin-Packing

## Environment Setup

This project uses **Python 3.10**. To simplify the process of setting up the development environment, we use **[PDM](https://pdm-project.org/en/latest/)** for Python package management and virtual environment management.

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

Next, create a Python 3.10 virtual environment using PDM:

```shell
pdm venv create 3.10
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
pdm venv create --name my-env-name 3.10
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
