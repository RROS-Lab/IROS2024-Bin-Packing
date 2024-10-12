#!/bin/sh

# The Python script you want to run
PYTHON_MODULE="src.sim_env.env"

# The base name for your data files
BASE_NAME="data_"

# The starting index for your file names
START_INDEX=1

# The number of times you want to run the Python script
NUM_RUNS=25

i=0
while [ $i -lt $NUM_RUNS ]; do
    # Calculate the current index
    CURRENT_INDEX=$((START_INDEX + i))

    # Construct the current data file name
    DATA_FILE_NAME="${BASE_NAME}${CURRENT_INDEX}.csv"  # Change .txt to your actual file extension

    # Run the Python script with the current data file name as an argument
    python3 -m $PYTHON_MODULE --data_filename $DATA_FILE_NAME

    # Increment the counter
    i=$((i + 1))

    # Optionally, add a sleep command if you want to pause between runs
    # sleep 1  # Sleep for 1 second
done
