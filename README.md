# Autonomous Generation of Reactions and Species-VAE
Machine Learning Project on generating Chemical Reactions using Auto Encoder Networks

## Dependencies
python = ">=3.7.1, <3.9"

keras = "^2.6.0"

numpy = "1.19.5"

tensorflow = "^2.6.0"

tensorflow-addons = "^0.14.0"

scipy = "^1.7.1"

pathlib = "^1.0.1"

ipykernel = "^6.4.1"

pandas = "^1.3.3"

## Installation

Step 1.

Clone the repository locally

Step 2.

Create a virtual environment with the necessary dependencies.

Step 3.

Create the tokenized dataset by running src/create_tokenized_dataset.py


## Train the model

To train the model and generate new equations run

python main.py

This will save the model, encoder and generator separately inside the `models` directory. 
This allows for the user to load the generator at any time to continue generating new equations.
