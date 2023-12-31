import pickle as pk
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd


batch_size = 25
epochs = 500
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
learning_rate = 1e-5
max_length_of_equation = 342

# Get raw smiles dataset
path_to_raw_smiles_data = Path.cwd().joinpath(
    "training_data", "smiles", "training_data.pkl"
)
raw_smiles_equations = pk.load(open(path_to_raw_smiles_data, "rb"))

# Fit tokenizer onto the dataset. It's important to note that the hyperparameters are important here
tokenizer = Tokenizer(num_words=None, filters="", lower=False, char_level=True)
tokenizer.fit_on_texts(raw_smiles_equations)
training_sequences = tokenizer.texts_to_sequences(raw_smiles_equations)

# create padded training data that can be used by the VAE.
padded_training_sequences = pad_sequences(
    training_sequences, maxlen=None, padding="post", value=tokenizer.word_index.get(" ")
)

reaction_data = {"training_data": padded_training_sequences, "tokenizer": tokenizer}

path_to_save_data = Path.cwd().joinpath("training_data", "reaction_data.pkl")
with open(path_to_save_data, "wb") as f:
    pk.dump(reaction_data, f, protocol=pk.HIGHEST_PROTOCOL)
# print("done")
