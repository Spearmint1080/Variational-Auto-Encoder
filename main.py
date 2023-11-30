import pickle as pk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from keras.models import Model
from keras.optimizers.legacy import Adam
from keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    Input,
    Lambda,
    LSTM,
    RepeatVector,
    Activation,
)

# from src import *
from keras import backend as K
from src.loss_layers import *

# CustomVariationalLayer
from src.util import *  # sampling, create_model_checkpoint
from src.equation_generation import *
from pathlib import Path

from keras import backend as K

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


def kl_loss(x, x_decoded_mean):
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(
            np.ones((batch_size, max_length_of_equation)), tf.float32
        )

    def vae_loss(self, x, x_decoded_mean):
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(
            tfa.seq2seq.sequence_loss(
                x_decoded_mean,
                labels,
                weights=self.target_weights,
                average_across_timesteps=False,
                average_across_batch=False,
            ),
            axis=-1,
        )  # ,
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return K.ones_like(x)

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"]
print(kerasBKED)

tf.compat.v1.disable_eager_execution()

path_to_training_data = Path.cwd().joinpath("training_data", "reaction_data.pkl")
with open("training_data/reaction_data.pkl", "rb") as f:
    data = pk.load(f)


dataset = data["training_data"]
tokenizer = data["tokenizer"]


number_of_equations = dataset.shape[0]
np.random.shuffle(dataset)
# current training method is unstable if batch size is not a fraction of the length of training data
training = dataset[:5900].astype(np.int32)
test = dataset[5900:6900].astype(np.int32)


batch_size = 25
epochs = 500
max_length_of_equation = 342  # len(dataset[1])
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
number_of_letters = len(tokenizer.word_index)
learning_rate = 1e-5
# optimizer = Adam(learning_rate=learning_rate)
optimizer = Adam()


# Start model construction
input = Input(shape=(max_length_of_equation,))
embedded_layer = Embedding(
    number_of_letters, intermediate_dimension, input_length=max_length_of_equation
)(input)
latent_vector = Bidirectional(
    LSTM(intermediate_dimension, return_sequences=False, recurrent_dropout=0.2),
    merge_mode="concat",
)(embedded_layer)
z_mean = Dense(latent_dimension)(latent_vector)
z_log_var = Dense(latent_dimension)(latent_vector)


z = Lambda(sampling, output_shape=(latent_dimension,))(
    [z_mean, z_log_var, batch_size, latent_dimension, epsilon_std]
)
repeated_context = RepeatVector(max_length_of_equation)
decoder_latent_vector = LSTM(
    intermediate_dimension, return_sequences=True, recurrent_dropout=0.2
)
decoder_mean = Dense(
    number_of_letters, activation="linear"
)  # softmax is applied in the seq2seqloss by tf #TimeDistributed()
latent_vector_decoded = decoder_latent_vector(repeated_context(z))
print(latent_vector_decoded)
input_decoded_mean = decoder_mean(latent_vector_decoded)


loss_layer = CustomVariationalLayer()([input, input_decoded_mean])
vae = Model(input, [loss_layer])


vae.compile(optimizer=optimizer, loss=[zero_loss], metrics=[kl_loss])
vae.summary()
#tf.keras.utils.plot_model(vae)
tf.compat.v1.experimental.output_all_intermediates(True)
# ======================= Model training ==============================#
checkpointer = create_model_checkpoint("models", "agoras_checkpoints")

vae.fit(
    training,
    training,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test, test),
    callbacks=[checkpointer]
)

print(K.eval(vae.optimizer.learning_rate))
K.set_value(vae.optimizer.learning_rate, learning_rate)


path = Path.cwd().joinpath("models")
vae.save_weights(path.joinpath("agoras_vae.h5"))


# build a model to project inputs on the latent space
encoder = Model(input, z_mean)
encoder.save(path.joinpath("agoras_encoder.h5"))

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dimension,))
_h_decoded = decoder_latent_vector(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation("softmax")(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)
generator.save(path.joinpath("agoras_generator.h5"))


def generate_indices_from_encoded_space(encoded_vector, generator):
    reconstructed_equation = generator.predict(encoded_vector, batch_size=1)
    reconstruct_indices = np.apply_along_axis(np.argmax, 1, reconstructed_equation[0])
    return np.max(np.apply_along_axis(np.max, 1, reconstruct_indices[0]))


def check_validation_equation(index_number, data, encoder, generator, tokenizer):
    encoded_equation = encoder.predict(data[index_number : index_number + 2, :])
    smiles_indices = generate_indices_from_encoded_space(encoded_equation, generator)
    smiles_equation = list(np.vectorize(tokenizer.index_word.get)(smiles_indices))
    print(f"The reconstructed equation is {''.join(smiles_equation)}")
    original_equation = list(np.vectorize(tokenizer.index_word.get)(data[index_number]))
    print(f"The original equation is {''.join(original_equation)}")


check_validation_equation(200, test, encoder, generator, tokenizer)


# ====================== Example ====================================#
equation1 = ["C=O ~ [OH-] > [CH]=O ~ O"]
equation2 = ["C.CCCO ~ O=O > CC(=O)C(C)=O ~ [OH-]"]

homology = calculate_equations_homology(
    equation1, equation2, 5, encoder, pad_equation=True
)
new_equations(homology, generator, latent_dimension, max_length_of_equation, tokenizer)

# A list of common errors to help eliminate bad equations from the generated set.

new_equations = generate_equations(
    dataset,
    500000,
    5,
    generator,
    encoder,
    latent_dimension,
    max_length_of_equation,
    tokenizer,
)
