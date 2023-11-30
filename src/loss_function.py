from keras import backend as K


batch_size = 25
epochs = 500
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
learning_rate = 1e-5
max_length_of_equation = 342


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


def kl_loss(x, x_decoded_mean):
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss
