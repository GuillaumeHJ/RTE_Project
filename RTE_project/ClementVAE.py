import tensorflow.keras as tfk
import tensorflow as tf
import New_load as NL

path = "data/"
x_train, x_val, sc = NL.load(path)

encoder_dims = [48, 48, 24, 12]
decoder_dims = [12, 24]
latent_dims = 3

# create VAE

# encoder
x_inputs = tfk.Input(shape=(48,))
x = x_inputs
for e_dim in encoder_dims:
    x = tfk.layers.Dense(e_dim, activation="relu")(x)

mu = tfk.layers.Dense(latent_dims, activation="linear")(x)
log_sigma = tfk.layers.Dense(latent_dims, activation="linear")(x)

encoder = tfk.Model(x_inputs, [mu, log_sigma], name="encoder")

# decoder

z_inputs = tfk.Input(shape=(latent_dims,))
z = z_inputs
for d_dim in decoder_dims:
    z = tfk.layers.Dense(d_dim, activation="relu")(z)
x_hat = tfk.layers.Dense(48, activation="linear")(z)

decoder = tfk.Model(z_inputs, x_hat, name="decoder")

# vae
z_mu, z_log_sigma = encoder(x_inputs)
# reparametrization trick
epsilon = tf.random.normal(shape=[latent_dims])
z_latent = z_mu + tf.exp(0.5 * z_log_sigma) * epsilon
x_output = decoder(z_latent)


def normal_kl_loss(mu, log_sigma):
    kl = tf.math.reduce_mean(0.5 * tf.math.reduce_sum((tf.exp(log_sigma) + tf.square(mu)) - 1 - log_sigma, axis=-1))
    return kl


vae = tfk.Model(x_inputs, x_output, name="vae")

vae.add_loss(normal_kl_loss(z_mu, z_log_sigma))

vae.compile(loss="mean_squared_error",
            optimizer="Adam")

# training
vae.fit(x=x_train[:, :48], y=x_train[:, :48], epochs=120, validation_split=0.1, verbose=0)

x_encoded, _ = encoder(x_train[:, :48])
