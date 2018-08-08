import tensorflow as tf

LEARNING_RATE = 1E-3

def residual_conv_block(x, num_channels, kernel_size, strides=1):

    residual = tf.layers.conv2d(x, filters=num_channels, kernel_size=1, padding='same', activation=tf.nn.relu)
    residual = tf.layers.batch_normalization(residual)

    conv1 = tf.layers.conv2d(x, filters=num_channels, kernel_size=kernel_size, strides=strides, padding='same', activation=tf.nn.relu)
    batch1 = tf.layers.batch_normalization(conv1)
    conv2 = tf.layers.conv2d(batch1, filters=num_channels, kernel_size=kernel_size, strides=strides, padding='same', activation=tf.nn.relu)
    batch2 = tf.layers.batch_normalization(conv2)
    block = tf.nn.relu(batch2 + residual)
    return tf.layers.batch_normalization(block)

def residual_deconv_block(x, num_channels, kernel_size, strides=1, padding='same', activation=tf.nn.relu):

    # residual = tf.layers.conv2d(x, filters=num_channels, kernel_size=1, strides=strides, padding='same', activation=tf.nn.relu)
    # residual = tf.layers.batch_normalization(residual)

    deconv1 = tf.layers.conv2d_transpose(x, filters=num_channels, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
    batch1 = tf.layers.batch_normalization(deconv1)
    deconv2 = tf.layers.conv2d_transpose(batch1, filters=num_channels, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
    batch2 = tf.layers.batch_normalization(deconv2)
    return batch2
    # block = tf.nn.relu(batch2 + residual)
    # return tf.layers.batch_normalization(block)


class Network(object):
    # Create model
    def __init__(self, waveform):
        self.latent_vec_size = 64
        self.frame_length = 160
        self.frame_step = 100
        self.fft_length = 256
        self.audio_length = 16000
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.waveform = waveform
        tf.summary.audio("waveform", self.waveform, max_outputs=3, sample_rate=16000)
        # self.waveform = tf.placeholder(dtype=tf.float32, shape=[None, self.audio_length])
        # self.latent_vector = tf.placeholder(tf.float32, [None, self.latent_vec_size])
        # tf.summary.image('{}/Originals'.format(self.logsfolder), self.image, 10)

        self.input = self.preprocessing(self.waveform)
        # self.z_mu, self.z_logvar = self.encoder(self.waveform)
        # self.z = self.sample_z(self.z_mu, self.z_logvar)

        self.encoding = self.encoder(self.input)
        self.reconstructed_input = self.decoder(self.encoding)

        self.reconstructed_waveform = self.postprocessing(self.reconstructed_input)

        tf.summary.audio("rec_waveform", self.reconstructed_waveform, max_outputs=3, sample_rate=16000)

        # tf.summary.image('{}/Reconstructions'.format(self.logsfolder), self.reconstructions, 10)

        self.loss = self.compute_loss()
        tf.summary.scalar('Loss', self.loss)

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        gradients = optimizer.compute_gradients(loss=self.loss)

        # for gradient, variable in gradients:
        #     tf.summary.histogram("gradients/" + variable.name, gradient)
        #     tf.summary.histogram("variables/" + variable.name, variable)

        self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def preprocessing(self, waveform):
        stft = tf.contrib.signal.stft(waveform, self.frame_length, self.frame_step, self.fft_length)
        tf.summary.image("STFT", tf.expand_dims(tf.abs(stft), -1), 3)
        real = tf.real(stft)
        imag = tf.imag(stft)
        return tf.stack([real, imag], axis=-1)

    def postprocessing(self, reconstruction_2D):
        """
        first transform the 2D reconstruction to a complex signal
        """
        stft = tf.complex(reconstruction_2D[:, :, :, 0], reconstruction_2D[:, :, :, 1])
        tf.summary.image("Rec_STFT", tf.expand_dims(tf.abs(stft), -1), 3)

        inverse_stft = tf.contrib.signal.inverse_stft(
            stft, self.frame_length, self.frame_step, self.fft_length,
            window_fn=tf.contrib.signal.inverse_stft_window_fn(self.frame_step))
        return inverse_stft

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def encoder(self, x):
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)

        # x = tf.layers.flatten(x)
        # z_mu = tf.layers.dense(x, self.latent_vec_size, name='z_mu')
        # z_logvar = tf.layers.dense(x, self.latent_vec_size, name='z_logvar')
        # return z_mu, z_logvar
        return x

    def decoder(self, z, reuse=False):
        print(z.get_shape())
        x = tf.layers.conv2d_transpose(z, filters=32, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=3, activation=tf.nn.relu, padding='valid', strides=2)
        print(x.get_shape())
        x = tf.layers.conv2d_transpose(x, filters=2, kernel_size=(3, 5), activation=None, padding='valid', strides=2)
        print(x.get_shape())
        # assert(False)
        return x

    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.input)
        labels_flat = tf.layers.flatten(self.reconstructed_input)
        reconstruction_stft_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        """
        the istft is not perfect at the beginning and end of the signal
        """
        w_original = self.waveform[:, 100:self.audio_length-100]
        w_reconstr = self.reconstructed_waveform[:, 100:self.audio_length-100]
        reconstruction_wave_loss = tf.reduce_sum(tf.square(w_original - w_reconstr), axis=1)
        # kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_stft_loss + reconstruction_wave_loss)# + kl_loss)
        return vae_loss
