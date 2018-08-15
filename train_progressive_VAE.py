import tensorflow as tf
import functools
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from network import Network
from read_tfrecords import input_fn

from signal_processing import wav_to_floats, floats_to_wav

BATCH_SIZE = 256

def preprocessing(waveform, resampling, fft_length, frame_step, frame_length):

	resampled_wave = waveform[:, ::resampling]
	stft = tf.contrib.signal.stft(
		resampled_wave,
		frame_length=frame_length,
		frame_step=frame_step,
		fft_length=fft_length
	)
	tf.summary.image("STFT", tf.expand_dims(tf.abs(stft), -1), 3)
	real = tf.real(stft)
	imag = tf.imag(stft)
	return tf.stack([real, imag], axis=-1), resampled_wave

def postprocessing(stft, fft_length, frame_step, frame_length):
	stft = tf.complex(stft[:, :, :, 0], stft[:, :, :, 1])

	istft = tf.contrib.signal.inverse_stft(
		stft,
		frame_length=frame_length,
		frame_step=frame_step,
		fft_length=fft_length,
		# forward_window_fn
		window_fn=tf.contrib.signal.inverse_stft_window_fn(
			frame_step,
			functools.partial(tf.contrib.signal.hann_window, periodic=True),
		)
	)
	return istft

def sample_z(mu, logvar):
	eps = tf.random_normal(shape=tf.shape(mu))
	return mu + tf.exp(logvar / 2) * eps
"""
GRAPH
"""
AUDIO_LENGTH = 16000
LATENT_VEC_SIZE = 64
LEARNING_RATE = 1E-3

# variables:
frame_lengths = [22, 50, 108, 160]
frame_steps = [15, 30, 61, 124]
resamplings= [64, 16, 4, 1]
fft_lengths = [31, 62, 127, 255]

step_counter = tf.placeholder(tf.int32, name='step_counter')
global_step = tf.Variable(0, name='global_step', trainable=False)
waveform = tf.placeholder(dtype=tf.float32, shape=[None, AUDIO_LENGTH])
stft_64, resampled_wave_64 = preprocessing(waveform, resamplings[0], fft_lengths[0], frame_steps[0], frame_lengths[0])
stft_16, resampled_wave_16 = preprocessing(waveform, resamplings[1], fft_lengths[1], frame_steps[1], frame_lengths[1])
stft_4, resampled_wave_4 = preprocessing(waveform, resamplings[2], fft_lengths[2], frame_steps[2], frame_lengths[2])
stft_1, resampled_wave_1 = preprocessing(waveform, resamplings[3], fft_lengths[3], frame_steps[3], frame_lengths[3])

### ENCODER ###

# 1 #
print(stft_1.get_shape())
x = tf.layers.conv2d(stft_1, filters=16, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_1_conv1")
x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, strides=2, padding='same', reuse=tf.AUTO_REUSE, name="enc_1_conv2")
x = tf.layers.conv2d(x, filters=2, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_1_conv3")
print(x.get_shape())

# 4 #
pred = tf.logical_and(tf.less(step_counter, 30), tf.less(19, step_counter))
x = tf.cond(pred, lambda: stft_4, lambda: x)
# Because ops that could lead to different shapes might be executed,
# we need to enforce a shape
shape = stft_4.get_shape()
x.set_shape([None, int(shape[1]), int(shape[2]), int(shape[3])])
print(x.get_shape())

x = tf.layers.conv2d(x, filters=16, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_4_conv1")
x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, strides=2, padding='same', reuse=tf.AUTO_REUSE, name="enc_4_conv2")
x = tf.layers.conv2d(x, filters=2, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_4_conv3")
print(x.get_shape())

# 16 #
pred = tf.logical_and(tf.less(step_counter, 20), tf.less(9, step_counter))
x = tf.cond(pred, lambda: stft_16, lambda: x)
# Because ops that could lead to different shapes might be executed,
# we need to enforce a shape
shape = stft_16.get_shape()
x.set_shape([None, int(shape[1]), int(shape[2]), int(shape[3])])
print(x.get_shape())

x = tf.layers.conv2d(x, filters=16, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_16_conv1")
x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, strides=2, padding='same', reuse=tf.AUTO_REUSE, name="enc_16_conv2")
x = tf.layers.conv2d(x, filters=2, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_16_conv3")
print(x.get_shape())

# 64 #
x = tf.cond(tf.less(step_counter, 10), lambda: stft_64, lambda: x)
# Because ops that could lead to different shapes might be executed,
# we need to enforce a shape
shape = stft_64.get_shape()
x.set_shape([None, int(shape[1]), int(shape[2]), int(shape[3])])
print(x.get_shape())
x = tf.layers.conv2d(x, filters=16, kernel_size=3, activation=tf.nn.relu, strides=1, padding='same', reuse=tf.AUTO_REUSE, name="enc_64_conv1")
x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, strides=2, padding='valid', reuse=tf.AUTO_REUSE, name="enc_64_conv2")
x = tf.layers.flatten(x)
x = tf.layers.dense(x, units=512, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="enc_64_dense1")
print(x.get_shape())

# mu and var #
z_mu = tf.layers.dense(x, units=LATENT_VEC_SIZE, name='z_mu', reuse=tf.AUTO_REUSE)
z_logvar = tf.layers.dense(x, units=LATENT_VEC_SIZE, name='z_logvar', reuse=tf.AUTO_REUSE)

print(z_mu.get_shape())

# Sample z
z = sample_z(z_mu, z_logvar)

# DECODER #
dec1 = tf.layers.dense(z, units=512, activation=tf.nn.relu, name='dec_dense1', reuse=tf.AUTO_REUSE)
reconstructed_stft = tf.reshape(dec1, [-1, 16, 16, 2])

reconstructed_wave = postprocessing(reconstructed_stft, fft_length, frame_step, frame_length)

padding = int((frame_length - frame_step)/2)

# Loss
logits_flat = tf.layers.flatten(reconstructed_stft)
labels_flat = tf.layers.flatten(stft)
reconstruction_stft_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
"""
the istft is not perfect at the beginning and end of the signal
"""

short_resampled_wave = resampled_wave[:, :247]

w_original = short_resampled_wave[:, padding: -padding]
w_reconstr = reconstructed_wave[:, 	padding: -padding]

reconstruction_wave_loss = tf.reduce_sum(tf.square(w_original - w_reconstr), axis=1)
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
vae_loss = tf.reduce_mean(reconstruction_stft_loss + reconstruction_wave_loss + kl_loss)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
gradients = optimizer.compute_gradients(loss=vae_loss)

capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

merged = tf.summary.merge_all()

def train_vae():
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('logdir')

		model_path = "saved_models/"
		model_name = model_path + 'model'

		audios = input_fn(BATCH_SIZE, source_directory='data/train/')

		init_op = tf.group(tf.global_variables_initializer(),
						   tf.local_variables_initializer())

		tf.global_variables_initializer().run()

		saver = tf.train.Saver(max_to_keep=1)

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
			print("Model restored from: {}".format(model_path))
		except:
			print("Could not restore saved model")

		step = global_step.eval()

		try:
			while(1):
				a = sess.run(audios)
				_, loss_value, summary, rec = sess.run([train_op, vae_loss, merged, reconstructed_wave],
									feed_dict={waveform: a})
				writer.add_summary(summary, step)

				if np.isnan(loss_value):
					raise ValueError('Loss value is NaN')

				step+=1
				if step > 0 and step % 100 == 0:
					print("Step: {:05d} - Loss: {:.3f}".format(step, loss_value))
					save_path = saver.save(sess, model_name, global_step=step)

				if step == 10:
					enc_flat = tf.layers.dense(enc_flat, units=64, name='enc_flat', reuse=tf.AUTO_REUSE)


		except (KeyboardInterrupt, SystemExit):
			print("Manual Interrupt")

		except Exception as e:
			print("Exception: {}".format(e))
		finally:
			print("Model was saved here: {}".format(model_name))
			save_path = saver.save(sess, model_name, global_step=step)

if __name__ == '__main__':
	train_vae()
