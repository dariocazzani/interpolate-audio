import tensorflow as tf
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from network import Network
from read_tfrecords import input_fn

from signal_processing import wav_to_floats, floats_to_wav

audio, fs = wav_to_floats('test.wav')
audio = audio-np.mean(audio)
audio /= np.max(np.abs(audio))
audio = np.reshape(audio, (1, -1))
TOT_STEPS = 5000
BATCH_SIZE = 128

def train_vae():
	with tf.Session() as sess:
		writer = tf.summary.FileWriter('logdir')

		model_path = "saved_models/"
		model_name = model_path + 'model'

		audios = input_fn(BATCH_SIZE, source_directory='data/train/')

		init_op = tf.group(tf.global_variables_initializer(),
						   tf.local_variables_initializer())

		network = Network(audios)
		tf.global_variables_initializer().run()

		saver = tf.train.Saver(max_to_keep=1)

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
			print("Model restored from: {}".format(model_path))
		except:
			print("Could not restore saved model")

		step = network.global_step.eval()

		try:
			while(step <= TOT_STEPS):
				feed_dict = {network.waveform: audio}
				_, loss_value, summary, rec = sess.run([network.train_op, network.loss, network.merged, network.reconstructed_waveform],
									feed_dict=feed_dict)
				writer.add_summary(summary, step)

				if np.isnan(loss_value):
					raise ValueError('Loss value is NaN')

				step+=1
				if step > 0 and step % 100 == 0:
					print("Step: {:05d} - Loss: {:.3f}".format(step, loss_value))

		except (KeyboardInterrupt, SystemExit):
			print("Manual Interrupt")

		except Exception as e:
			print("Exception: {}".format(e))
		finally:
			print("Model was saved here: {}".format(model_name))

if __name__ == '__main__':
	train_vae()
