import tensorflow as tf
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

from network import Network
from read_tfrecords import input_fn

from signal_processing import wav_to_floats, floats_to_wav

BATCH_SIZE = 256

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
			while(1):
				_, loss_value, summary, rec = sess.run([network.train_op, network.loss, network.merged, network.reconstructed_waveform])
				writer.add_summary(summary, step)

				if np.isnan(loss_value):
					raise ValueError('Loss value is NaN')

				step+=1
				if step > 0 and step % 100 == 0:
					print("Step: {:05d} - Loss: {:.3f}".format(step, loss_value))
					save_path = saver.save(sess, model_name, global_step=step)

		except (KeyboardInterrupt, SystemExit):
			print("Manual Interrupt")

		except Exception as e:
			print("Exception: {}".format(e))
		finally:
			print("Model was saved here: {}".format(model_name))
			save_path = saver.save(sess, model_name, global_step=step)

def load_vae(model_path, generate=False):

	graph = tf.Graph()
	with graph.as_default():
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config, graph=graph)

		network = Network(generate=generate)
		init = tf.global_variables_initializer()
		sess.run(init)

		saver = tf.train.Saver(max_to_keep=1)

		try:
			saver.restore(sess, tf.train.latest_checkpoint(model_path))
		except:
			raise ImportError("Could not restore saved model")

		return sess, network

if __name__ == '__main__':
	train_vae()

	audio, fs = wav_to_floats('test.wav')
	audio = audio-np.mean(audio)
	audio /= np.max(np.abs(audio))
	audio = np.reshape(audio, (1, -1))
	model_path = "saved_models/"
	sess, network = load_vae(model_path)

	reconstructed_test = sess.run(network.reconstructed_waveform, feed_dict={network.waveform: audio})
	floats_to_wav("test2.wav", np.squeeze(reconstructed_test), 16000)

	z = np.random.randn(1, 64)
	del(sess)
	del(network)

	sess, network = load_vae(model_path, generate=True)
	generated = sess.run(network.generated_waveform, feed_dict={network.latent_vector: z})
	floats_to_wav("test3.wav", np.squeeze(generated), 16000)
