import functools
import tensorflow as tf
import tensorflow.contrib
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
frame_lengths = [45, 50, 70, 108, 136, 160]
frame_steps = [25, 30, 50, 64, 86, 123]
resamplings= [32, 16, 8, 4, 2, 1]
fft_lengths = [36, 62, 76, 120, 182, 256]

frame_lengths = [22, 50, 108, 160]
frame_steps = [15, 30, 61, 124]
resamplings= [64, 16, 4, 1]
fft_lengths = [31, 62, 127, 255]

params = list(zip(resamplings, fft_lengths, frame_lengths, frame_steps))

import sys
if __name__ == '__main__':
	show = False
	if len(sys.argv) > 1:
		show = True
	for resampling, fft_length, frame_length, frame_step in params:
		AUDIO_LENGTH = 16000

		x = tf.placeholder(tf.float32, shape=[AUDIO_LENGTH])

		resampled_x = x[::resampling]
		stft = tf.contrib.signal.stft(
			resampled_x,
			frame_length=frame_length,
			frame_step=frame_step,
			fft_length=fft_length
		)

		stft_shape = stft.shape

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

		with tf.Session() as sess:
			audio = np.random.randn(AUDIO_LENGTH)
			reconstructed, resampled_x_val = sess.run([istft, resampled_x], feed_dict={x: audio})

			plt.plot(resampled_x_val[:len(reconstructed)], label='orig')
			plt.plot(reconstructed, label='reconstructed')
			plt.legend()
			if show:
				plt.show()

			short_resampled_x = resampled_x_val[:len(reconstructed)]
			padding = int((frame_length - frame_step)/2)
			print("FFT LENGTH: {} - STFT shape: {} - Error: {:.6f}".format(fft_length,
												stft_shape,
												np.mean(np.abs(short_resampled_x[padding: -padding] - reconstructed[padding: -padding]))))
