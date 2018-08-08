import tensorflow as tf
import glob, random
import numpy as np

_MAX_LENGTH = 16000


def _parse(record):
	features={
		'audio_raw': tf.FixedLenFeature([], tf.string)
	}
	parsed_record = tf.parse_single_example(record, features)
	audio = tf.decode_raw(parsed_record['audio_raw'], tf.float32)
	audio = tf.reshape(audio, [_MAX_LENGTH])
	return audio

def input_fn(batch_size, shuffle=True, source_directory='./data/train'):
	filenames = glob.glob('{}/*tfrecords'.format(source_directory))
	random.shuffle(filenames)
	dataset = (tf.data.TFRecordDataset(filenames)
		.map(_parse))
	if shuffle:
		dataset = dataset.shuffle(buffer_size=20000)

	dataset = dataset.repeat(None) # Infinite iterations: let experiment determine num_epochs
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_one_shot_iterator()
	audios = iterator.get_next()

	return audios

if __name__ == '__main__':
	audios = input_fn(128)

	init_op = tf.group(tf.global_variables_initializer(),
					   tf.local_variables_initializer())
	with tf.Session()  as sess:
		sess.run(init_op)
		for i in range(1000):
			audio_raw = sess.run(audios)
			print(audio_raw.shape)
