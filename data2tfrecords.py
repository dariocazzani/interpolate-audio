#! /usr/env/bin python3
"""Convert Audio Dataset to local TFRecords"""

import argparse
import os, sys
import glob
import tqdm
import numpy as np
import tensorflow as tf
import random
import multiprocessing as mp

from signal_processing import wav_to_floats

parser = argparse.ArgumentParser()

parser.add_argument(
    '--destination-directory', required=True,
    help='Directory where TFRecords will be stored')

parser.add_argument(
    '--source-directory', required=True,
    help='Directory where original training WAV files are')

parser.add_argument(
    '--num-samples-per-file', default=10000, type=int,
    help='Number of wave files to put in a single TFRecord file')

parser.add_argument(
    '--max_length', default=16000, type=int,
    help='Maximum length - in samples - for each wave file')

args = parser.parse_args()


def pad_audio(audio:np.array, length:int) ->np.array:
	if len(audio) < length:
		audio = np.lib.pad(audio, (0, length - len(audio)), 'constant', constant_values=(0))
	return audio

def _preprocessing(file:str) -> np.array:
    data, _ = wav_to_floats(file)
    data = pad_audio(data, args.max_length)
    return data.astype('float32')

def _data_path(data_directory:str, name:str) -> str:

    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, '{}.tfrecords'.format(name))

def _bytes_feature(value:str) -> tf.train.Features.FeatureEntry:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

    audio_files = glob.glob(args.source_directory+'/**/*wav')
    for _ in range(7):
        random.shuffle(audio_files)

    # Each TFRecord file constains at most args.num_samples_per_file samples
    train_audio_files_list = [audio_files[i:i+args.num_samples_per_file] for i in range(0, len(audio_files), args.num_samples_per_file)]

    def convert(data:list):
        audio_files_list, name, number = data
        filename = _data_path(os.path.join(args.destination_directory, name), '{:03d}'.format(number))

        with tf.python_io.TFRecordWriter(filename) as writer:
            for chosen_one in tqdm.tqdm(audio_files_list):
                audio = _preprocessing(chosen_one)

                if (len(audio) > args.max_length):
                    print("skipping: {} - length {}".format(chosen_one, len(audio)))
                    continue

                audio_raw = audio.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'audio_raw': _bytes_feature(audio_raw)
                }))
                writer.write(example.SerializeToString())

    names = ['train']*len(train_audio_files_list)
    numbers = list(range(len(train_audio_files_list)))
    with mp.Pool(mp.cpu_count()) as p:
        list(tqdm.tqdm(p.imap(convert, list(zip(train_audio_files_list, names, numbers))), total=len(train_audio_files_list)))
