import wave
import struct
from scipy.io.wavfile import write
import numpy as np
from scipy import signal

def floats_to_wav(filename, audio, fs):
	scaled = np.int16(audio/np.max(np.abs(audio)) * np.iinfo(np.int16).max)
	write(filename, fs, scaled)

def wav_to_floats(wave_file):
	w = wave.open(wave_file, 'rb')
	astr = w.readframes(w.getnframes())
	a = struct.unpack("%ih" % (len(astr) / 2), astr)
	a = np.asarray(a, dtype=np.float32) / pow(2, 15)
	w.close()
	return a, w.getframerate()
