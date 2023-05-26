import os
import argparse

import librosa
import numpy as np
import soundfile as sf

import util

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default='./data')
parser.add_argument('--train_wav_dir', default='train/training_voice_data')
args = parser.parse_args()


pid_all, _, ans_all = util.read_csv()
ORI_DATA_DIR = os.path.join(args.data_dir, args.train_wav_dir)

# for noise augmentation
NOISE_RATIOS = [4, 3]
# for voice shift augmentation
SHIFT_NUMS = [-4, -3, -2, 2, 3, 4]

pid_all = set([pid for pid, ans in zip(pid_all, ans_all) if ans in (3, 4)])
in_wavs = os.listdir(ORI_DATA_DIR)
for wav in in_wavs:
    pid = wav.split('.')[0]
    if pid not in pid_all:
        continue
    wav_path = os.path.join(ORI_DATA_DIR, wav)
    y, sr = librosa.load(wav_path, sr=None)

    for noise in NOISE_RATIOS:
        print(wav, noise)
        out_dir = ORI_DATA_DIR + '_noise_{}'.format(noise)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, 0o755)
        y_new = y + np.random.randn(y.shape[0]) * np.max(y) / noise
        sf.write(os.path.join(out_dir, wav), y_new, sr, subtype='PCM_16')
    
    for shift in SHIFT_NUMS:
        print(wav, shift)
        out_dir = ORI_DATA_DIR + '_shift_{}'.format(shift)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, 0o755)
        y_new = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
        sf.write(os.path.join(out_dir, wav), y_new, sr, subtype='PCM_16')

