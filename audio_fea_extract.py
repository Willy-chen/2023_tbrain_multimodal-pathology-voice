import os
import argparse

import librosa
import numpy as np

import util

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--feat_dir', default='./feature')
parser.add_argument('--feat_type', default='mel')
parser.add_argument('--train_data', default='train/training datalist_SORTED.csv')
parser.add_argument('--train_wav_dir', default='train/training_voice_data')
parser.add_argument('--public_data', default='public/test_datalist_public.csv')
parser.add_argument('--public_wav_dir', default='public/test_data_public')
parser.add_argument('--private_data', default='private/test_datalist_private.csv')
parser.add_argument('--private_wav_dir', default='public/test_data_private')
parser.add_argument('--frame_size', default=8192)
parser.add_argument('--hop_size', default=4096)
parser.add_argument('--n_mel_bin', default=128)
parser.add_argument('--n_mfcc', default=12)
parser.add_argument('--n_chroma', default=12)
parser.add_argument('--align_op', default='cut')
args = parser.parse_args()


if args.mode == 'train':
    CSV_PATH = os.path.join(args.data_dir, args.train_data)
elif args.mode == 'public':
    CSV_PATH = os.path.join(args.data_dir, args.public_data)
elif args.mode == 'private':
    CSV_PATH = os.path.join(args.data_dir, args.private_data)

FEA_TYPE = args.feat_type
FRAME_SIZE = args.frame_size
HOP_SIZE = args.hop_size

# for mel spectrogram
N_MEL_BIN = args.n_mel_bin
# for mfcc
N_MFCC = args.n_mfcc
# for chroma_stft
N_CHROMA = args.n_chroma

ALIGN_OP = args.align_op

output_to = args.feat_dir + '/{}/{}_{}_{}'.format(args.mode,FEA_TYPE, FRAME_SIZE, HOP_SIZE)
if FEA_TYPE == 'mel':
    output_to = output_to + '_{}'.format(N_MEL_BIN)
elif FEA_TYPE in ('mfcc', 'mfcc_delta'):
    output_to = output_to + '_{}'.format(N_MFCC)
elif FEA_TYPE in ('chroma_stft', 'chroma_cqt', 'chroma_cens', 'chroma_vqt'):
    output_to = output_to + '_{}'.format(N_CHROMA)
elif FEA_TYPE == 'wav':
    output_to = output_to.split('_')[0]

output_to = output_to + '_' + ALIGN_OP

if args.mode == 'train':
    pid_all, _, ans_all = util.read_csv(CSV_PATH, mode='train')
else:
    pid_all, _, ans_all = util.read_csv(CSV_PATH, mode='test')
print('Data shapes:', len(pid_all), ans_all.shape)

fea_all = []
frame_num_all = []
for idx, pid in enumerate(pid_all):
    if idx % 50 == 0:
        print('Processing {}/{}'.format(idx, len(pid_all)))
    
    if args.mode == 'train':
        wav_path = os.path.join(os.path.join(args.data_dir, args.train_wav_dir), pid+'.wav')
    elif args.mode == 'public':
        wav_path = os.path.join(os.path.join(args.data_dir, args.public_wav_dir), pid+'.wav')
    elif args.mode == 'private':
        wav_path = os.path.join(os.path.join(args.data_dir, args.private_wav_dir), pid+'.wav')

    y, sr = librosa.load(wav_path, sr=None)
    if FEA_TYPE == 'mel':
        fea_mat = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MEL_BIN)
    elif FEA_TYPE == 'mfcc':
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MEL_BIN)
        log_s = librosa.amplitude_to_db(s, ref=np.max)
        fea_mat = librosa.feature.mfcc(S=log_s, n_mfcc=N_MFCC)
    elif FEA_TYPE == 'mfcc_delta':
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=N_MEL_BIN)
        log_s = librosa.amplitude_to_db(s, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_s, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        fea_mat = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    elif FEA_TYPE == 'stft':
        fea_mat = librosa.amplitude_to_db(np.abs(librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)))
    elif FEA_TYPE == 'chroma_stft':
        fea_mat = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_chroma=N_CHROMA)
    elif FEA_TYPE == 'chroma_cqt':
        fea_mat = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_SIZE, n_chroma=N_CHROMA)
    elif FEA_TYPE == 'chroma_cens':
        fea_mat = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_SIZE, n_chroma=N_CHROMA)
    elif FEA_TYPE == 'chroma_vqt':
        fea_mat = librosa.feature.chroma_vqt(y=y, sr=sr, intervals='ji5',hop_length=HOP_SIZE, bins_per_octave=N_CHROMA)
    elif FEA_TYPE == 'rms':
        fea_mat = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)
    elif FEA_TYPE == 'spectral_centroid':
        fea_mat = librosa.feature.spectral_centroid(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    elif FEA_TYPE == 'spectral_bandwidth':
        fea_mat = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    elif FEA_TYPE == 'spectral_contrast':
        S = np.abs(librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        fea_mat = librosa.feature.spectral_contrast(S=S, sr=sr)
    elif FEA_TYPE == 'spectral_flatness':
        fea_mat = librosa.feature.spectral_flatness(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    elif FEA_TYPE == 'spectral_rolloff':
        fea_mat = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    elif FEA_TYPE == 'poly_features':
        S = np.abs(librosa.stft(y=y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE))
        fea_mat = librosa.feature.poly_features(S=S, order=0)
    elif FEA_TYPE == 'tonnetz':
        y = librosa.effects.harmonic(y=y)
        fea_mat = librosa.feature.tonnetz(y=y, sr=sr)
    elif FEA_TYPE == 'zero_crossing_rate':
        fea_mat = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE);
    elif FEA_TYPE == 'wav':
        fea_mat = y[np.newaxis, :]
    else:
        raise Exception('Unsupported FEA_TYPE: {}'.format(FEA_TYPE))
    fea_all.append(fea_mat) # Shape of fea_mat: (freq, time)
    frame_num_all.append(fea_mat.shape[1])

frame_num_min = min(frame_num_all)
frame_num_max = max(frame_num_all)
if ALIGN_OP == 'cut':
    fea_all = [fea[:, :frame_num_min] for fea in fea_all]
elif ALIGN_OP == 'pad':
    fea_all = [np.pad(fea, ((0, 0),(0, frame_num_max-fea.shape[1]))) for fea in fea_all]
elif ALIGN_OP == 'pre-avg':
    fea_all = [np.mean(fea, axis=1) for fea in fea_all]
elif ALIGN_OP == 'min-stack':
    stacked_fea_all = []
    stacked_pid_all = []
    for fea, pid in zip(fea_all, pid_all):
        for begin in range(0, fea.shape[1], frame_num_min):
            end = begin + frame_num_min
            if end > fea.shape[1]:
                break
            stacked_fea_all.append(fea[:, begin:end])
            stacked_pid_all.append(pid)
    fea_all = stacked_fea_all
    print('Stacked pid len:', len(stacked_pid_all))
    with open(output_to + '.txt', 'w') as fout:
        fout.write('\n'.join(stacked_pid_all))

fea_all = np.stack(fea_all)
print('fea shapes:', fea_all.shape)
np.save(output_to + '.npy', fea_all)
