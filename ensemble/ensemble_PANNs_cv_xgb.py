from copy import deepcopy
import argparse

import os
import sys
import numpy as np
import librosa
import torch
from xgboost import XGBClassifier

from models import *
import util


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default='../data')
parser.add_argument('--feat_dir', '-fd', default='../feature')
parser.add_argument('--train_feat', default='train/mel_8192_4096_128_cut.npy')
parser.add_argument('--train_data', default='train/training datalist_SORTED.csv')
parser.add_argument('--train_wav_dir', default='train/training_voice_data')
parser.add_argument('--test', '-t', action='store_true', default=False)
parser.add_argument('--public_feat', default='public/mel_8192_4096_128_cut.npy')
parser.add_argument('--public_data', default='public/test_datalist_public.csv')
parser.add_argument('--public_wav_dir', default='public/test_data_public')
parser.add_argument('--private_feat', default='private/mel_8192_4096_128_cut.npy')
parser.add_argument('--private_data', default='private/test_datalist_private.csv')
parser.add_argument('--private_wav_dir', default='private/test_data_private')
parser.add_argument('--write_train_result', '-w', action='store_true', default=False)
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=1024)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--mel_bins', type=int, default=64)
parser.add_argument('--fmin', type=int, default=50)
parser.add_argument('--fmax', type=int, default=14000) 
parser.add_argument('--model_type', type=str, default='Cnn6')
parser.add_argument('--checkpoint_path', type=str, default='./model/Cnn6_mAP=0.343.pth')
parser.add_argument('--cnn_train_feat', default='train/cnn_1024_320_64.npy')
parser.add_argument('--cnn_public_feat', default='public/cnn_1024_320_64.npy')
parser.add_argument('--cnn_private_feat', default='private/cnn_1024_320_64.npy')

args = parser.parse_args()

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

classes_num = 527
labels = ['Phonotrauma','Incomplete glottic closure','Vocal palsy','Neoplasm','Normal']

device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')

Model = eval(args.model_type)
model = Model(sample_rate=args.sample_rate, window_size=args.window_size, hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax, classes_num=classes_num)

checkpoint = torch.load(args.checkpoint_path, map_location=device)
model_dict = model.state_dict()
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Parallel
print('GPU number: {}'.format(torch.cuda.device_count()))
model = torch.nn.DataParallel(model)

# pid_all, fea_all, ans_all = util.read_csv('../data/test/test_datalist_private.csv', mode='test')
if not os.path.exists(args.cnn_train_feat):

    pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.train_data))
    audio_fea_all = []
    for idx, pid in enumerate(pid_all):
        if idx % 50 == 0:
            print('Processing {}/{}'.format(idx, len(pid_all)))
        audio_path = os.path.join(os.path.join(args.data_dir, args.train_wav_dir), pid+'.wav')
        
        # Load audio
        (waveform, _) = librosa.load(audio_path, sr=args.sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)

            # Print embedding
            if 'embedding' in batch_output_dict.keys():
                embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
                # print('embedding: {}'.format(embedding.shape))
                audio_fea_all.append(embedding)

    np.save(os.path.join(args.feat_dir,args.cnn_train_feat), audio_fea_all)

if args.test and (not os.path.exists(args.cnn_public_feat) or not os.path.exists(args.cnn_private_feat)):
    
    pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.public_data), mode='test')
    audio_fea_all = []
    for idx, pid in enumerate(pid_all):
        if idx % 50 == 0:
            print('Processing {}/{}'.format(idx, len(pid_all)))
        audio_path = os.path.join(os.path.join(args.data_dir, args.public_wav_dir), pid+'.wav')
        
        # Load audio
        (waveform, _) = librosa.load(audio_path, sr=args.sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)

            # Print embedding
            if 'embedding' in batch_output_dict.keys():
                embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
                # print('embedding: {}'.format(embedding.shape))
                audio_fea_all.append(embedding)

    np.save(os.path.join(args.feat_dir,args.cnn_public_feat), audio_fea_all)

    pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.private_data), mode='test')
    audio_fea_all = []
    for idx, pid in enumerate(pid_all):
        if idx % 50 == 0:
            print('Processing {}/{}'.format(idx, len(pid_all)))
        audio_path = os.path.join(os.path.join(args.data_dir, args.private_wav_dir), pid+'.wav')
        
        # Load audio
        (waveform, _) = librosa.load(audio_path, sr=args.sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, device)

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)

            # Print embedding
            if 'embedding' in batch_output_dict.keys():
                embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
                # print('embedding: {}'.format(embedding.shape))
                audio_fea_all.append(embedding)

    np.save(os.path.join(args.feat_dir,args.cnn_private_feat), audio_fea_all)

N_FOLD = 7
param = {
    'n_estimators': 210,
    'max_depth': 5,
    'learning_rate': 0.055,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'colsample_bytree': 1.0,
    # 'use_label_encoder': False,
}
param_cnn = {
    'n_estimators': 310,
    'max_depth': 3,
    'learning_rate': 0.055,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'colsample_bytree': 1.0,
    # 'use_label_encoder': False,
}
Classifier = XGBClassifier

#for cnn
pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.train_data))
audio_fea_all = np.load(os.path.join(args.feat_dir,args.cnn_train_feat))
audio_fea_all = np.stack(audio_fea_all)
audio_fea_all = np.expand_dims(audio_fea_all, axis=1)
print(audio_fea_all.shape)

index_all = np.arange(fea_all.shape[0])
print('Data shapes:', len(pid_all), fea_all.shape, ans_all.shape, index_all.shape)

# audio_fea_all = np.squeeze(audio_fea_all, axis=1)
audio_fea_all = np.median(audio_fea_all, axis=2)
# audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
print('Audio fea shape:', audio_fea_all.shape)

fea_all = np.hstack([fea_all, audio_fea_all])
print('Combined fea shape:', fea_all.shape)

pred_all = []
ans_for_eval_all = []
raw_idx_for_eval_all = []
best_n_est_all = []
models = []
for fold in range(N_FOLD):
    train_idx = np.where(index_all % N_FOLD != fold)[0]
    test_idx = np.where(index_all % N_FOLD == fold)[0]
    print('Fold {}, tr num {}, te num {}'.format(fold, train_idx.size, test_idx.size))

    # 1st pass
    model = Classifier(**param_cnn)
    model.fit(
        fea_all[train_idx],
        ans_all[train_idx],
        sample_weight=[0.05 if ans <= 2 else 1.0 for ans in ans_all[train_idx]],
        # eval_set=[(fea_all[test_idx], ans_all[test_idx])],
    )

    pred = model.predict_proba(fea_all[test_idx])
    pred_all.append(pred)
    ans_for_eval_all.append(ans_all[test_idx])
    raw_idx_for_eval_all.append(index_all[test_idx])
    models += [model]

print('Best n_ests:', best_n_est_all)

pred_all = np.vstack(pred_all)
ans_for_eval_all = np.hstack(ans_for_eval_all)
print('Finished, shapes:', pred_all.shape, ans_for_eval_all.shape)

conf_mat, uar = util.get_conf_mat_and_uar(np.argmax(pred_all, axis=1), ans_for_eval_all)
print(conf_mat)
print('UAR: {:.4f}%'.format(100*uar))

# for xgb
FEAT_PATH = os.path.join(args.feat_dir,args.train_feat)

pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.train_data))
index_all = np.arange(fea_all.shape[0])
print('Data shapes:', len(pid_all), fea_all.shape, ans_all.shape, index_all.shape)

audio_fea_all = np.load(FEAT_PATH)
audio_fea_all = np.median(audio_fea_all, axis=2)
# audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
print('Audio fea shape:', audio_fea_all.shape)

fea_all = np.hstack([fea_all, audio_fea_all])
print('Combined fea shape:', fea_all.shape)

pred_all = []
ans_for_eval_all = []
raw_idx_for_eval_all = []
best_n_est_all = []
model = Classifier(**param)
for fold in range(N_FOLD):
    train_idx = np.where(index_all % N_FOLD != fold)[0]
    test_idx = np.where(index_all % N_FOLD == fold)[0]
    print('Fold {}, tr num {}, te num {}'.format(fold, train_idx.size, test_idx.size))

    # 1st pass
    model.fit(
        fea_all[train_idx],
        ans_all[train_idx],
        sample_weight=[0.05 if ans <= 2 else 1.0 for ans in ans_all[train_idx]],
        # eval_set=[(fea_all[test_idx], ans_all[test_idx])],
    )

    pred = model.predict_proba(fea_all[test_idx])
    pred_all.append(pred)
    ans_for_eval_all.append(ans_all[test_idx])
    raw_idx_for_eval_all.append(index_all[test_idx])
    # models += [deepcopy(model)]

print('Best n_ests:', best_n_est_all)

pred_all = np.vstack(pred_all)
ans_for_eval_all = np.hstack(ans_for_eval_all)
print('Finished, shapes:', pred_all.shape, ans_for_eval_all.shape)

conf_mat, uar = util.get_conf_mat_and_uar(np.argmax(pred_all, axis=1), ans_for_eval_all)
print(conf_mat)
print('UAR: {:.4f}%'.format(100*uar))

if args.test:
    print('public testing...')

    FEAT_PATH = os.path.join(args.feat_dir,args.cnn_public_feat)

    pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.public_data), mode='test')
    index_all = np.arange(fea_all.shape[0])
    print('Data shapes:', len(pid_all), fea_all.shape, ans_all.shape, index_all.shape)

    audio_fea_all = np.load(FEAT_PATH)
    audio_fea_all = np.expand_dims(audio_fea_all, axis=1)
    audio_fea_all = np.median(audio_fea_all, axis=2)
    # audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
    print('Audio fea shape:', audio_fea_all.shape)

    cnn_fea_all = np.hstack([fea_all, audio_fea_all])
    print('Combined fea shape:', cnn_fea_all.shape)

    # pred_all = np.zeros((500, 5))
    pred_all = []
    for cnn_model in models:
        pred = cnn_model.predict_proba(cnn_fea_all)
        print(pred)
        pred_all += [pred]
        # print(pred_all.shape)
    pred_all = np.array(pred_all)
    print(pred_all.shape)
    res = np.max(pred_all, axis=0)
    # print(res.shape)
    # res = model.predict_proba(cnn_fea_all)
    # cnn_res = np.argmax(pred_all, axis=1) + 1
    cnn_res = res
    print(cnn_res.shape)
    # print(cnn_res)
    
    FEAT_PATH = os.path.join(args.feat_dir,args.public_feat)
    audio_fea_all = np.load(FEAT_PATH)
    audio_fea_all = np.median(audio_fea_all, axis=2)
    # audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
    print('Audio fea shape:', audio_fea_all.shape)

    fea_all = np.hstack([fea_all, audio_fea_all])
    print('Combined fea shape:', fea_all.shape)

    res = model.predict_proba(fea_all)
    # res = np.argmax(res, axis=1) + 1  
    print(res.shape)
    # print(res)
    
    combined_res = np.array([0.5*le+ri if np.argmax(le, axis=0) == 3 else ri for le, ri in zip(cnn_res, res)])
    # combined_res = np.array([le if np.argmax(le, axis=0) == 3 else ri for le, ri in zip(cnn_res, res)])
    print(combined_res.shape)
    # combined_res = np.max(combined_res, axis=0)
    combined_res_public = np.argmax(combined_res, axis=1) + 1

    print('private testing...')

    FEAT_PATH = os.path.join(args.feat_dir,args.cnn_private_feat)

    pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir, args.private_data), mode='test')
    index_all = np.arange(fea_all.shape[0])
    print('Data shapes:', len(pid_all), fea_all.shape, ans_all.shape, index_all.shape)

    audio_fea_all = np.load(FEAT_PATH)
    audio_fea_all = np.expand_dims(audio_fea_all, axis=1)
    audio_fea_all = np.median(audio_fea_all, axis=2)
    # audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
    print('Audio fea shape:', audio_fea_all.shape)

    cnn_fea_all = np.hstack([fea_all, audio_fea_all])
    print('Combined fea shape:', cnn_fea_all.shape)

    # pred_all = np.zeros((500, 5))
    pred_all = []
    for cnn_model in models:
        pred = cnn_model.predict_proba(cnn_fea_all)
        print(pred)
        pred_all += [pred]
        # print(pred_all.shape)
    pred_all = np.array(pred_all)
    print(pred_all.shape)
    res = np.max(pred_all, axis=0)
    # print(res.shape)
    # res = model.predict_proba(cnn_fea_all)
    # cnn_res = np.argmax(pred_all, axis=1) + 1
    cnn_res = res
    print(cnn_res.shape)
    # print(cnn_res)
    
    FEAT_PATH = os.path.join(args.feat_dir,args.private_feat)
    audio_fea_all = np.load(FEAT_PATH)
    audio_fea_all = np.median(audio_fea_all, axis=2)
    # audio_fea_all = np.hstack([np.mean(audio_fea_all, axis=2), np.median(audio_fea_all, axis=2)])
    print('Audio fea shape:', audio_fea_all.shape)

    fea_all = np.hstack([fea_all, audio_fea_all])
    print('Combined fea shape:', fea_all.shape)

    res = model.predict_proba(fea_all)
    # res = np.argmax(res, axis=1) + 1  
    print(res.shape)
    # print(res)
    
    combined_res = np.array([0.5*le+ri if np.argmax(le, axis=0) == 3 else ri for le, ri in zip(cnn_res, res)])
    # combined_res = np.array([le+ri if np.argmax(le, axis=0) == 3 else ri for le, ri in zip(cnn_res, res)])
    print(combined_res.shape)
    # combined_res = np.max(combined_res, axis=0)
    combined_res_private = np.argmax(combined_res, axis=1) + 1


    with open('./results/submission.csv', 'w') as fp:
        for id, cat in zip(pid_all, combined_res_public):
            fp.write('{},{}\n'.format(id,cat))
            
        for id, cat in zip(pid_all, combined_res_private):
            fp.write('{},{}\n'.format(id,cat))