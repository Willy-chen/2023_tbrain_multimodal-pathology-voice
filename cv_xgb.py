import os
import argparse
from copy import deepcopy

import numpy as np
from xgboost import XGBClassifier

import util

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default='./data')
parser.add_argument('--feat_dir', default='./feature')
parser.add_argument('--train_feat', default='train/mel_8192_4096_128_cut.npy')
parser.add_argument('--train_data', default='train/training datalist_SORTED.csv')
parser.add_argument('--test', '-t', action='store_true', default=False)
parser.add_argument('--public_feat', default='public/mel_8192_4096_128_cut.npy')
parser.add_argument('--public_data', default='public/test_datalist_public.csv')
parser.add_argument('--private_feat', default='private/mel_8192_4096_128_cut.npy')
parser.add_argument('--private_data', default='private/test_datalist_private.csv')
parser.add_argument('--write_train_result', '-w', action='store_true', default=False)
args = parser.parse_args()

# for data augmentation
shifts = [-4, -3, -2, -1, 1, 2, 3, 4]
noises = [4, ]

# for XGBoost
N_FOLD = 7
param = {
    'n_estimators': 210,
    'max_depth': 7,
    'learning_rate': 0.055,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'colsample_bytree': 1.0,
    'use_label_encoder': False,
}
Classifier = XGBClassifier
ans_to_weight = [0.05, 0.1, 0.1, 1, 1]

# Data preparation
FEAT_PATH = os.path.join(args.feat_dir,args.train_feat)

pid_all, fea_all, ans_all = util.read_csv(os.path.join(args.data_dir,args.train_data), mode='train')
index_all = np.arange(fea_all.shape[0])
pid_to_idx = {pid: idx for pid, idx in zip(pid_all, index_all)}
print('Data shapes:', len(pid_all), fea_all.shape, ans_all.shape, index_all.shape)

audio_fea_all = np.load(FEAT_PATH)
audio_fea_all = np.median(audio_fea_all, axis=2)
print('Audio fea shape:', audio_fea_all.shape)

## for Data augmentation
aug_fea_all = []
aug_idx_all = []
# for shift in (-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4):
for shift in shifts:
    shifted_fea_path = FEAT_PATH[:-4] + '_shift_{}.npy'.format(shift)
    shifted_pid_path = FEAT_PATH[:-4] + '_shift_{}.txt'.format(shift)
    shifted_fea = np.load(shifted_fea_path)
    with open(shifted_pid_path, 'r') as fin:
        shifted_pid = fin.read().splitlines()
    aug_fea_all.append(shifted_fea)
    aug_idx_all.extend([pid_to_idx[pid] for pid in shifted_pid])
for noise in noises:
    noise_fea_path = FEAT_PATH[:-4] + '_noise_{}.npy'.format(noise)
    noise_pid_path = FEAT_PATH[:-4] + '_noise_{}.txt'.format(noise)
    noise_fea = np.load(noise_fea_path)
    with open(noise_pid_path, 'r') as fin:
        noise_pid = fin.read().splitlines()
    aug_fea_all.append(noise_fea)
    aug_idx_all.extend([pid_to_idx[pid] for pid in noise_pid])

aug_fea_all = np.concatenate(aug_fea_all)
aug_fea_all = np.median(aug_fea_all, axis=2)
aug_idx_all = np.array(aug_idx_all)
print('Shifted data shapes:', aug_fea_all.shape, aug_idx_all.shape)

aug_fea_all = np.hstack([fea_all[aug_idx_all], aug_fea_all])
aug_ans_all = ans_all[aug_idx_all]
print('Combined shifted data shape:', aug_fea_all.shape, aug_ans_all.shape)

fea_all = np.hstack([fea_all, audio_fea_all])
print('Combined fea shape:', fea_all.shape)

# Training
pred_all = []
ans_for_eval_all = []
raw_idx_for_eval_all = []
best_n_est_all = []
for fold in range(N_FOLD):
    train_idx = np.where(index_all % N_FOLD != fold)[0]
    train_aug_idx = np.where(aug_idx_all % N_FOLD != fold)[0]
    test_idx = np.where(index_all % N_FOLD == fold)[0]
    print('===== Fold {}, tr num {}, te num {} ====='.format(fold, train_idx.size, test_idx.size))

    # 1st pass
    ans_tr = np.concatenate([ans_all[train_idx], aug_ans_all[train_aug_idx]])
    model = Classifier(**param)
    model.fit(
        np.concatenate([fea_all[train_idx], aug_fea_all[train_aug_idx]]),
        ans_tr,
        sample_weight=[ans_to_weight[ans] for ans in ans_tr],
        # eval_set=[(fea_all[test_idx], ans_all[test_idx])],
    )

    pred = model.predict_proba(fea_all[test_idx])
    pred_all.append(pred)
    ans_for_eval_all.append(ans_all[test_idx])
    raw_idx_for_eval_all.append(index_all[test_idx])

    # conf_mat, uar = util.get_conf_mat_and_uar(np.argmax(pred, axis=1), ans_all[test_idx])
    # print(conf_mat)
    # for i in range(conf_mat.shape[0]):
    #     print('Recall of class {}: {:.4f}%'.format(i, 100*conf_mat[i, i] / np.sum(conf_mat[i, :])))
    # print('UAR: {:.4f}%'.format(100*uar))

# cross validation results
print('Best n_ests:', best_n_est_all)

pred_all = np.vstack(pred_all)
ans_for_eval_all = np.hstack(ans_for_eval_all)
print('Finished, shapes:', pred_all.shape, ans_for_eval_all.shape)

conf_mat, uar = util.get_conf_mat_and_uar(np.argmax(pred_all, axis=1), ans_for_eval_all)
print(conf_mat)
for i in range(conf_mat.shape[0]):
    print('Recall of class {}: {:.4f}%'.format(i, 100*conf_mat[i, i] / np.sum(conf_mat[i, :])))
print('UAR: {:.4f}%'.format(100*uar))

if args.write_train_result:
    with open('results/xgb_params.csv', 'a') as cf:
        cf.write('{},{},{},{},{:.4f}%\n'.format(FEAT_PATH.split('/')[1].split('.')[0],param['n_estimators'], param['max_depth'], param['learning_rate'], 100*uar))

if args.test:
    print('public testing...')
    
    public_feat_path = os.path.join(os.path.join(args.feat_dir,args.public_feat))
    
    pid_pub, fea_pub, ans_pub = util.read_csv(os.path.join(args.data_dir,args.public_data), mode='test')
    index_pub = np.arange(fea_pub.shape[0])
    print('Data shapes:', len(pid_pub), fea_pub.shape, ans_pub.shape, index_pub.shape)

    audio_fea_pub = np.load(public_feat_path)
    audio_fea_pub = np.median(audio_fea_pub, axis=2)
    # audio_fea_pub = np.hstack([np.mean(audio_fea_pub, axis=2), np.median(audio_fea_pub, axis=2)])
    print('Audio fea shape:', audio_fea_pub.shape)

    fea_pub = np.hstack([fea_pub, audio_fea_pub])
    print('Combined fea shape:', fea_pub.shape)

    pred_pub = model.predict_proba(fea_pub)
    pub_pred = np.argmax(pred_pub, axis=1)

    print('private testing...')
    
    private_feat_path = os.path.join(os.path.join(args.feat_dir,args.private_feat))
    
    pid_pri, fea_pri, ans_pri = util.read_csv(os.path.join(args.data_dir,args.private_data), mode='test')
    index_pri = np.arange(fea_pri.shape[0])
    print('Data shapes:', len(pid_pri), fea_pri.shape, ans_pri.shape, index_pri.shape)

    audio_fea_pri = np.load(private_feat_path)
    audio_fea_pri = np.median(audio_fea_pri, axis=2)
    # audio_fea_pri = np.hstack([np.mean(audio_fea_pri, axis=2), np.median(audio_fea_pri, axis=2)])
    print('Audio fea shape:', audio_fea_pri.shape)

    fea_pri = np.hstack([fea_pri, audio_fea_pri])
    print('Combined fea shape:', fea_pri.shape)

    pred_pri = model.predict_proba(fea_pri)
    pri_pred = np.argmax(pred_pri, axis=1)

    with open('result/submission.csv', 'w') as fp:
        for pid, pred in zip(pid_pub, pub_pred):
            fp.write('{},{}\n'.format(pid, pred+1))
        for pid, pred in zip(pid_pri, pri_pred):
            fp.write('{},{}\n'.format(pid, pred+1))