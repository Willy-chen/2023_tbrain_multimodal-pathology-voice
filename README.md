# 2023_tbrain_multimodal-pathology-voice
# TEAM_3410

This is our implementation of this contest: https://tbrain.trendmicro.com.tw/Competitions/Details/27. 
We got rank 5 in the Private Leaderboard and rank 3 in the public Leaderboard.

## Run Our Implementation

### Required libraries

- librosa
- xgboost
- numpy

Required for PANNs implementation:
- pytorch 
- torchlibrosa 

PANNs model can be downloaded from the following url: <br>
https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth?download=1 <br>
** Versions are not restricted as long as they're new enough. **

### Preprocess
* The official csv and wav file should be in `data` dir.
```bash
python3 sort_tabular_data.py
```
* sort the train csv file in order of label, age and id.
* output sorted csv file to `data/train`.
### Data Augmentation
```bash
python3 data_aug.py
```
* output noise and shift training audio to `data/train` dir.

### Feature Extraction
```bash
python3 audio_fea_extract.py --mode train
python3 audio_fea_extract.py --mode public
python3 audio_fea_extract.py --mode private
python3 audio_fea_extract.py --mode train -data_aug
```
* mode: dataset to extract feature
* feat_type: audio feature extraction methods, including most librosa feature functions, e.g. mel, mfcc, stft...
* frame_size
* hop_size
* n_mel_bin
* n_mfcc
* n_chroma
* align_op: methods to align different length features, e.g. 'cut', 'pad', 'pre-avg' 
* data_aug: exrtact feature from augmented audio data
* output npy file is in the corresponding dataset dir in `data` dir.
### Training
#### Single layer XGBoost 
```bash
python3 cv_xgb.py
```
* train and show model cross validation score
#### Ensemble of PANNs and XGBoost
```bash
cd ensemble
python3 ensemble_PANNs_cv_xgb.py
```
* train and show model cross validation score

### Inference

#### Single layer XGBoost 
```bash
python3 cv_xgb.py -t
```
* `-t`: option for testing on both public and private dataset.
* output is in the `result` dir as submission.csv
#### Ensemble of PANNs and XGBoost
```bash
cd ensemble
python3 ensemble_PANNs_cv_xgb.py -t
```
* `n_fold`: number of cross validation folds.
* output is in the `result` dir as submission.csv

## 簡介
### 問題概述

* 提供資料： 每位病患嗓音的錄音檔以及其 26 項個人相關資料
* 預測目標： 每位病患 ID 所對應的五類嗓音疾病
* 評估標準： UAR (Unweighted Average Recall)
![UAR](img/UAR.png)

### 執行環境

程式語言為 Python 3，未特別指定版本，使用 miniconda 創建環境；函式庫如本說明前半部份所示，pytorch 僅有 PANNs 相關實驗所使用。

### 特徵截取

官方給出的資料格式，是每位病患的 1~3 秒鐘的錄音以及 26 項病患的相關資料。音檔的部份我們使用的是 mel spectrogram 來取得 128 維的特徵，並將所以特徵統一擷取至最短的長度後，對所有的特徵取中位數。表格資料的部份就直觀的取其數值，缺值補零作為特徵。


### 預測目標

共五個類別，分別代表病患五種不同類型的病理嗓音
### 模型設計與訓練

本次比賽使用的模型架構如下圖，主體為 XGBoost 分類器

![model](img/model.png)

訓練方式為 7 folds cross validation，細節參數如下，未提及之參數係依照 xgboost 預設值，未進行修改：
*	n estimators: 210, 
*	max depth: 7, 
*	learning rate: 0.055, 
*	eval metric: mlogloss, 


由此產生出的 7 個模型，我們取用最後一個模型進行預測。

### 預測

預測時會只取最後一個模型的結果，取機率的 argmax 作為預測結果。
