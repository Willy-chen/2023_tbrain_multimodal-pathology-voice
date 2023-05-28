import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


def read_csv(path='data/Training Dataset/training datalist_SORTED.csv', elim = [], mode = 'train'):
    with open(path, 'r') as fin:
        cnt = fin.read().splitlines()[1:]
    pid_all = []
    fea_all = []
    ans_all = []
    for line in cnt:
        if mode == 'train':
            pid, fea, ans = arr_to_xgb_fea(line.split(','), elim, mode)
        elif mode == 'test':
            pid, fea, ans = arr_to_xgb_fea(line.split(','), elim, mode)            
        pid_all.append(pid)
        fea_all.append(fea)
        ans_all.append(ans)
    fea_all = np.vstack(fea_all)
    ans_all = np.array(ans_all)
    return pid_all, fea_all, ans_all


def arr_to_xgb_fea(arr, elim = [], mode = 'train'):
    pid = arr[0]
    if mode == 'train':
        fea = [
            int(arr[1]), # sex
            int(arr[2]), # age
            int(arr[4]),
            int(arr[5]),
            int(arr[6]),
            int(arr[7]),
            int(arr[8]),
            int(arr[9]),
            int(arr[10]),
            int(arr[11]),
            int(arr[12]),
            int(arr[13]),
            0 if arr[14] == '' else float(arr[14]), # PPD
            int(arr[15]),
            int(arr[16]),
            int(arr[17]),
            int(arr[18]),
            int(arr[19]),
            int(arr[20]),
            int(arr[21]),
            int(arr[22]),
            int(arr[23]),
            int(arr[24]),
            int(arr[25]),
            int(arr[26]),
            0 if arr[27] == '' else float(arr[27]), # Voice handicap index (VHI)
        ]
    elif mode == 'test':
        fea = [
            int(arr[1]), # sex
            int(arr[2]), # age
            int(arr[3]),
            int(arr[4]),
            int(arr[5]),
            int(arr[6]),
            int(arr[7]),
            int(arr[8]),
            int(arr[9]),
            int(arr[10]),
            int(arr[11]),
            int(arr[12]),
            0 if arr[13] == '' else float(arr[13]), # PPD
            int(arr[14]),
            int(arr[15]),
            int(arr[16]),
            int(arr[17]),
            int(arr[18]),
            int(arr[19]),
            int(arr[20]),
            int(arr[21]),
            int(arr[22]),
            int(arr[23]),
            int(arr[24]),
            int(arr[25]),
            0 if arr[26] == '' else float(arr[26]), # Voice handicap index (VHI)
        ]
    fea = [i for j, i in enumerate(fea) if j not in elim]
    
    ans = []
    if mode == 'train':
        ans = int(arr[3]) - 1 # [1, 5] to [0, 4]

    return pid, fea, ans


def get_categories(path='data/Training Dataset/training datalist_SORTED.csv'):
    with open(path, 'r') as fin:
        cat = fin.read().splitlines()[0].split(',')
    return cat


def get_conf_mat_and_uar(pred, ans, n_class=5):
    conf_mat = np.zeros((n_class, n_class)).astype('int')
    for gt, pred in zip(ans, pred):
        conf_mat[gt, pred] += 1

    recall_all = []
    for i in range(n_class):
        recall_all.append(conf_mat[i, i] / np.sum(conf_mat[i, :]))
    
    return conf_mat, np.mean(recall_all)


class Data2TorchAE(Dataset):
    def __init__(self, fea):
        self.fea = fea

    def __getitem__(self, index):
        return torch.from_numpy(self.fea[index]).float()

    def __len__(self):
        return self.fea.shape[0]


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            # nn.Linear(2757, 1),
            # nn.Flatten(),
            # nn.ReLU()
        )

        # Decoder network
        self.decoder = nn.Sequential(
            # nn.Unflatten(1, (128, 1)),
            # nn.Linear(1, 2757),
            # nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        self.pool = nn.MaxPool1d(2757, return_indices=True)
        self.unpool = nn.MaxUnpool1d(2757)

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        encoded, index = self.pool(encoded)
        decoded = self.unpool(encoded, index)
        decoded = self.decoder(decoded)
        return encoded.squeeze(dim=2), decoded


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(16, 1, 44100).to(device)
    model = Autoencoder().to(device)
    y = model(x)
