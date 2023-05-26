import numpy as np


def read_csv(path, elim = [], mode = 'train'):
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
        else:
            raise Exception('Unsupported mode: {}'.format(mode))         
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