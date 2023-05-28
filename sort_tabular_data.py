import numpy as np

INPUT_CSV_FILE = 'data/train/training datalist.csv'
OUTPUT_CSV_FILE = 'data/train/training datalist_SORTED.csv'
# INPUT_CSV_FILE = 'data/Public Testing Dataset/training_datalist_public.csv'
# OUTPUT_CSV_FILE = 'data/Public Testing Dataset/testing_datalist_public_SORTED.csv'

with open(INPUT_CSV_FILE, 'r') as fin:
    cnt = fin.read().splitlines()

dis_cat_all = []
age_all = []
id_all = []
for line in cnt[1:]:
    arr = line.split(',')

    age = int(arr[2])
    age_all.append(age)

    dis_cat = int(arr[3])
    dis_cat_all.append(dis_cat)

    id_all.append(arr[0])

sorted_idx = np.lexsort([id_all, age_all, dis_cat_all])

with open(OUTPUT_CSV_FILE, 'w') as fout:
    fout.write(cnt[0] + '\n')
    for si in sorted_idx:
        fout.write(cnt[1:][si] + '\n')
