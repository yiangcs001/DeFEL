import pandas as pd
from utils import sub_sequence, BASE_DIR


def create_data(dataset, IR=2):
    # read train.xlsx in {dataset} and separate positive samples from negative samples
    df = pd.read_excel(f'{BASE_DIR}/data/{dataset}/train.xlsx')
    df.seq = df.seq.map(lambda seq:sub_sequence(seq, 101))
    df = df.sample(frac=1)
    pos_data = df[df['label'] == 1]
    neg_data = df[df['label'] == 0]

    # divide the negative samples nto {10/IR} subsets
    n_neg = len(neg_data) # number of negative samples
    n_subsets = int(10/IR) # number of subsets
    n_neg_in_subsets = int(n_neg/n_subsets) # number of negative samples in each subset

    for i in range(n_subsets):
        # combine the positive samples and each negative samples subset to form a training subset
        start = i*n_neg_in_subsets
        end = min(start+n_neg_in_subsets, n_neg)
        subset_neg = neg_data.iloc[start:end]
        subset_pos = pos_data
        subset = pd.concat([subset_pos, subset_neg])
        subset = subset.sample(frac=1)

        subset.to_excel(f"{BASE_DIR}/data/{dataset}/train{i+1}.xlsx")


if __name__ == '__main__':
    # Set your dataset.
    for dataset in ['hf','hm','mf','mm']:
        create_data(dataset)