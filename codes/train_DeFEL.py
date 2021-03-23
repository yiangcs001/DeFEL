import random
import numpy as np
import tensorflow as tf
from joblib import dump
from multiprocessing import Process
from utils import DEFEL_CONFIG, get_args
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from EncoderModel import extract_features_from_raw_data


def get_train_data(params):
    """
    Load training data and test data,
    and split the training data into {params.IR} training subsets.
    """
    print('data preprocessing...')

    print("train data:\n")
    train_data, train_label = extract_features_from_raw_data(params.train_file, params)
    train_data, train_label = split_data(train_data, train_label, params.IR)

    return train_data,train_label


def split_data(data,labels,IR):
    """
    Split the negative samples of the training data into {IR} subsets,
    and then combine each negative samples subset with the positive samples to form {IR} training subsets.
    """
    pos_index = list(np.where(labels == 1)[0])
    neg_indices = list(np.where(labels == 0)[0])
    random.shuffle(neg_indices)
    neg_index_list = [[] for i in range(IR)]
    n_neg = np.where(labels == 0)[0].shape[0]
    n_each_neg = int(n_neg/IR)
    n_encoder = len(data)

    for i in range(IR):
        start = i*n_each_neg
        end = min(start+n_each_neg, n_neg)
        neg_index = neg_indices[start:end]
        neg_index_list[i] = neg_index

    pos_data = []
    neg_data = []
    for i in range(n_encoder):
        pos_data.append(np.array(data[i][pos_index]))
        sub_neg_data = []
        for j in range(IR):
            sub_neg_data.append(np.array(data[i][neg_index_list[j]]))
        neg_data.append(sub_neg_data)

    ret_data = {'positive': pos_data, 'negative': neg_data}
    ret_labels = {'positive': np.array(labels[pos_index]), 'negative': [np.array(labels[neg_index_list[i]]) for i in range(IR)]}

    return ret_data, ret_labels


def task(feat_id, task_id, train_X, train_Y, params):
    rf = RandomForestClassifier(n_estimators=300).fit(train_X, train_Y)
    lr = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial',
                            class_weight='balanced', max_iter=2000, penalty='l1').fit(train_X, train_Y)

    dump(rf, f"{params.rf_model_dir}/{feat_id}_{task_id}_rf.joblib", compress=9)
    dump(lr, f"{params.lr_model_dir}/{feat_id}_{task_id}_lr.joblib", compress=9)
    print(f"proc-{task_id} exit.")


def train_defel(train_data, train_label, params):
    num_features = len(train_data['positive'])
    num_sub_train = params.IR

    print("Trainin & Testing...")
    for i in range(num_features):
        print(f"### encoder-{i+1} ###")

        p_list = []
        for j in range(num_sub_train):
            train_X = np.concatenate((train_data['positive'][i], train_data['negative'][i][j]))
            train_Y = np.concatenate((train_label['positive'], train_label['negative'][j]))
            p = Process(target=task, args=(i, j, train_X, train_Y, params,))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()


def run(params):

    train_data, train_label = get_train_data(params)

    train_defel(train_data, train_label, params)


if __name__ == "__main__":
    print('train_DeFEL.py start.')

    config_file = get_args("defel_config.yaml")
    params = DEFEL_CONFIG(config_file)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    run(params)

    print('train_DeFEL.py end.')