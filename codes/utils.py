import os
import sys
import copy
import yaml
import getopt
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent.parent


class METRIC:
    def __init__(self):
        self.loss = []
        self.proba = []
        self.y_pred = []
        self.y_true = []
        self.has_state = False
    def update_state(self, y_true, proba):
        self.proba.extend(proba)
        self.y_true.extend(y_true)
        y_pred = [1 if s >= 0.5 else 0 for s in proba]
        self.y_pred.extend(y_pred)
        self.has_state = True
    def reset_state(self):
        self.loss = []
        self.proba = []
        self.y_pred = []
        self.y_true = []
        self.has_state = False
    def result_auroc(self):
        if not self.has_state:
            return None
        auroc = metrics.roc_auc_score(self.y_true, self.proba)
        return auroc
    def result_auprc(self):
        if not self.has_state:
            return None
        precision, recall, threshold = metrics.precision_recall_curve(self.y_true, self.proba)
        auprc = metrics.auc(recall, precision)
        return auprc
    def result_acc(self):
        if not self.has_state:
            return None
        acc = metrics.accuracy_score(self.y_true, self.y_pred)
        return acc
    def result_f1(self):
        f1 = metrics.f1_score(self.y_true, self.y_pred)
        return f1
    def result_mcc(self):
        if not self.has_state:
            return None
        mcc = metrics.matthews_corrcoef(self.y_true, self.y_pred)
        return mcc


class DEEP_MODEL_CONFIG:
    """
    codes/deep_model_config.yaml
    """
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            yaml_data = yaml.full_load(f)

        # read parameters of codes/deep_model_config.yaml
        self.dataset = yaml_data['dataset']
        self.IR = yaml_data['IR']
        self.deep_model = yaml_data['deep_model']
        self.iterations = yaml_data['iterations']
        self.epochs = yaml_data['epochs']
        self.batch_size = yaml_data['batch_size']
        self.dropout_rate = yaml_data['dropout_rate']
        self.learning_rate = yaml_data['learning_rate']

        # set local parameters
        self.data_dir = f"{BASE_DIR}/data/{self.dataset}"
        self.model_dir = f"{BASE_DIR}/models/{self.dataset}/DM/{self.deep_model}"
        self.performance_file = f"{BASE_DIR}/models/{self.dataset}/DM/deep_models_performances.txt"

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


class DEFEL_CONFIG:
    """
    codes/defel_config.yaml
    """

    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            yaml_data = yaml.full_load(f)
        # read parameters of defel_config.yaml
        self.dataset = yaml_data['dataset'].split('_')[0]
        self.w_rf = yaml_data['weight_of_random_forest']
        self.w_lr = yaml_data['weight_of_logistic_regression']
        self.weights = yaml_data['weights']
        self.IR = yaml_data['IR']
        self.set_weights()

        # sequence length of different features
        self.spectrum_seq_len = 253 if self.dataset in ['hf', 'mf'] else 111
        self.decimal_seq_len = 21
        self.graph_seq_len = 81
        self.onehot_seq_len = 101
        self.chemical_seq_len = 101

        # set local parameters
        self.train_file = f"{BASE_DIR}/data/{self.dataset}/train.xlsx"
        self.test_file = f"{BASE_DIR}/data/{self.dataset}/test.xlsx"
        self.metrics_file = f"{BASE_DIR}/outputs/{self.dataset}/metrics.xlsx"
        self.scores_dir = f"{BASE_DIR}/outputs/{self.dataset}/scores"
        self.rf_model_dir = f"{BASE_DIR}/models/{self.dataset}/RF"
        self.lr_model_dir = f"{BASE_DIR}/models/{self.dataset}/LR"
        self.deep_onehot_model_dir = f"{BASE_DIR}/models/{self.dataset}/DM/onehot"
        self.deep_chemical_model_dir = f"{BASE_DIR}/models/{self.dataset}/DM/chemical"

        for dir in [self.scores_dir, self.rf_model_dir, self.lr_model_dir, self.deep_onehot_model_dir, self.deep_chemical_model_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)


    def set_weights(self):
        if self.weights == 'auto':
            self.weights = np.array(get_weights(n_weight=5))
        elif(self.weights.__class__ == dict):
            spectrum = self.weights['spectrum']
            decimal = self.weights['decimal']
            graph = self.weights['graph']
            onehot = self.weights['onehot']
            chemical = self.weights['chemical']
            if spectrum + decimal + graph + onehot + chemical == 1:
                self.weights = np.array([[spectrum, decimal, graph, onehot, chemical]])
            else:
                print('the sum of weights must be 1')
                exit(1)
        else:
            print("weights: auto or use a dict to assign weights")
            exit(1)


def sub_sequence(seq,seq_len):
    central_index = int(len(seq)/2)

    return seq[central_index-int((seq_len-1)/2):central_index+int((seq_len-1)/2)+1]


def batch_iter(x, y, batch_size=16):
    data_len = len(x)
    num_batch = (data_len + batch_size - 1) // batch_size
    indices = np.random.permutation(np.arange(data_len))
    x_shuff = x[indices]
    y_shuff = y[indices]
    for i in range(num_batch):
        start_offset = i * batch_size
        end_offset = min(start_offset + batch_size, data_len)
        yield i, num_batch, x_shuff[start_offset:end_offset], y_shuff[start_offset:end_offset]


def get_weights(n_weight=3,step=0.05,weights = [],w = []):
    if len(w) == n_weight:
        if sum(w) == 1:
            weights.append(copy.copy(w))
        return weights
    else:
        for i in range(21):
            w.append(round(i*step,3))
            weights = get_weights(n_weight,step,weights,w)
            w.pop()
        return weights


def update_metrics(loss, auroc, auprc, acc, f1, mcc):
    metrics = {}
    metrics['loss'] = loss
    metrics['auroc'] = auroc
    metrics['auprc'] = auprc
    metrics['acc'] = acc
    metrics['f1'] = f1
    metrics['mcc'] = mcc
    return metrics


def need_to_update_model(before_metrics: dict, cur_metrics: dict):
    # The model has not been saved before,
    # and all metrics are None at this time,
    # then directly return True to save the current model.
    if None in list(before_metrics.values()):
        return True
    # If there is a model before,
    # compare the metrics and then decide whether to save the current model.
    metrics_name = ['auroc', 'auprc', 'acc', 'f1', 'mcc']
    before_metrics_list = [before_metrics[name] for name in metrics_name]
    cur_metrics_list = [cur_metrics[name] for name in metrics_name]
    for c_metric, b_metric in zip(cur_metrics_list, before_metrics_list):
        if c_metric > b_metric:
            return True
        if c_metric < b_metric:
            return False
    #  If the metrics are all equal, finally compare the loss.
    if cur_metrics['loss'] < before_metrics['loss']:
        return True
    if cur_metrics['loss'] > before_metrics['loss']:
        return False
    # If metrics are all equal, then don’t update.
    return False


def load_sample(fn,input_shape):
    """ load samples for codes/train_deep_model.py"""
    def one_hot(seq):
        encoder = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        encoding = []
        for nu in seq:
            encoding.append(encoder[nu])
        return np.array(encoding)

    def chemical(seq):
        encoder = {'A': [1, 1, 1], 'G': [1, 0, 0], 'C': [0, 1, 0], 'T': [0, 0, 1]}
        encoding = []
        for nu in seq:
            encoding.append(encoder[nu])
        return encoding

    if input_shape == (101,4,1):
        encoding_func = one_hot
    elif input_shape == (101,3,1):
        encoding_func = chemical
    else:
        print("input_shape not equal to (101,4,1) or (101,3,1)")
        exit(1)

    df = pd.read_excel(fn)
    data = np.array(list(map(lambda seq:encoding_func(sub_sequence(seq, 101)), df.seq)))\
        .astype(float)\
        .reshape([len(df.seq), input_shape[0], input_shape[1], input_shape[2]])
    labels = df.label.to_numpy()

    return data, labels


def split_train_and_val(x, y, r=0.1):
    """
    Split the training subset into training data and validation data
    Default 10% as verification data.
    """
    # Get the indexes of positive and negative samples.
    pos_indices = list(np.where(y == 1)[0])
    neg_indices = list(np.where(y == 0)[0])
    # Shuffle the indexes of positive and negative samples,
    # so that the training data and validation data obtained
    # by running codes/train_deep_model.py each time are different.
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)
    # Calculate the number of positive and negative samples.
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    # Calculate the number of positive and negative samples of the validation data.
    n_pos_val = int(n_pos * r)
    n_neg_val = int(n_neg * r)
    # Get the indexes of the validation data.
    val_pos_indices = pos_indices[:n_pos_val]
    val_neg_indices = neg_indices[:n_neg_val]
    val_indices = val_pos_indices + val_neg_indices
    random.shuffle(val_indices)
    # Get the indexes of the training data.
    train_pos_indices = pos_indices[n_pos_val:]
    train_neg_indices = neg_indices[n_neg_val:]
    train_indices = train_pos_indices + train_neg_indices
    random.shuffle(train_indices)

    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]


def get_files_from_model_dir(params):
    files = []
    for root, subdir, file_names in os.walk(params.model_dir):
        for file in file_names:
            files.append(file)
    return files


def files_with_name_x_in_dir_y(x, ys):
    files = []
    for y in ys:
        if x in y:
            files.append(y)
    return files


def save_performance(performances, index, best_metrics, steps):
    performances[index] = {
        'auroc': best_metrics['auroc'],
        'auprc': best_metrics['auprc'],
        'acc': best_metrics['acc'],
        'f1': best_metrics['f1'],
        'mcc': best_metrics['mcc'],
        'loss': best_metrics['loss'],
        'steps': steps
    }


def output_performances(performances, path, deep_model):
    with open(path, 'a') as f:
        for index in list(performances.keys()):
            auroc = performances[index]['auroc']
            auprc = performances[index]['auprc']
            acc = performances[index]['acc']
            f1 = performances[index]['f1']
            mcc = performances[index]['mcc']
            loss = performances[index]['loss']
            steps = performances[index]['steps']
            f.write(
                f"deep_{deep_model}_{index}\t[AUROC]:{auroc} [AUPRC]:{auprc} [ACC]:{acc} [F1]:{f1} [MCC]:{mcc} [LOSS]:{loss} [STEPS]:{steps}\n")


def calculate_metrics(y_true, y_score):
    def calculate_auroc(y_true, y_pred):
        fpr, tpr, thresholds1 = metrics.roc_curve(y_true, y_pred)
        auroc = metrics.auc(fpr, tpr)
        return auroc

    def calculate_auprc(y_true, y_pred):
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        auprc = metrics.auc(recall, precision)
        return auprc

    def cal_accuracy(y_true, y_pred):
        acc = metrics.accuracy_score(y_true, y_pred)
        return acc

    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    auroc = calculate_auroc(y_true, y_score)
    auprc = calculate_auprc(y_true, y_score)
    acc = cal_accuracy(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    return [auroc, auprc, acc, f1, mcc]

def get_args(fn):
    config_file   = fn  # -f

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:",
                                   ["help", "config_file="])
    except getopt.GetoptError:
        if fn == "deep_model_config.yaml":
            print("Error: deep_model_config.py -f <Configuration File>")
        elif fn == "defel_config.yaml":
            print("Error: train_DeFEL.py -f <Configuration File> or test_DeFEL.py -f <Configuration File>")
        sys.exit()

    for opt, arg in opts:
        if (opt not in ["-h", "--help", "-f"]):
            print("Error:The configuration file, which sets the parameters, is necessary.")
            sys.exit()

    for opt, arg in opts:
        if (opt in ("-h", "--help")):
            print(
                "Try: train_deep_model.py/train_DeFEL.py/test_DeFEL.py -f <Configuration File>")
            sys.exit()
        elif (opt in ("-f")):
            config_file = arg
            if os.path.splitext(config_file)[-1] != ".yaml":
                print("Error: The extension of configuration file must be .yaml.")
                sys.exit()
            continue
    return config_file