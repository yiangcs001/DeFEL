import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from multiprocessing import Process, Manager
from EncoderModel import extract_features_from_raw_data
from utils import DEFEL_CONFIG, calculate_metrics, get_args


def get_test_data(params):
    print("test data:\n")
    test_data, test_label = extract_features_from_raw_data(params.test_file, params)

    return test_data, test_label


def evaluate(y_true,pred_scores,params):

    print('caculating metrics...')
    weights = params.weights
    y_scores = np.matmul(weights, pred_scores)

    # metrics.csv
    metrics = np.array([calculate_metrics(y_true, y_score) for y_score in y_scores])
    print('output file...')
    df = pd.DataFrame(metrics)
    df.columns = ["AUROC", "AUPRC", "ACC", "F1", "MCC"]
    df.index = ['_'.join(weight) for weight in weights.astype(str)]
    df.to_excel(params.metrics_file)

    # *_score.txt
    for weight, scores in zip(weights.astype(str), y_scores):
        with open(f"{params.scores_dir}/{'_'.join(weight)}_score.txt", "w") as f:
            for score in scores:
                f.write(str(score) + "\n")


def task(feat_id, task_id, shared_list, test_X, params):
    rf = load(f"{params.rf_model_dir}/{feat_id}_{task_id}_rf.joblib")
    lr = load(f"{params.rf_model_dir}/{feat_id}_{task_id}_lr.joblib")

    rf_pred = rf.predict_proba(test_X)[:, 1]
    lr_pred = lr.predict_proba(test_X)[:, 1]
    pred =  params.w_rf * rf_pred + params.w_lr * lr_pred

    shared_list.append(pred)
    print(f"proc-{task_id} exit.")


def test_model(test_data, params):
    num_features = len(test_data)
    num_samples = len(test_data[0])
    prediction_score = [np.zeros(num_samples,dtype=float) for _ in range(num_features)]

    print("Trainin & Testing...")
    for i in range(num_features):
        print(f"### encoder-{i+1} ###")

        manager = Manager()
        shared_list = manager.list()
        p_list = []
        for j in range(params.IR):
            p = Process(target=task,
                        args=(i, j, shared_list, test_data[i], params,))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()

        if len(shared_list) != params.IR:
            print("something wrong in multi processes.")
            exit(1)
        for pred in shared_list:
            prediction_score[i] += pred
        prediction_score[i] /= params.IR

    return np.array(prediction_score)


def run_test(params):

    test_data, test_label = get_test_data(params)

    prediction_score = test_model(test_data, params)

    evaluate(test_label, prediction_score, params)


if __name__ == "__main__":
    print('test_DeFEL.py start.')

    config_file = get_args("defel_config.yaml")
    params = DEFEL_CONFIG(config_file)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    run_test(params)

    print('test_DeFEL.py end.')