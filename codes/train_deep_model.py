import os
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from DeepModel import OneHotDeepModel, ChemicalDeepModel
from utils import get_args, METRIC, batch_iter, update_metrics, need_to_update_model, load_sample, split_train_and_val, get_files_from_model_dir, files_with_name_x_in_dir_y, save_performance, output_performances, DEEP_MODEL_CONFIG


def train(xy_train, xy_val, params, index):
    epochs = params.epochs
    batch_size = params.batch_size
    dropout_rate = params.dropout_rate
    learning_rate = params.learning_rate

    if params.deep_model == 'onehot':
        model = OneHotDeepModel(dropout_rate=dropout_rate)
    else:
        model = ChemicalDeepModel(dropout_rate=dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_metrics_recorder = METRIC()
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_metrics_recorder = METRIC()

    # @tf.function
    def train_step(input_x, input_y, training, params):
        pos_weight = (1 + params.IR) / 1
        neg_weight = (1 + params.IR) / params.IR
        with tf.GradientTape() as tape:
            raw_prob = model(input_x, training)
            raw_prob = tf.clip_by_value(raw_prob, 1e-10, 1.0)
            raw_prob = tf.cast(raw_prob, 'float64')
            if params.IR == 1:
                pred_loss = binary_crossentropy(input_y, raw_prob)
            else:
                pred_loss = -tf.reduce_sum(pos_weight * input_y * tf.math.log(raw_prob[:, 1])
                                           + neg_weight * (1 - input_y) * (tf.math.log(1 - raw_prob[:, 1]))) / len(
                    input_y)
        if training:
            gradients = tape.gradient(pred_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return raw_prob, pred_loss

    has_model = True
    steps = 0
    ret_metrics = {'loss': None, 'auroc': None, 'auprc': None, 'acc': None, 'f1': None, 'mcc': None}
    # Counter, which records the number of times that the model is not updated continuously,
    # and breaks when it reaches 10 times
    cnt = 0
    for ep in range(1, epochs + 1):
        ##### training step #####
        train_loss_metric.reset_states()
        train_metrics_recorder.reset_state()

        batch_train = batch_iter(xy_train[0], xy_train[1], batch_size=batch_size)
        for batch_no, batch_tot, data_x, data_y in batch_train:
            train_prob, train_loss = train_step(data_x, data_y, True, params)
            train_prob = train_prob.numpy()[:, 1]
            train_loss_metric.update_state(train_loss)
            train_metrics_recorder.update_state(data_y, train_prob)
        print("[train ep %d]  [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f" %
              (ep,
               "loss", train_loss_metric.result().numpy(),
               "AUROC", train_metrics_recorder.result_auroc(),
               "AUPRC", train_metrics_recorder.result_auprc(),
               "ACC", train_metrics_recorder.result_acc(),
               "F1", train_metrics_recorder.result_f1(),
               "MCC", train_metrics_recorder.result_mcc(),
               )
              )

        ##### validation step ####
        val_loss_metric.reset_states()
        val_metrics_recorder.reset_state()

        batch_val = batch_iter(xy_val[0], xy_val[1], batch_size=batch_size)
        for batch_no, batch_tot, data_x, data_y in batch_val:
            val_prob, val_loss = train_step(data_x, data_y, False, params)
            val_prob = val_prob.numpy()[:, 1]
            val_loss_metric.update_state(val_loss)
            val_metrics_recorder.update_state(data_y, val_prob)
        val_metrics = update_metrics(val_loss_metric.result().numpy(), val_metrics_recorder.result_auroc(),
                                     val_metrics_recorder.result_auprc(), val_metrics_recorder.result_acc(),
                                     val_metrics_recorder.result_f1(), val_metrics_recorder.result_mcc())
        print("[val ep %d]  [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f" %
              (ep,
               "loss", val_metrics['loss'],
               "AUROC", val_metrics['auroc'],
               "AUPRC", val_metrics['auprc'],
               "ACC", val_metrics['acc'],
               "F1", val_metrics['f1'],
               "MCC", val_metrics['mcc']
               )
              )

        if val_metrics['auroc'] <= 0.5:
            print("try again when auroc <= 0.50.")
            if ep == 1:
                has_model = False
            break

        # Compare with the current optimal metrics,
        # decide whether to save the current model,
        # and update the metrics.
        if need_to_update_model(before_metrics=ret_metrics, cur_metrics=val_metrics):
            cnt = 0
            model.save_weights(f"{params.model_dir}/model{index}_temp")
            steps = ((len(xy_train[1]) + batch_size - 1) // batch_size) * ep
            ret_metrics = update_metrics(loss=val_metrics['loss'], auroc=val_metrics['auroc'],
                                         auprc=val_metrics['auprc'], acc=val_metrics['acc'],
                                         f1=val_metrics['f1'], mcc=val_metrics['mcc'])
        else:
            cnt += 1

        # Break the loop when the model does not update for ten epochs.
        if cnt == 10:
            break

    return has_model, ret_metrics, steps


def run_train(params, input_shape=(101, 4, 1)):
    if params.deep_model == "chemical":
        input_shape = (101, 3, 1)

    performances = {}  # Record the optimal metrics of each model.
    for index in range(1, int(10 / params.IR) + 1):
        print(f"##### model{index} #####")
        train_data_file = f"{params.data_dir}/train{index}.xlsx"
        if not os.path.exists(train_data_file):
            print("run codes/create_data_4_deep_model.py before run codes/train_deep_model.py")
            exit(1)
        train_val_x, train_val_y = load_sample(train_data_file, input_shape)
        train_x, train_y, val_x, val_y = split_train_and_val(train_val_x, train_val_y)

        best_metrics = {'loss': None, 'auroc': None, 'auprc': None, 'acc': None, 'f1': None, 'mcc': None}
        steps = 0
        for iter in range(params.iterations):
            print(f"*** [iter-{iter + 1}] ***")
            has_model, cur_metrics, cur_steps = train([train_x, train_y], [val_x, val_y], params, index)

            if not has_model:
                continue
            # Decide whether to update the model according to the metrics.
            if need_to_update_model(before_metrics=best_metrics, cur_metrics=cur_metrics):
                # Metrics has been improved, and update the model.
                files_in_model_dir = get_files_from_model_dir(params)
                #   1.If there is a model before, delete the previous model.
                for file in files_with_name_x_in_dir_y(f"model{index}.", files_in_model_dir):
                    os.remove(f"{params.model_dir}/{file}")
                #   2.Rename the current model "model{index}_temp" to "model{index}".
                for file in files_with_name_x_in_dir_y(f"model{index}_temp.", files_in_model_dir):
                    suffix = file.split('.')[1]  # File extension
                    os.rename(f"{params.model_dir}/{file}", f"{params.model_dir}/model{index}.{suffix}")
                #   3.Update best_metrics and steps.
                steps = cur_steps
                best_metrics = update_metrics(loss=cur_metrics['loss'], auroc=cur_metrics['auroc'],
                                              auprc=cur_metrics['auprc'], acc=cur_metrics['acc'],
                                              f1=cur_metrics['f1'], mcc=cur_metrics['mcc'])
            else:
                # Do not update the model, and delete the model.
                files_in_model_dir = get_files_from_model_dir(params)
                for file in files_with_name_x_in_dir_y(f"model{index}_temp.", files_in_model_dir):
                    os.remove(f"{params.model_dir}/{file}")
        save_performance(performances, index, best_metrics, steps)
    # Output the best metrics of each model as a file model_performance.txt.
    output_performances(performances, params.performance_file, params.deep_model)


if __name__ == "__main__":
    print('train_deep_model.py start.')

    config_file = get_args("deep_model_config.yaml")
    params = DEEP_MODEL_CONFIG(config_file)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    run_train(params)

    print('train_deep_model.py end.')

