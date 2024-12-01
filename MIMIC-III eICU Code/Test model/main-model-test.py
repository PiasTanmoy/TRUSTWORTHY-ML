from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

import warnings
warnings.filterwarnings("ignore")

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger
import keras
import pandas as pd
import keras.backend as K
K.set_learning_phase(0)
def sigmoid_func(x):
    z=1/(1+np.exp(-x))
    return z

def tanh_func(x):
    z = np.tanh(x)
    return z

import shap

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()


if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

print('--------------------------------------------------------')
print(args,args.small_part, args.save_every)

print('--------------------------------------------------------')
# Build readers, discretizers, normalizers

# Tanmoy
# training with original dataset
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

# train with repair set
# train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train-repair/train-DBP-Glucose/Glucose_H2k_L2k_DBP_H2k_L2k_Original'),
#                                          listfile=os.path.join(args.data, 'train-repair/train-DBP-Glucose/Glucose_H2k_L2k_DBP_H1k_L1k_Original_2xshuffled.csv'),
#                                          period_length=48.0)
# train with repair set
# train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train-repair/train-temperature/Temp_C1_high_2k_orginal_train'),
#                                          listfile=os.path.join(args.data, 'train-repair/train-temperature/Temp_C1_high_500_original_train_name_list.csv'),
#                                          period_length=48.0)

# train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
#                                          listfile=os.path.join(args.data, 'train_listfile.csv'),
#                                          period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

# val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
#                                        listfile=os.path.join(args.data, 'test/listfile.csv'),
#                                        period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state

if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(normalizer_state)

normalizer.load_params(normalizer_state)

print(normalizer)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)


# print("sharmin printing......... ", train_raw[0].shape)
# print("sharmin checking..........",train_raw[0][0:50].shape )
# print("Pias check  ", target_repl)

if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

    print("sharmin printing......... ", train_raw[0].shape)
    print("sharmin checking..........", train_raw[0][0:50].shape)

# original
if args.mode == 'train-LSTM':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states_LSTM_Run1/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_LSTM_Run1_log')

    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


elif args.mode == 'train-LR':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states_LR_Run1/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_LR_Run1_log')

    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


elif args.mode == 'train-CW-LSTM':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states_CW_LSTM_Run1/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_CW_LSTM_Run1_log')

    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


# create by Tanmoy for repair ML
elif args.mode == 'train-repair-DBP':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states_Repair_Glucose_H2k_L2k_DBP_L1k_H1k_LSTM/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_Repair_Glucose_H2k_L2k_DBP_L1k_H1k_LSTM')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


# create by Tanmoy for repair ML
elif args.mode == 'train-repair-glucose':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states_Repair_glucose_H2k_L2k_LSTM/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_Repair_glucose_H2k_L2k_LSTM')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)




# create by Tanmoy for repair ML
elif args.mode == 'train-repair-Temperature-500':

    # Prepare training
    path = os.path.join(args.output_dir,
                        'keras_states_Repair_Temperature_500_high_LSTM3/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_Repair_Temperature_500_high_LSTM3')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


# create by Tanmoy for repair ML
elif args.mode == 'train-repair-DBP-Glucose':

    # Prepare training
    path = os.path.join(args.output_dir,
                        'keras_states_Repair_Glucose_H2k_L2k_DBP_L1k_H1k_LSTM3/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_Repair_Glucose_H2k_L2k_DBP_L1k_H1k_LSTM3')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


# create by Tanmoy for repair ML
elif args.mode == 'train-repair-Respiratory':

    # Prepare training
    path = os.path.join(args.output_dir,
                        'keras_states_Repair_Respiratory_L2k_1_10_H2k_LSTM/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose)

    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_states_Repair_Respiratory_L2k_1_10_H2k_LSTM')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit(x=train_raw[0],
              y=train_raw[1],
              validation_data=val_raw,
              epochs=n_trained_chunks + args.epochs,
              initial_epoch=n_trained_chunks,
              callbacks=[metrics_callback, saver, csv_logger],
              shuffle=True,
              verbose=args.verbose,
              batch_size=args.batch_size)


elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_final_base_repaired", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


elif args.mode == 'test-neuron-viz':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Diastolic BP set - Seed1 - SD 15'),
                                            listfile=os.path.join(args.data, 'test-2/Diastolic BP set - Seed1 - SD 15/name_list.csv'),
                                            period_length=48.0)

    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    # check output from lstm layer neurons
    print(model.layers)

    lstm_layer_access = model.layers[3]
    bidirectional_layer_access = model.layers[2]
    print("which layer are you accessing? ", lstm_layer_access)
    # attn_func = K.function(inputs = [model.layers[0].input], outputs = [bidirectional_layer_access.output, lstm_layer_access.output])
    #attn_func = K.function(inputs=[model.layers[0].input], outputs=[bidirectional_layer_access.output])

    attn_func = K.function(inputs=[model.layers[0].input], outputs=[lstm_layer_access.output])
    print("data type", type(data))

    # output_bi, output_lstm= attn_func([data])
    output_lstm = attn_func([data])

    print("output type", type(output_lstm))
    print("output len", len(output_lstm))
    print("output inside list shape", output_lstm[0].shape)

    output = output_lstm[0]
    bidirectional = True
    bidirectional = False

    if bidirectional:
        print("new value", output[0][:, -1].shape)
        output = output[:, -1]

    lstm_output_sigmoid_list = []
    # print(train['Glascow coma scale verbal response'])
    counter = 0
    for abc in range(output.shape[0]):
        temp_sig_output = sigmoid_func(output[abc])  # sigmoid_function
        print(counter, "temp_sig_output", temp_sig_output.shape)
        counter+=1
        # print("how about list", list(temp_sig_output))
        lstm_output_sigmoid_list.append(list(temp_sig_output))

    df_lstm_output = pd.DataFrame(lstm_output_sigmoid_list)
    print("df_lstm_output shape", df_lstm_output.shape)

    # print(df_lstm_output.head())
    #path2 = os.path.join(args.output_dir, "test_base_neuron_viz1", "df_bidirectional_layer_neuron_output_sigmoid") + ".csv"
    # df_lstm_output.to_csv(args.output_dir + "/test_base_neuron_viz1/df_bidirectional_layer_neuron_output_sigmoid.csv")  # output_function
    #df_lstm_output.to_csv('LSTM-Layer-Repaired-LSTM-Glucose-5.csv')
    df_lstm_output.to_csv('LSTM-Layer-DBP-Glu-Repaired-LSTM-DBP-1.csv')

    path = os.path.join(args.output_dir, "neuron_viz", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


elif args.mode == 'test-shap':

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    print(type(model))

    # sharmin added next two line
    print("model", type(model))
    print("model layers ", model.layers[0].input)
    print("model last layers", model.layers[-1].output)
    print("model bidirectional layer ", model.layers[3].input)
    print('Train_raw', type(train_raw[0]))
    print('Train_raw shape', train_raw[0].shape)

    attn_func = K.function(inputs=[model.layers[0].input], outputs=[model.layers[-1].output])

    e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), train_raw[0][0:8000],
                           keras.backend.get_session())

    test1 = train_raw[0][8000:11000]
    shap_val = e.shap_values(test1)

    print("shap_val ", shap_val)
    shap_val = np.array(shap_val)
    print("Tanmoy shap_val printing... ", shap_val.shape)
    shap_val = np.reshape(shap_val, (int(shap_val.shape[1]), int(shap_val.shape[2]), int(shap_val.shape[3])))
    print("Tanmoy printing reshape value ", shap_val.shape)
    shap_abs = np.absolute(shap_val)
    print("third line of printing ", shap_abs.shape)
    sum_0 = np.sum(shap_abs, axis=0)
    sum_00 = np.sum(sum_0, axis=0)
    print("forth printing ", sum_0.shape)
    f_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
               'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
               'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30',
               'F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40',
               'F41', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47', 'F48', 'F49', 'F50',
               'F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57', 'F58', 'F59', 'F60',
               'F61', 'F62', 'F63', 'F64', 'F65', 'F66', 'F67', 'F68', 'F69', 'F70',
               'F71', 'F72', 'F73', 'F74', 'F75', 'F76']

    x_pos = [i for i, _ in enumerate(f_names)]
    print("x_pos length printing ", len(x_pos))
    print("X_pos printing ", x_pos)
    print("printing sum_00 ", sum_00)
    for i in range(len(sum_00)):
        print(sum_00[i])

    d = {'SHAP': sum_00, 'F': f_names}
    df = pd.DataFrame(d)
    df.to_csv('SHAP-Repaired-Glu-DBP2.csv', index=False)

    # sum_00 should have 76 shap values for 76 encoded features--> (Capillary refill rate->0.0, Capillary refill rate->1.0, Diastolic blood pressure, Fraction inspired oxygen,	Glascow coma scale eye opening->To Pain, Glascow coma scale eye opening->3 To speech, Glascow coma scale eye opening->1 No Response, Glascow coma scale eye opening->4 Spontaneously, Glascow coma scale eye opening->None,	Glascow coma scale eye opening->To Speech, Glascow coma scale eye opening->Spontaneously, Glascow coma scale eye opening->2 To pain, Glascow coma scale motor response->1 No Response, Glascow coma scale motor response->3 Abnorm flexion, Glascow coma scale motor response->Abnormal extension, Glascow coma scale motor response->No response, Glascow coma scale motor response->4 Flex-withdraws, Glascow coma scale motor response->Localizes Pain, Glascow coma scale motor response->Flex-withdraws, Glascow coma scale motor response->Obeys Commands, Glascow coma scale motor response->Abnormal Flexion, Glascow coma scale motor response->6 Obeys Commands, Glascow coma scale motor response->5 Localizes Pain, Glascow coma scale motor response->2 Abnorm extensn, Glascow coma scale total->11, Glascow coma scale total->10, Glascow coma scale total->13, Glascow coma scale total->12, Glascow coma scale total->15, Glascow coma scale total->14, Glascow coma scale total->3, Glascow coma scale total->5, Glascow coma scale total->4, Glascow coma scale total->7, Glascow coma scale total->6, Glascow coma scale total->9, Glascow coma scale total->8, Glascow coma scale verbal response->1 No Response, Glascow coma scale verbal response->No Response, Glascow coma scale verbal response->Confused, Glascow coma scale verbal response->Inappropriate Words, Glascow coma scale verbal response->Oriented, Glascow coma scale verbal response->No Response-ETT, Glascow coma scale verbal response->5 Oriented, Glascow coma scale verbal response->Incomprehensible sounds, Glascow coma scale verbal response->1.0 ET/Trach, Glascow coma scale verbal response->4 Confused, Glascow coma scale verbal response->2 Incomp sounds, Glascow coma scale verbal response->3 Inapprop words, Glucose, Heart Rate, Height, Mean blood pressure, Oxygen saturation, Respiratory rate, Systolic blood pressure, Temperature, Weight, pH, mask->Capillary refill rate, mask->Diastolic blood pressure, mask->Fraction inspired oxygen, mask->Glascow coma scale eye opening, mask->Glascow coma scale motor response, mask->Glascow coma scale total, mask->Glascow coma scale verbal response, mask->Glucose, mask->Heart Rate, mask->Height, mask->Mean blood pressure, mask->Oxygen saturation, mask->Respiratory rate, mask->Systolic blood pressure, mask->Temperature, mask->Weight, mask->pH)
    # encoded features of same feaure variable is summed or averaged.

# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-val_set':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=48.0)

    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    #path = os.path.join(args.output_dir, "Repaired_DBP_C1_HL_4k_model_pred_respiratory_cases", os.path.basename(args.load_state)) + ".csv"
    path = os.path.join(args.output_dir, 'val_set_pred_LSTM_1', os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-Oxygen':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11'),
                                            listfile=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, 'pred-test-Oxy-5-SD-11-CWLSTM2-E24',
                        os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


    # this code sengment is wrriten by Tanmoy
elif args.mode == 'test-DBP-Glucose':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/DBP Glucose set - Seed1 - SD 15-48'),
                                            listfile=os.path.join(args.data, 'test-2/DBP Glucose set - Seed1 - SD 15-48/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    # path = os.path.join(args.output_dir, "Repaired_DBP_C1_HL_4k_model_pred_respiratory_cases", os.path.basename(args.load_state)) + ".csv"
    path = os.path.join(args.output_dir, 'pred-test-DBP-SD15-Glu-SD48-Seed1-DBP-2k-Glu-4k-repaired-LSTM-E40',
                        os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)



# this code segment is writen by Tanmoy
elif args.mode == 'test-glucose':
    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Glucose set - Seed5 - SD 48'),
                                            listfile=os.path.join(args.data, 'test-2/Glucose set - Seed5 - SD 48/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-Glucose-5-SD-48-Glucose-repaired-LSTM-L500-H500-E45", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code segment is writen by Tanmoy
elif args.mode == 'test-DBP':
    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw
    #Repaired-LSTM 2/Resp LSTM 1k_1_10 SD 5 E29.state
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Diastolic BP set - Seed1 - SD 15'),
                                            listfile=os.path.join(args.data, 'test-2/Diastolic BP set - Seed1 - SD 15/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]
    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-DBP-1-SD-15-repaired-Temp-LSTM-E46", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-respiratory':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Respiratory rate set - Seed5 - SD 5'),
                                            listfile=os.path.join(args.data, 'test-2/Respiratory rate set - Seed5 - SD 5/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-Resp-5-SD-15-repaired-DBP-Glu-LSTM-E40", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-temperature':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Temperature set - Seed5 - SD 0.7'),
                                            listfile=os.path.join(args.data, 'test-2/Temperature set - Seed5 - SD 0.7/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-temp-5-SD-0.7-repaired-DBP-Glu-LSTM-E40", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-oxygen':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11'),
                                            listfile=os.path.join(args.data, 'test-2/Oxygen set - Seed5 - SD 11/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-oxy-5-SD-11-repaired-DBP-Glu-LSTM-E40", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-SBP':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/Systolic BP set - Seed5 - SD 15'),
                                            listfile=os.path.join(args.data, 'test-2/Systolic BP set - Seed5 - SD 15/name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "pred-test-SBP-5-SD-15-repaired-DBP-Glu-LSTM-E40", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)



# this code sengment is wrriten by Tanmoy
elif args.mode == 'test-ideal':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/ideal_cases2/ideal_cases2'),
                                            listfile=os.path.join(args.data, 'test-2/ideal_cases2/ideal_cases_name_list.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test-ideal_repair_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


# this code segment is written by Tanmoy
elif args.mode == 'test-high-critial-cases':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/ideal_cases2/High'),
                                            listfile=os.path.join(args.data, 'test-2/ideal_cases2/name_list_high2.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test-High_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

# this code segment is written by Tanmoy
elif args.mode == 'test-low-critical-cases':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test-2/ideal_cases2/Low'),
                                            listfile=os.path.join(args.data, 'test-2/ideal_cases2/name_list_low2.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test-Low_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)



    # this code segment is written by Tanmoy
elif args.mode == 'test-trainset':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)
    ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                          return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test-train_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)


else:
    raise ValueError("Wrong value for args.mode")