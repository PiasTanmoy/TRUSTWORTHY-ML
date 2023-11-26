from __future__ import print_function,absolute_import
'''
usage: python gen_diff.py -h
'''
'''
ALl the imports are here
'''


import argparse
from keras.datasets import mnist
from keras.layers import Input
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import imageio
imsave = imageio.imsave
from configs import bcolors
from utils import *
import numpy as np
import argparse
import os
import imp
import re
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout
#from tensorflow.keras.layers.wrappers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask

from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# these 2 lines are added to make the MNIST the source folder for importing Models
# these are not required while running from terminal
# online required if running from Python Console
#####################################################################


'''
This is to set the working directory
'''

import os
os.chdir('/home/tanmoy/Downloads/Trustworthiness/deepxplore/MIMIC-III')
import sys
sys.path.append('/home/tanmoy/Downloads/Trustworthiness/deepxplore/MIMIC-III')


'''
This class is created to replace the arg input from command line.
The bypass of arg input from command line was generating some
weird outputs. 
'''

class args:
    transformation = 'blackout'
    weight_diff = 1
    weight_nc = 0.1
    step = 1
    seeds = 10
    grad_iterations = 1000
    threshold = 0
    id = 1
    target_model = 0
    start_point = (0, 0)
    occlusion_size = (10, 10)


    target_repl_coef = 0.0
    data = os.path.join(os.path.dirname(__file__), 'data/in-hospital-mortality/')
    output_dir = 'mimic3models/in_hospital_mortality'
    save_every = 1
    batch_norm = False
    batch_size = 8
    beta_1 = 0.9
    depth = 2
    dim = 16
    dropout = 0.3
    epochs = 100
    imputation = 'previous'
    l1 = 0
    l2 = 0


    load_state1 = "mimic3models/in_hospital_mortality/Base_Models/LSTM/keras_states_LSTM_Run1/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch39.test0.2848591662846242.state"
    load_state2 = "mimic3models/in_hospital_mortality/Base_Models/LSTM/keras_states_LSTM_Run2/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch29.test0.284651877840693.state"
    load_state3 = "mimic3models/in_hospital_mortality/Base_Models/LSTM/keras_states_LSTM_Run3/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch33.test0.283285996203168.state"


    lr = 0.001
    mode = 'test'
    network = 'mimic3models/keras_models/lstm.py'
    normalizer_state = None
    optimizer = 'adam'
    output_dir = 'mimic3models/in_hospital_mortality'
    prefix = ''
    rec_dropout = 0.0
    save_every = 1
    size_coef = 4.0
    small_part = False
    target_repl_coef = 0.0
    timestep = 1.0
    verbose = 2


# This is the standard (example) values of different arguments
'''
Namespace(batch_norm=False, batch_size=8, beta_1=0.9, 
data='/home/tanmoy/Downloads/Trustworthiness/mimic3-benchmarks/mimic3models/in_hospital_mortality/../../data/in-hospital-mortality/', 
depth=2, dim=16, dropout=0.3, epochs=100, imputation='previous', l1=0, l2=0, 
load_state='mimic3models/in_hospital_mortality/keras_states_350_oversampled/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch20.test0.28812097264586733.state', 
lr=0.001, mode='test', network='mimic3models/keras_models/lstm.py', normalizer_state=None, 
optimizer='adam', output_dir='mimic3models/in_hospital_mortality', prefix='', rec_dropout=0.0, 
save_every=1, size_coef=4.0, small_part=False, target_repl_coef=0.0, timestep=1.0, verbose=2)
'''

'''
data set preprocessing
'''

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

# train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
#                                          listfile=os.path.join(args.data, 'train_listfile.csv'),
#                                          period_length=48.0)

train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile2.csv'),
                                         period_length=48.0)

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize

# mimic3-benchmarks/mimic3models/in_hospital_mortality/ihm_ts1.0.input_str:previous.start_time:zero.normalizer
normalizer_state = '/home/tanmoy/Downloads/Trustworthiness/deepxplore/MIMIC-III/mimic3models/in_hospital_mortality/ihm_ts1.0.input_str:previous.start_time:zero.normalizer'

normalizer.load_params(normalizer_state)

f = args()
args_dict = dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl


test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)

ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                      return_names=True)

ret = utils.load_data(train_reader, discretizer, normalizer, args.small_part,return_names=True )
data = ret["data"][0]
labels = ret["data"][1]
names = ret["names"]

# Read data
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)



# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)


'''
This input tensor is very important for the multi network gradient calculation.
This input layer is connect to all of the LSTM networks (3)
'''
input_tensor = Input(shape=(None, 76), dtype='float32', name='X-Pias')


####################################################################################################
model1 = model_module.Network(input = input_tensor, **args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")

model1.final_name = args.prefix + model1.say_name() + suffix
print("==> model.final_name:", model1.final_name)


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

model1.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model1.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state1 != "":
    model1.load_weights(args.load_state1)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state1).group(1))
    print("Model1 Loaded!")


##########################################################################################


####################################################################################################
model2 = model_module.Network(input = input_tensor, **args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")

model2.final_name = args.prefix + model2.say_name() + suffix
print("==> model.final_name:", model2.final_name)


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

model2.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model2.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state2 != "":
    model2.load_weights(args.load_state2)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state2).group(1))
    print("Model2 Loaded!")
##########################################################################################


####################################################################################################
model3 = model_module.Network(input = input_tensor, **args_dict)
suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep,
                                   ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")

model3.final_name = args.prefix + model3.say_name() + suffix
print("==> model.final_name:", model3.final_name)


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

model3.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model3.summary()


# Load model weights
n_trained_chunks = 0
if args.load_state3 != "":
    model3.load_weights(args.load_state3)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state3).group(1))
    print("Model3 Loaded!")
##########################################################################################

'''
The model's predictions are working. 
'''
predictions1 = model1.predict(data, batch_size=args.batch_size, verbose=1)
predictions1 = np.array(predictions1)[:, 0]
metrics.print_metrics_binary(labels, predictions1)


predictions2 = model2.predict(data, batch_size=args.batch_size, verbose=1)
predictions2 = np.array(predictions2)[:, 0]
metrics.print_metrics_binary(labels, predictions2)


predictions3 = model3.predict(data, batch_size=args.batch_size, verbose=1)
predictions3 = np.array(predictions3)[:, 0]
metrics.print_metrics_binary(labels, predictions3)

'''
Neuron Coverage calculation
'''

model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)


'''
This is threshold for mortality and safe.
threshold = 0.22 is taken from Nature paper of Dr. Yao
'''
threshold = 0.22

p1 = model1.predict(data[0:1], batch_size=args.batch_size, verbose=1)
p1 = np.array(p1)[:, 0][0]
label1 = p1 > threshold

p2 = model2.predict(data[0:1], batch_size=args.batch_size, verbose=1)
p2 = np.array(p2)[:, 0][0]
label2 = p2 > threshold

p3 = model3.predict(data[0:1], batch_size=args.batch_size, verbose=1)
p3 = np.array(p3)[:, 0][0]
label3 = p3 > threshold



#if label1 == label2 == label3:
# if all label agrees
orig_label = label1
layer_name1, index1 = neuron_to_cover(model_layer_dict1)
layer_name2, index2 = neuron_to_cover(model_layer_dict2)
layer_name3, index3 = neuron_to_cover(model_layer_dict3)

loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])

# construct joint loss function
loss1 = K.mean(model1.get_layer('dense-Pias').output)
loss2 = K.mean(model2.get_layer('dense-Pias').output)
loss3 = K.mean(model3.get_layer('dense-Pias').output)

layer_output = (loss1 + loss2 + loss3)

# for adversarial image generation
final_loss = K.mean(layer_output)
grads = normalize(K.gradients(final_loss, input_tensor)[0])
iterate = K.function([input_tensor], [loss1, loss2, loss3, final_loss, grads])

'''
check the index of grad in the iterate function
now it's 4
'''


#################################################################################


'''
Diastolic Blood Pressure
2
'''

'''
discretizer_header[54]
'Respiratory rate'
'''
#### save original file in processed format
# print(data.shape)
# item_list = ['68251_episode2_timeseries.csv', '82293_episode1_timeseries.csv',
#              '3011_episode1_timeseries.csv', '52021_episode1_timeseries.csv',
#              '64371_episode1_timeseries.csv', '29861_episode3_timeseries.csv',
#              '3794_episode1_timeseries.csv']
#
# for item in item_list:
#     print(type(names))
#     idx = names.index(item)
#     print(idx, names[idx])
#     gen_img = data[idx:idx+1].copy()
#     gen_img_48_76 = normalizer.reverseSD(gen_img.reshape(48, 76).copy())
#     np.savetxt("Gradients2/Original/" + names[idx], gen_img_48_76, delimiter=",")

# print(args_dict['header'])
# for i in range(len(args_dict['header'])):
#     print(i, args_dict['header'][i])


'''
discretizer_header[]
#     DBP_col = 2
#     SBP_col = 55
#     resp_col = 54
#     glu_col = 49
#     #heart_col = 50
#     oxy_col = 53
#     temp_col = 56
'''



for idx in range(data.shape[0]):

    #idx = 1
    gen_img_org = data[idx:idx+1].copy()
    gen_img = data[idx:idx+1].copy()
    gen_img_temp = data[idx:idx+1].copy()
    print(names[idx])

    l11=l22=l33=False
    con = 0

    # DBP_col = 2
    SBP_col = 55
    # resp_col = 54
    # glu_col = 49
    # heart_col = 50
    # oxy_col = 53
    # temp_col = 56


    p11_list = []
    p22_list = []
    p33_list = []
    attribute_val_list = []


    all = []

    flag = ""
    flag_save = False
    gen_img_48_76 = False

    while l11 == l22 == l33:

        if l11 == True or l22 == True or l33 == True:
            print("True ... ")
            break

        grad_out = iterate([gen_img])[4]


        # gen_img[:, :, heart_col] += grad_out[:, :, heart_col] * 0.1
        # gen_img[:, :, oxy_col] += grad_out[:, :, oxy_col] * 0.01
        # gen_img[:, :, temp_col] += grad_out[:, :, temp_col] * 0.001
        # gen_img[:, :, glu_col] += grad_out[:, :, glu_col] * 5
        gen_img[:, :, SBP_col] += grad_out[:, :, SBP_col] * 0.2
        # gen_img[:, :, DBP_col] += grad_out[:, :, DBP_col] * 0.01
        # gen_img[:, :, resp_col] += grad_out[:, :, resp_col] * 0.01


        # apply negative clipping on original values (not normalized)
        gen_img_48_76 = normalizer.reverseSD(gen_img.reshape(48, 76).copy())
        gen_img_48_76 = np.clip(gen_img_48_76, 0, None)
        gen_img = normalizer.transform(gen_img_48_76)
        gen_img = gen_img.reshape(1, 48, 76)

        flag_save = True

        p11 = model1.predict(gen_img, batch_size=args.batch_size, verbose=1)
        p11 = np.array(p11)[:, 0][0]
        l11 = p11 > threshold

        p22 = model2.predict(gen_img, batch_size=args.batch_size, verbose=1)
        p22 = np.array(p22)[:, 0][0]
        l22 = p22 > threshold

        p33 = model3.predict(gen_img, batch_size=args.batch_size, verbose=1)
        p33 = np.array(p33)[:, 0][0]
        l33 = p33 > threshold

        if con >= 1000:
            flag = "X"
            break
        con += 1

        p11_list.append(p11)
        p22_list.append(p22)
        p33_list.append(p33)

        attribute_val = np.average(gen_img_48_76[ :, SBP_col])
        attribute_val_list.append(attribute_val)

        if attribute_val <= 0.0:
            flag = "X"
            break

        all.append([attribute_val, p11, p22, p33])

        gen_img = normalizer.transform(gen_img_48_76)
        gen_img = gen_img.reshape(1, 48, 76)

        print(con, attribute_val, p11, p22, p33, l11, l22, l33)

    if flag_save:
        print("-------------------------------------------------------------------------")
        np.savetxt("Gradients2/SBP/Modified_" + names[idx], gen_img_48_76, delimiter=",")

        d = {
            "Attribute value": attribute_val_list,
            "C1": p11_list,
            "C2": p22_list,
            "C3": p33_list
        }

        import pandas as pd
        dd = pd.DataFrame(d)
        dd.to_csv("Gradients2/SBP/Pred_" + names[idx] + flag + ".csv", index = False)

