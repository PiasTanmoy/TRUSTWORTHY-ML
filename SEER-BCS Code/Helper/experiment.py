import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve, auc, \
    mean_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import matplotlib as mpl
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from lib.sampling import distant_method, replicated_oversampling, gamma_oversampling, double_prioritize, cross_group_DP, remove_race_features, remove_irr_features, combined_double_prioritize
from sklearn.utils import shuffle

from lib.weighted_loss import calculate_weight_vector, reweighted_binary_crossentropy

import csv

mpl.use('Agg')
import matplotlib.pyplot as plt


class Experiment:
    """ Class for main functionality. """

    def __init__(self, data, model, task, sampling, dpunits, subgroup, comb_method, reweight, dpreweight, valid_ratio, test_ratio, model_type, encodings, encode_categorical_inputs,
                 plot_results, output_directory):
        """ Initialize main functionality and split data according to given ratios. """
        self.model = model
        self.model_type = model_type
        self.task = task
        #self.sampling = sampling
        self.reweight = reweight
        self.dpreweight = dpreweight
        self.subgroup = subgroup

        self.plot_results = plot_results
        self.output_directory = output_directory

        self.input_columns = list(data.frame)


        data.frame.to_csv(self.output_directory + "raw_dataset.csv", index=False)

        ################################
        # write cols to a file
        
        with open(self.output_directory + 'input_columns.txt', 'w') as f:
            for i in self.input_columns:
                f.write(i + '\n')
        
        ################################

        x, y = [], []

        n = int(task[-2:])
        self.input_columns.remove("Survived cancer for " + str(n) + " months")


        # data.frame is the original dataset

        class_1 = data.frame[data.frame["Survived cancer for " + str(n) + " months"] == 1]
        count_class_1 = class_1.shape[0]
        class_0 = data.frame[data.frame["Survived cancer for " + str(n) + " months"] == 0]
        count_class_0 = class_0.shape[0]

        # inputs
        x = data.frame[self.input_columns].values.astype(np.float32)
        # labels
        y = data.frame["Survived cancer for " + str(n) + " months"].values.reshape(
            (data.frame["Survived cancer for " + str(n) + " months"].values.shape[0],)).astype(np.int32)


        self.total_1 = np.count_nonzero(y)
        self.total_0 = y.shape[0] - self.total_1

        # Fix random state to obtain same sets across experiments
        self.train_x, self.test_x, self.train_y, self.test_y \
            = train_test_split(x, y, test_size=valid_ratio + test_ratio, shuffle=True, random_state=73)

        # Fix random state to obtain same sets across experiments
        self.valid_x, self.test_x, self.valid_y, self.test_y \
            = train_test_split(self.test_x, self.test_y, test_size=(test_ratio / (valid_ratio + test_ratio)),
                               shuffle=True, random_state=63)


        #############################
        # save train / test files for additional experiments
        np.save(self.output_directory + '/X_train.npy', self.train_x)
        np.save(self.output_directory + '/y_train.npy', self.train_y)
        np.save(self.output_directory + '/X_valid.npy', self.valid_x)
        np.save(self.output_directory + '/y_valid.npy', self.valid_y)
        np.save(self.output_directory + '/X_test.npy', self.test_x)
        np.save(self.output_directory + '/y_test.npy', self.test_y)

        #self.train_x.to_csv(self.output_directory + 'train_x.csv')
        #############################

        ###########################################################
        #
        # added! start sampling!
        #
        ###########################################################
        if sampling == 'Under':
            # undersampling
            rus = RandomUnderSampler(random_state=42)
            self.train_x, self.train_y = rus.fit_resample(self.train_x, self.train_y)
        elif sampling == 'NearMiss1':
            # undersample near miss 1
            undersample = NearMiss(version=1, n_neighbors=3)
            self.train_x, self.train_y = undersample.fit_resample(self.train_x, self.train_y)
            print(self.train_x.shape)
        elif sampling == 'NearMiss3':
            # undersample near miss 3
            undersample = NearMiss(version=3, n_neighbors=3)
            self.train_x, self.train_y = undersample.fit_resample(self.train_x, self.train_y)
            print(self.train_x.shape)
        elif sampling == 'Distant':
            # distant method
            self.train_x, self.train_y = distant_method(self.train_x, self.train_y)
        elif sampling == 'Over':
            # replicated oversampling
            self.train_x, self.train_y = replicated_oversampling(self.train_x, self.train_y)
            """
            # random oversampling
            ros = RandomOverSampler(random_state=42)
            self.train_x, self.train_y = ros.fit_resample(self.train_x, self.train_y)
            """
        elif sampling == 'DP':
            self.train_x, self.train_y = double_prioritize(self.train_x, self.train_y, dpunits, subgroup, self.input_columns)
        elif sampling == 'SMOTE':
            # smote oversampling
            self.train_x, self.train_y = SMOTE().fit_resample(self.train_x, self.train_y)
        elif sampling == 'ADASYN':
            # ADASYN oversampling
            self.train_x, self.train_y = ADASYN().fit_resample(self.train_x, self.train_y)
        elif sampling == 'Gamma':
            # gamma
            self.train_x, self.train_y = gamma_oversampling(self.train_x, self.train_y)
            print("gamma", self.train_x.shape, self.train_y.shape)
        elif sampling == 'CombinedDP':
            self.train_x, self.train_y = combined_double_prioritize(self.train_x, self.train_y, dpunits, subgroup, self.input_columns, comb_method)
        elif sampling == 'None':
            # nothing...
            print("No sampling")
        else:
            logging.error('Invalid sampling method.')
            exit(-1)



        ####################################################
        # end sampling
        ###################################################

        if self.dpreweight != 1:
            self.weight_vector = calculate_weight_vector(self.train_x, self.train_y, self.input_columns, self.subgroup, self.dpreweight)


        ######################################################
        # added for outputing training and validating set data count
        ######################################################
        # save train / test files for additional experiments
        #np.save(self.output_directory + '/X_train.npy', self.train_x)
        #np.save(self.output_directory + '/y_train.npy', self.train_y)

        #np.save(self.output_directory + '/X_test.npy', self.test_x)
        #np.save(self.output_directory + '/y_test.npy', self.test_y)

        #self.train_x = np.load('data/lung/distant/X_train.npy')
        #self.train_y = np.load('data/lung/distant/y_train.npy')

        # shuffling to check
        #self.train_x, self.train_y = shuffle(self.train_x, self.train_y, random_state=11)

        self.train_1 = np.count_nonzero(self.train_y)
        self.train_0 = self.train_y.shape[0] - self.train_1
        self.valid_1 = np.count_nonzero(self.valid_y)
        self.valid_0 = self.valid_y.shape[0] - self.valid_1

        # end data count

        #self.train_x = remove_race_features(self.train_x, self.input_columns)
        #self.train_x = remove_irr_features(self.train_x, self.input_columns)

        ###########################################################
        # added for output test data after shuffling
        # only for one hot encoding
        ###########################################################
        # save before normalizing age!!! T-T

        self.train_output_df = pd.DataFrame(self.train_x, columns=self.input_columns)
        self.valid_output_df = pd.DataFrame(self.valid_x, columns=self.input_columns)
        self.test_output_df = pd.DataFrame(self.test_x, columns=self.input_columns)

        self.train_output_df.to_csv(self.output_directory + "/train_output_df.csv", index=False)
        self.valid_output_df.to_csv(self.output_directory + "/valid_output_df.csv", index=False)
        self.test_output_df.to_csv(self.output_directory + "/test_output_df.csv", index=False)



        # one hot encoding
        useful_columns = ['Race recode Y 1', 'Race recode Y 2', 'Race recode Y 4',\
        'Origin Recode NHIA 1',
        'Age at diagnosis continuous', 'Sex 1']

        """
        # for non one-hot
        # subgroup
        useful_columns = ['Race recode Y', \
        'Origin Recode NHIA',
        'Age at diagnosis', 'Sex']
        """

        self.valid_output_df = self.valid_output_df[useful_columns]
        self.test_output_df = self.test_output_df[useful_columns]
        #self.train_output_df = self.train_output_df[useful_columns]
        #self.train_output_df['true y'] = self.train_y

        ###########################################################
        # end added code
        ###########################################################

        # Unique hash for set splits to ensure that same sets are used throughout experiments
        self.set_split_hash = hash(np.sum(self.train_x)) + hash(np.sum(self.train_y)) + hash(np.sum(self.valid_x)) \
                              + hash(np.sum(self.valid_y)) + hash(np.sum(self.test_x)) + hash(np.sum(self.test_y))

        # Normalize data -> Correct: Standardize 
        if encode_categorical_inputs:
            # Only normalize continuous fields
            continuous_columns = [idc for idc, c in enumerate(list(data.frame)) if c.endswith(' continuous')]
            scaler = preprocessing.StandardScaler().fit(self.train_x[:, continuous_columns])
            self.train_x[:, continuous_columns] = scaler.transform(self.train_x[:, continuous_columns])
            self.valid_x[:, continuous_columns] = scaler.transform(self.valid_x[:, continuous_columns])
            self.test_x[:, continuous_columns] = scaler.transform(self.test_x[:, continuous_columns])
        else:
            # Normalize all fields
            scaler = preprocessing.StandardScaler().fit(self.train_x)
            self.train_x = scaler.transform(self.train_x)
            self.valid_x = scaler.transform(self.valid_x)
            self.test_x = scaler.transform(self.test_x)

        # save normalized data
        np.save(self.output_directory + '/X_train_normalized.npy', self.train_x)
        np.save(self.output_directory + '/y_train_normalized.npy', self.train_y)
        np.save(self.output_directory + '/X_valid_normalized.npy', self.valid_x)
        np.save(self.output_directory + '/y_valid_normalized.npy', self.valid_y)
        np.save(self.output_directory + '/X_test_normalized.npy', self.test_x)
        np.save(self.output_directory + '/y_test_normalized.npy', self.test_y)

        # self.train_x.to_csv(self.output_directory + '/X_train_normalized.csv', index = False)
        # self.train_y.to_csv(self.output_directory + '/y_train_normalized.csv', index = False)
        # self.test_x.to_csv(self.output_directory + '/X_test_normalized.csv', index = False)
        # self.test_y.to_csv(self.output_directory + '/y_test_normalized.csv', index = False)
        # self.valid_x.to_csv(self.output_directory + '/X_valid_normalized.csv', index = False)
        # self.valid_y.to_csv(self.output_directory + '/y_valid_normalized.csv', index = False)


        logging.info("Data:  " + str(data.frame.shape) + " -> x:" + str(x.shape) + ", y:" + str(y.shape))
        logging.info("Train: x:{0}, y:{1}".format(str(self.train_x.shape), str(self.train_y.shape)))
        logging.info("Valid: x:{0}, y:{1}".format(str(self.valid_x.shape), str(self.valid_y.shape)))
        logging.info("Test:  x:{0}, y:{1}".format(str(self.test_x.shape), str(self.test_y.shape)))

        # Split up the data into distinct inputs for each embedding
        if model_type == 'MLPEmb':
            print("")
            logging.info("Embed input data.")
            encoding_splits = np.cumsum(list(encodings.values()))
            self.train_x = np.hsplit(self.train_x, encoding_splits)[:-1]
            self.train_x = [np.expand_dims(x, axis=2) for x in self.train_x]
            self.valid_x = np.hsplit(self.valid_x, encoding_splits)[:-1]
            self.valid_x = [np.expand_dims(x, axis=2) for x in self.valid_x]
            self.test_x = np.hsplit(self.test_x, encoding_splits)[:-1]
            self.test_x = [np.expand_dims(x, axis=2) for x in self.test_x]

    def train(self, mlp_epochs):
        """ Training procedure. """
        mlp_batch_size = 20 # make batch size a parameter?

        if self.model_type in ['MLP', 'MLPEmb']:


            # compile with class weight before training
            #weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.train_y), y=self.train_y)
            #print('class_weight fit', weights)
            #self.model.model = mlp_compile_with_weights(self.model.model, weights=weights)

            ### end added

            ####
            # use class_weight in fit instead of modify loss function
            ####
            #weight_dict = dict(enumerate(weights))
            ###
            
            ###############################################################
            # I have added save_validation_loss_callback to save the validaiton loss while training which is going to be used to select the best epoch
            class SaveValidationLossCallback(tf.keras.callbacks.Callback):
                def __init__(self, csv_file):
                    super().__init__()
                    self.csv_file = csv_file

                def on_epoch_end(self, epoch, logs=None):
                    val_loss = logs.get('val_loss')
                    with open(self.csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, val_loss])

            # Create a CSV file to save the validation loss
            csv_file = self.output_directory + 'validation_loss.csv'

            # Create the custom callback to save validation loss to CSV
            save_validation_loss_callback = SaveValidationLossCallback(csv_file)
            ######################################################################



            # save checkpoint after each epoch
            checkpoint_filepath = self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt' # + {epoch}
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_loss',
                save_freq='epoch',
                verbose=1
            )

            if self.dpreweight != 1:
                #self.weight_vector = calculate_weight_vector(self.train_x, self.train_y, self.input_columns, self.subgroup, self.dpreweight)
                #self.model.model = mlp_model_reweight(weight_vector)
                """
                loss = reweighted_binary_crossentropy(self.weight_vector)
                self.model.model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
                print('reweight', self.subgroup, self.dpreweight)

                self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                     validation_data=(self.valid_x, self.valid_y),
                                     callbacks=[model_checkpoint_callback])
                                     """
                print('reweight', self.subgroup, self.dpreweight)

                print('weight_vector sum:', sum(self.weight_vector))

                self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                      validation_data=(self.valid_x, self.valid_y),
                                      callbacks=[model_checkpoint_callback],
                                      sample_weight=self.weight_vector)

            elif self.reweight:
                weights = class_weight.compute_class_weight('balanced', classes=np.unique(self.train_y), y=self.train_y)
                weight_dict = dict(enumerate(weights))
                print('reweight classes')
                print(weight_dict)

                self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                     validation_data=(self.valid_x, self.valid_y),
                                     callbacks=[model_checkpoint_callback],
                                     class_weight=weight_dict)
            else:
                #I have added save_validation_loss_callback to save the validaiton loss while training which is going to be used to select the best epoch
                
                self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                     validation_data=(self.valid_x, self.valid_y),
                                     callbacks=[model_checkpoint_callback, save_validation_loss_callback])

                # self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                #                      validation_data=(self.valid_x, self.valid_y),
                #                      callbacks=[model_checkpoint_callback])
            ######################################
            # save this model
            # self.model.model.save(self.output_directory + 'breast_whole_mlp.h5')
            ######################################
            """
            # extra training on black
            train_x_b = np.load('data/X_train_b.npy')
            train_y_b = np.load('data/y_train_b.npy')
            train_x_b = np.repeat(train_x_b, 10, axis=0)
            train_y_b = np.repeat(train_y_b, 10, axis=0)
            train_x_b, train_y_b = shuffle(train_x_b, train_y_b, random_state=0)
            self.model.model.fit(train_x_b, train_y_b, epochs=1)
            """
        elif self.model_type in ['LogR', 'NAIVE']:
            self.model.model.fit(self.train_x, self.train_y)

    def validate(self, mlp_epochs):
        """ Validation evaluation wrapper. """
        print('Validation results: ', end='')

        if self.model_type == 'LogR':
            return self.evaluate(self.valid_x, self.valid_y, 'valid', self.model)

        # otherwise it's mlp
        results = ''
        # evaluate on validation data for each model saved on each epoch
        for i in range(1, mlp_epochs+1):
            current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i))
            #current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i), custom_objects={'loss': reweighted_binary_crossentropy(self.weight_vector)})
            results = self.evaluate(self.valid_x, self.valid_y, 'valid', i, current_model)

        print(', '.join(results)) # print performance of the last epoch only
        print(results)
        return results # of last epoch

    def test(self, mlp_epochs):
        """ Testing evaluation wrapper. """
        print('Test results: ', end='')

        if self.model_type == 'LogR':
            return self.evaluate(self.test_x, self.test_y, 'test', self.model)

        # otherwise it's mlp
        results = ''
        # evaluate on validation data for each model saved on each epoch
        for i in range(1, mlp_epochs+1):
            current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i))
            #current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i), custom_objects={'loss': reweighted_binary_crossentropy(self.weight_vector)})
            results = self.evaluate(self.test_x, self.test_y, 'test', i, current_model)
        
        np.save(self.output_directory + '/test_x2.npy', self.test_x)
        np.save(self.output_directory + '/test_y2.npy', self.test_y)

        print(', '.join(results)) # print performance of the last epoch only
        print(results)
        return results # of last epoch
        #return self.evaluate(self.test_x, self.test_y, 'test')

    def evaluate(self, eval_x, eval_y, eval_type, current_epoch, model):
        """ Generic evaluation method. """
        ######################################
        # save this model
        #self.model.model.save(self.output_directory + 'breast_whole_mlp')
        ######################################

        if self.task in ['survival12', 'survival60'] and (self.model_type == 'SVM' or self.model_type == 'LogR'):
            # Use decision function value as score
            # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
            scores_y = model.decision_function(eval_x)
        else:
            scores_y = model.predict(eval_x)

        measurements = []
        m = {}

        # Classification
        if self.model_type == 'LogR':
            predict_y = model.predict(eval_x)
        else:
            predict_y = scores_y.round() # this maps <0.5 to 0 and >= 0.5 to 1 (threshold 0.5)

        ###########################################################
        # added for output test data after shuffling
        # only for one hot encoding
        ###########################################################
        # append true and predict labels and save to file
        # for subgroup training

        if eval_type == 'valid':
            self.valid_output_df['true y'] = eval_y
            self.valid_output_df['predict y'] = predict_y
            self.valid_output_df['score y'] = scores_y
            if self.model_type == 'LogR':
                self.valid_output_df['predict y proba'] = self.model.model.predict_proba(eval_x)[:,1]
                self.valid_output_df.to_csv(self.output_directory + 'valid_results_data_frame.csv')
            else:
                self.valid_output_df.to_csv(self.output_directory + 'valid_results_data_frame_' + str(current_epoch) + '_epo.csv')
        else: # test
            self.test_output_df['true y'] = eval_y
            self.test_output_df['predict y'] = predict_y
            self.test_output_df['score y'] = scores_y
            if self.model_type == 'LogR':
                self.test_output_df['predict y proba'] = self.model.model.predict_proba(eval_x)[:,1]
                self.test_output_df.to_csv(self.output_directory + 'test_results_data_frame.csv')
            else:
                self.test_output_df.to_csv(self.output_directory + 'test_results_data_frame_' + str(current_epoch) + '_epo.csv')

        #self.train_output_df.to_csv(self.output_directory + 'train_data_frame.csv')

        ###########################################################
        # end added code
        ###########################################################

		###
		# for output logits
		###
        #predict_y = (predict_y > 0).astype(np.int)
		###

        measurements.append('auc = ' + str(roc_auc_score(eval_y, scores_y)))
        m['auc'] = roc_auc_score(eval_y, scores_y)

        measurements.append('f1 = ' + str(f1_score(eval_y, predict_y)))
        m['f1'] = f1_score(eval_y, predict_y)

        measurements.append('acc = ' + str(accuracy_score(eval_y, predict_y)))
        m['acc'] = accuracy_score(eval_y, predict_y)

        ################################################
        #---added------------------------------------
        measurements.append('balanced_accuracy = ' + str(balanced_accuracy_score(eval_y, predict_y)))
        m['balanced_accuracy'] = balanced_accuracy_score(eval_y, predict_y)

        measurements.append('precision class 1 = ' + str(precision_score(eval_y, predict_y)))
        m['precision class 1'] = precision_score(eval_y, predict_y)

        measurements.append('recall class 1 = ' + str(recall_score(eval_y, predict_y)))
        m['recall class 1'] = recall_score(eval_y, predict_y)

        tn, fp, fn, tp = confusion_matrix(eval_y, predict_y).ravel()
        recall0 = tn / (tn + fp)
        precision0 = tn / (tn + fn)

        measurements.append('f1 class 0 = ' + str(2 * precision0 * recall0 / (precision0 + recall0)))
        m['f1 class 0'] = 2 * precision0 * recall0 / (precision0 + recall0)

        measurements.append('precision class 0 = ' + str(precision0))
        m['precision class 0'] = precision0

        measurements.append('recall class 0 = ' + str(recall0))
        m['recall class 0'] = recall0

        measurements.append('FPR = ' + str(fp / (fp + tn)))
        m['FPR'] = fp / (fp + tn)

        measurements.append('FNR = ' + str(fn / (fn + tp)))
        m['FNR'] = fn / (fn + tp)
        
        measurements.append('TPR = ' + str(tp / (fn + tp)))
        m['TPR'] = tp / (fn + tp)

        measurements.append('TNR = ' + str(tn / (tn + fp)))
        m['TNR'] = tn / (tn + fp)

        try:
            curve_precision, curve_recall, _ = precision_recall_curve(eval_y, scores_y)
            measurements.append('AUC_PR_C1 = ' + str( auc(curve_recall, curve_precision)))
            m['AUC_PR_C1'] = auc(curve_recall, curve_precision)
        except:
            measurements.append('AUC_PR_C1 = ' + '0')
            m['AUC_PR_C1'] = 0

        true_y_filp = ((eval_y) == 0).astype(np.int)
        if (scores_y < 0).sum() > 0: # negative value, logr decision scores
            score_y_filp = np.negative(scores_y)
        else: # mlp scores
            score_y_filp = (1-scores_y)
        
        try:
            curve_precision0, curve_recall0, _ = precision_recall_curve(true_y_filp, score_y_filp)
            measurements.append('AUC_PR_C0 = ' + str(auc(curve_recall0, curve_precision0)))
            m['AUC_PR_C0'] = auc(curve_recall0, curve_precision0)
        except:
            measurements.append('AUC_PR_C0 = ' + '0')
            m['AUC_PR_C0'] = 0

        m['tp'] = tp
        m['tn'] = tn
        m['fp'] = fp
        m['fn'] = fn

        MCC = matthews_corrcoef(eval_y, predict_y)
        m['MCC'] = MCC
        measurements.append('MCC = ' + str(MCC))

        minpse = np.max([min(x, y) for (x, y) in zip(curve_precision0, curve_recall0)])
        m['minpse'] = minpse
        measurements.append('minpse = ' + str(minpse))


        m_df = pd.DataFrame(m.items(), columns=['Metrics', 'Value'])


        if eval_type == 'valid':
            m_df.to_csv(self.output_directory + 'valid_measurement_data_frame_' + str(current_epoch) + '_epo.csv')
        else:
            m_df.to_csv(self.output_directory + 'test_measurement_data_frame_' + str(current_epoch) + '_epo.csv')
        ################################################
        

        if self.plot_results:
            fig = plt.figure(dpi=200)
            self.plot_roc(eval_y, scores_y, plt)
            fig.savefig(self.output_directory + 'roc.png')

        print(', '.join(measurements))
        return measurements

    def importance(self, encodings):
        """ Method that analyzes the importance of input variables for LogR/LinR and MLP* models. """
        importance = []

        if self.model_type in ['LogR'] and self.task in ['survival12', 'survival60']:

            # Use coefficients
            abs_coefficients = np.abs(self.model.model.coef_[0])
            i = 0
            for column, encoding_size in encodings.items():
                coefficient_sum = 0.
                for idx in range(i, i + encoding_size):
                    coefficient_sum += abs_coefficients[idx]
                i += encoding_size
                importance.append(coefficient_sum / encoding_size) # sum or average???
            importance = np.array(importance)
            """
            # Ablate attributes and measure effect on output
            scores_y = self.model.model.predict_proba(self.test_x)[:,1]
            i = 0
            for column, encoding_size in encodings.items():
                ablated_test_x = self.test_x.copy()

                ablated_test_x[:, i:(i + encoding_size)] = 0
                i += encoding_size

                ablated_scores_y = self.model.model.predict_proba(ablated_test_x)[:,1]
                ablated_diff = np.sum(np.abs(scores_y - ablated_scores_y))
                importance.append(ablated_diff)
            """

        if self.model_type in ['MLP', 'MLPEmb'] and self.task in ['survival12', 'survival60']:
            # Ablate attributes and measure effect on output
            scores_y = self.model.model.predict(self.test_x)
            i = 0
            for column, encoding_size in encodings.items():
                ablated_test_x = self.test_x.copy()
                if self.model_type == 'MLP':
                    ablated_test_x[:, i:(i + encoding_size)] = 0
                    i += encoding_size
                elif self.model_type == 'MLPEmb':
                    ablated_test_x[i][:, :] = 0
                    i += 1
                ablated_scores_y = self.model.model.predict(ablated_test_x)
                ablated_diff = np.sum(np.abs(scores_y - ablated_scores_y))
                importance.append(ablated_diff)

            importance = np.array(importance)

        # Normalize importance
        importance = importance / np.sum(importance)
        result = dict(zip([column for column, encoding_size in encodings.items()], importance))

        # Sort results
        result = [(k, result[k]) for k in sorted(result, key=result.get, reverse=True)]
        return result

    @staticmethod
    def plot_scatter(labels, predictions, plot):
        """ Method to plot a scatter plot of predictions vs labels """
        plot.scatter(labels, predictions)
        plot.xlabel('Labels')
        plot.ylabel('Predictions')
        plot.title('Labels vs predictions')

    @staticmethod
    def plot_roc(labels, scores, plot):
        """ Method to plot ROC curve from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plot.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plot.xlim([0.0, 1.0])
        plot.ylim([0.0, 1.05])
        plot.xlabel('False Positive Rate')
        plot.ylabel('True Positive Rate')
        plot.title('Receiver operating characteristic example')
        plot.legend(loc="lower right")
