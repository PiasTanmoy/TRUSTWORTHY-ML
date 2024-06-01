import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve, auc, \
    mean_absolute_error, precision_score, recall_score, classification_report, confusion_matrix, balanced_accuracy_score, precision_recall_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import matplotlib as mpl
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from lib.sampling import distant_method, replicated_oversampling, gamma_oversampling, double_prioritize, cross_group_DP, remove_race_features, remove_irr_features, combined_double_prioritize
from sklearn.utils import shuffle

from lib.weighted_loss import calculate_weight_vector, reweighted_binary_crossentropy

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


    def train(self, mlp_epochs):
        """ Training procedure. """
        mlp_batch_size = 20 # make batch size a parameter?

        if self.model_type in ['MLP', 'MLPEmb']:

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
                self.model.model.fit(self.train_x, self.train_y, epochs=mlp_epochs, batch_size=mlp_batch_size, verbose=2,
                                     validation_data=(self.valid_x, self.valid_y),
                                     callbacks=[model_checkpoint_callback])
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

        print(', '.join(results)) # print performance of the last epoch only
        return results # of last epoch
        #return self.evaluate(self.test_x, self.test_y, 'test')
    
    def test_single(self, mlp_epochs, dir):
        """ Testing evaluation wrapper. """
        print('Test results: ', end='')

        useful_columns = ['Race recode Y 1', 'Race recode Y 2', 'Race recode Y 4',\
        'Origin Recode NHIA 1',
        'Age at diagnosis continuous', 'Sex 1']

        self.test_output_df = pd.read_csv('processed_data_TS_2/test_output_df.csv')
        self.test_output_df = self.test_output_df[useful_columns]

        self.test_x = np.load('processed_data_TS_2/X_test_normalized.npy')
        self.test_y = np.load('processed_data_TS_2/y_test_normalized.npy')

        if self.model_type == 'LogR':
            return self.evaluate(self.test_x, self.test_y, 'test', self.model)

        # otherwise it's mlp
        results = ''
        # evaluate on validation data for each model saved on each epoch
        current_model = tf.keras.models.load_model(dir + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=mlp_epochs))
        #current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i), custom_objects={'loss': reweighted_binary_crossentropy(self.weight_vector)})
        results = self.evaluate(self.test_x, self.test_y, 'test', mlp_epochs, current_model)

        print(', '.join(results)) # print performance of the last epoch only
        return results # of last epoch
        #return self.evaluate(self.test_x, self.test_y, 'test')

    def validate_single(self, mlp_epochs, dir, valid_df_path = 'Tests/num_lymph_exam_seed_0_range_0_25'):
        """ Validation evaluation wrapper. """
        print('Validation results: ', end='')

        useful_columns = ['Race recode Y 1', 'Race recode Y 2', 'Race recode Y 4',\
        'Origin Recode NHIA 1',
        'Age at diagnosis continuous', 'Sex 1']

        self.valid_output_df = pd.read_csv(valid_df_path + '_x.csv')
        self.valid_output_df = self.valid_output_df[useful_columns]

        # self.valid_output_df = pd.read_csv('processed_data_TS_2/valid_output_df.csv')
        # self.valid_output_df = self.valid_output_df[useful_columns]

        self.valid_x = np.load(valid_df_path + '_x.npy')
        self.valid_y = np.load(valid_df_path + '_y.npy')

        # self.valid_x = np.load('processed_data_TS_2/X_valid_normalized.npy')
        # self.valid_y = np.load('processed_data_TS_2/y_valid_normalized.npy')

        if self.model_type == 'LogR':
            return self.evaluate(self.valid_x, self.valid_y, 'valid', self.model)

        # otherwise it's mlp
        results = ''
        # evaluate on validation data for each model saved on each epoch
        
        current_model = tf.keras.models.load_model(dir + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=mlp_epochs))

        #tf.keras.utils.plot_model(current_model, to_file='model.png')
        print(current_model.summary())

        #current_model = tf.keras.models.load_model(self.output_directory + 'checkpoint/cp-{epoch:04d}.ckpt'.format(epoch=i), custom_objects={'loss': reweighted_binary_crossentropy(self.weight_vector)})
        results = self.evaluate(self.valid_x, self.valid_y, 'valid', mlp_epochs, current_model)


        print(', '.join(results)) # print performance of the last epoch only
        return results # of last epoch


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
            predict_y = scores_y.round()

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
        m['f1 c0'] = 2 * precision0 * recall0 / (precision0 + recall0)

        measurements.append('precision class 0 = ' + str(precision0))
        m['precision c0'] = precision0

        measurements.append('recall class 0 = ' + str(recall0))
        m['recall c0'] = recall0

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
            measurements.append('AUC_PR_C1 = ' + str(auc(curve_recall, curve_precision)))
            m['AUC_PR_C1'] = auc(curve_recall, curve_precision)
        except:
            measurements.append('AUC_PR_C1 = 0')
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
            measurements.append('0')
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

        #print(', '.join(measurements))
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
