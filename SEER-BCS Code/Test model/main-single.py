import logging
import datetime
import os
import random as rn
import numpy as np
from keras import backend as k
import tensorflow as tf
import pandas as pd

from lib import pipelines
from lib.data import Data
from lib.model import Model
from lib.options import parseargs
from lib.experiment_single import Experiment


class Data_Frame:
    def __init__(self, df):
        self.frame = df




def main():
    """ The main routine. """

    # Fix random seeds for reproducibility - these are themselves generated from random.org
    # From https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(91)
    rn.seed(95)
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #tf.set_random_seed(47)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #k.set_session(sess)

    # Enable simple logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command line arguments
    args = parseargs()

    # Create run folder
    output_directory = create_output_folder(args.output)

    # Write arguments to file
    with open(output_directory + 'arguments.txt', 'a') as arguments_file:
        for arg in vars(args):
            arguments_file.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')

    ##############
    # Prepare data
    # print('')
    # data = Data(incidences_file=args.incidences, specifications_file=args.specifications, plot_data=args.plotData,
    #             output_directory=output_directory)
    # data.state(message='Raw data')

    # data.filter_cases(cases_file=args.cases)
    # data.state(message='Filtered SEER*Stat cases from ASCII')

    # # Determine inputs, filter, and pre process them
    # data.apply_data_pipeline(pipelines.data_pipeline_full, args.oneHotEncoding)
    # data.state(message='Remove irrelevant, combined, post-diagnosis, and treatment attributes')

    # data.create_target(args.task)
    # data.state(message='Create target label indicating cancer survival for ' + args.task)

    # encodings = data.finalize()
    # data.state(message='Remove inputs with constant values')

    # Tanmoy
    # seer_data/Tests/CS_Tumor_Size_valid_seed_0_Size_0_40.csv
    # df = pd.read_csv('processed_data_TS/raw_dataset.csv')
    # df = pd.read_csv('Tests/CS_Tumor_Size_valid_seed_0_Size_0_40.csv') # This is not CS tumor but num of lymph node
    
    # this doesn't matter
    df = pd.read_csv('processed_data_TS/raw_dataset.csv')
    data = Data_Frame(df)

    ###############
    # Prepare model
    # model = Model(model_type=args.model, task=args.task, input_dim=(len(data.frame.columns) - 1),
    #               encodings=encodings, mlp_layers=args.mlpLayers, mlp_width=args.mlpWidth,
    #               mlp_dropout=args.mlpDropout, mlp_emb_neurons=args.mlpEmbNeurons, logr_c=args.logrC)
    

    # if args.plotData:
    #     model.plot_model(output_directory)

    ################
    # Carry out task
    # experiment = Experiment(model=model, data=data, task=args.task, sampling=args.sampling, dpunits=args.DPUnits, subgroup=args.subgroup, comb_method=args.combining, reweight=args.reweight, dpreweight=args.DPreweight,
    #                         valid_ratio=0.1, test_ratio=0.1,
    #                         model_type=args.model, encodings=encodings, encode_categorical_inputs=args.oneHotEncoding,
    #                         plot_results=args.plotResults, output_directory=output_directory)
    
    experiment = Experiment(model='', data=data, task=args.task, sampling=args.sampling, dpunits=args.DPUnits, subgroup=args.subgroup, comb_method=args.combining, reweight=args.reweight, dpreweight=args.DPreweight,
                            valid_ratio=0.1, test_ratio=0.1,
                            model_type=args.model, encodings='', encode_categorical_inputs=args.oneHotEncoding,
                            plot_results=args.plotResults, output_directory=output_directory)

    '''
    train function is disabled for testing purposes
    '''
    # experiment.train(mlp_epochs=args.mlpEpochs)
    # seer_data/Test2/CS_Tumor_Size/CS_Turmor_Size_valid_norm_range_0_42_seed_0_x.csv
    # seer_data/Test2/Num_lymph/num_lymph_valid_norm_range_0_10_seed_0_x.csv
    # seer_data/Test2/Pos_lymph/pos_lymph_valid_norm_range_0_25_seed_0_x.csv
    
    # seer_data/Test2/CS_Tumor_Num_lymph/CS_Tumor_Num_lymph_valid_norm_range_seed_0_x.csv
    # seer_data/Test2/CS_Tumor_Pos_lymph/CS_Tumor_Pos_lymph_valid_norm_range_seed_0_x.csv
    # seer_data/Test2/Pos_lymph_Num_lymph/Pos_lymph_Num_lymph_valid_norm_range_seed_0_x.csv
    # seer_data/Test2/CS_Tumor_Pos_lymph2/CS_Tumor_Pos_lymph_valid_norm_range_seed_0_x.csv
    # seer_data/Test2/CS_Tumor_Pos_lymph3/CS_Tumor_Pos_lymph_valid_norm_range_seed_0_x.csv
    # seer_data/Test2/Multi/Multi_valid_norm_x.csv

    # seer_data/Test2/Grades/Grade 1/Grades_valid_norm_x.csv
    # seer_data/Test2/Grades/Class_1/Grade 1/Grades_valid_norm_x.csv

    # seer_data/Test2/Original_Test_Set/test_norm_x.csv

    # seer_data/Base_MLP_models2/2024-4-2_17-54-44_experiment-1
    # seer_data/Base_MLP_models2/2024-4-2_17-55-50_experiment-5

    '''
    mlp_epochs=args.mlpEpochs -> set the  --mlpEpochs in terminal command
    dir: set the model (folder) under test -> mlpEpochs will automatically load the specific epoch model instance
    valid_df_path: test set directory with prefix of the model name (exclusing _x.csv)
    '''
    results_validate = experiment.validate_single(mlp_epochs=args.mlpEpochs, 
                                                  dir='Base_MLP_models2/2024-4-2_17-54-44_experiment-1/',
                                                  valid_df_path = 'Test2/Original_Test_Set/test_norm')
    
    # Write validation results to file
    with open(output_directory + 'results_validate.txt', 'a') as results_file:
        for res in results_validate:
            results_file.write(res + '\n')

    # Only test final model, do not use for tuning
    if args.test:
        results_test = experiment.test_single(mlp_epochs=args.mlpEpochs, 
                                              dir='/home/tanmoysarkar/Trustworthiness/SEER/seer_data/2023-11-28_0-51-18_experiment-0 (selected)/')
        # Write validation results to file
        with open(output_directory + 'results_test.txt', 'a') as results_file:
            for res in results_test:
                results_file.write(res + '\n')

    ###################
    # Input importance
    # if args.importance:
    #     importance = experiment.importance(encodings=encodings)
    #     # Write importance results to file
    #     with open(output_directory + 'results_importance.txt', 'a') as results_file:
    #         for (column, rel) in importance:
    #             results_file.write(column + '=' + str(rel) + '\n')

    # ###########################################
    # # added------------------------------------
    # with open(output_directory + 'data_counts.txt', 'a') as results_file:
    #     results_file.write("total\n")
    #     #results_file.write(data.counts.to_string() + '\n')
    #     results_file.write('1 ' + str(experiment.total_1) + '\n')
    #     results_file.write('0 ' + str(experiment.total_0) + '\n')
    #     results_file.write("training\n")
    #     results_file.write('1 ' + str(experiment.train_1) + '\n')
    #     results_file.write('0 ' + str(experiment.train_0) + '\n')
    #     results_file.write("validating\n")
    #     results_file.write('1 ' + str(experiment.valid_1) + '\n')
    #     results_file.write('0 ' + str(experiment.valid_0) + '\n')
    ###########################################


def create_output_folder(output):
    """ Create a unique output folder. """
    now = datetime.datetime.now()
    run_folder_name = ('{0}-{1}-{2}_{3}-{4}-{5}_experiment'.format(str(now.year), str(now.month), str(now.day),
                                                                   str(now.hour), str(now.minute), str(now.second)))
    output_directory = output + ('' if output[-1] == '/' else '/') + run_folder_name

    i = 0
    while os.path.exists(output_directory + '-%s' % i):
        i += 1

    try:
        os.makedirs(output_directory + '-' + str(i))
    except FileExistsError:
        # Race condition: other process created directory - just try again (recursion should not go to deep here)
        return create_output_folder(output)

    return output_directory + '-' + str(i) + '/'


if __name__ == "__main__":
    main()
