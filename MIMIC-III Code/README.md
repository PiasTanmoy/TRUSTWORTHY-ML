# TRUSTWORTHY-ML

## Prepare data and codes
1. MIMIC III dataset can be downloaded from https://mimic.physionet.org/
2. We construct a benchmark machine learning dataset from https://github.com/YerevaNN/mimic3-benchmarks using the MIMIC III dataset
3. In the Benchmark GitHub, please follow the six steps specified in the “Building a benchmark” subsection to create a benchmark machine learning dataset for the In-hospital mortality prediction task.
4. Replace "preprocessing.py" and "metrics.py" located in mimic3-benchmarks/mimic3models with the files located in the "Helper" directory.
5. Create two virtual environments using two files in the "Requirements" directory. One with Python 3.7 and another with Python 3.9

## Train Model
Activate virtual environment 3.7

LSTM training
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --epochs 100 --output_dir mimic3models/in_hospital_mortality

Train-Channel Wise LSTM
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train-repair-Respiratory-CW-LSTM --batch_size 8 --epochs 100 --output_dir mimic3models/in_hospital_mortality

Train-Test-Logistic Regression 
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/LR_trained

For LSTM and CW-LSTM output (saved models and predictions) will be saved "mimic3models/in_hospital_mortality/keras_states" and for LR the output dir is "mimic3models/in_hospital_mortality/LR_trained/results"

Select the epoch with best performance on validation set. We considered AUPRC and loss as main metrics. 
1. Use the code "Analysis/MIMIC_III_Training_State_Selection.ipynb" to load the training log.
2. Sort the training log using "val AUPRC". Select 3 epochs with maximum value of "val AUPRC".
3. Select one epoch from the 3 using lowest validation loss.

## Test baseline models using the  original MIMIC III test set
Determine the best model from the epochs and load the model. In our case, the 39th epoch performed the best. 

Testing LSTM (epoch 39):
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_LSTM_trail_1/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch39.test0.2848591662846242.state

Note: We have renamed the "mimic3models/in_hospital_mortality/keras_states" to ""mimic3models/in_hospital_mortality/keras_states_LSTM_trail_1".

Test Channel Wise LSTM (epoch 37)
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_CW_LSTM_trail_1/k_channel_wise_lstms.n16.szc4.0.d0.3.dep2.bs8.ts1.0.epoch37.test0.28432124478986437.state

Note: We have renamed the "mimic3models/in_hospital_mortality/keras_states" to ""mimic3models/in_hospital_mortality/keras_states_CW_LSTM_trail_1".

## Create attribute-based test sets
First we selected seeds which we going to be used to create the test sets. We selected 5 seeds to create 5 versions of each test sets to acount the test set variability. We selected the following 5 seeds from the "data/in-hospital-mortality/train" directory.
1. 5482_episode1_timeseries.csv
2. 6003_episode1_timeseries.csv
3. 6552_episode2_timeseries.csv
4. 12605_episode1_timeseries.csv
5. 15645_episode1_timeseries.csv

Copy the mentioned 5 seeds and save them to the "Generate test sets/Seeds" folder

Run the Python codes located in the "Generate test sets" directory to generate attribute-based test sets which are derived from original patient cases from the MIMIC III dataset. 

For example: Generate_DBP_cases.py will create diastolic blood pressure patient cases from a single seed.
Change the seed to create multiple test sets. 

Note: You can set custom seed and output dir in the file. Just change the following 2 variables in each python file
(In Generate_DBP_cases.py)
seed = '15645_episode1_timeseries.csv'
save_dir = 'Diastolic BP set - Seed5 - SD 15/'

## Gradient ascent based tests
1. Activate the virtual environment with Python 3.9
2. Open one of the Python files. For example: "Gradient test/gradient_ascent_Diastolic_BP_cases.py"
3. Set the working directory
4. Set three LSTM model states 1, 2, and 3
5. Set the path of normalizer_state
6. Run the Python file and it will generate gradient cases


## Test model with Test set
To test LSTM and CW-LSTM model, go to the file "mimic3-benchmarks/mimic3models/in_hospital_mortality
/main.py" and set the test set directory as follows

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'custom_test_directory'),
                                            listfile=os.path.join(args.data, 'custom_test_listfile.csv'),
                                            period_length=48.0)

You can also set the test output directory in the following statement
path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"

Testing LSTM (epoch 39):
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_LSTM_trail_1/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch39.test0.2848591662846242.state

Test Channel Wise LSTM (epoch 37)
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_CW_LSTM_trail_1/k_channel_wise_lstms.n16.szc4.0.d0.3.dep2.bs8.ts1.0.epoch37.test0.28432124478986437.state

For LR test, we have to train and test as follows.
Go to "mimic3-benchmarks/mimic3models/in_hospital_mortality/logistic/main.py"
Change the custom_test directory 

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

Train-Test-Logistic Regression 
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/LR_trained


