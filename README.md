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
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --epochs 60 --output_dir mimic3models/in_hospital_mortality

Train-Channel Wise LSTM
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train-repair-Respiratory-CW-LSTM --batch_size 8 --output_dir mimic3models/in_hospital_mortality

Train-Test-Logistic Regression 
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/LR_trained

## Test baseline models using the  original MIMIC III test set
Determine the best model from the epochs and load the model. In our case, the 39th epoch performed the best. 

Testing LSTM (epoch 39):
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_LSTM_trail_1/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch39.test0.2848591662846242.state

Test Channel Wise LSTM (epoch 37)
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test-DBP --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_CW_LSTM_trail_1/k_channel_wise_lstms.n16.szc4.0.d0.3.dep2.bs8.ts1.0.epoch37.test0.28432124478986437.state

## Create attribute-based test sets
Run the Python codes located in the "Generate test sets" directory to generate attribute-based test sets which are derived from original patient cases from the MIMIC III dataset. 
For example: Generate_DBP_cases.py will create diastolic blood pressure patient cases from a single seed.
Change the seed to create multiple test sets. 

## Gradient ascent based tests
1. Activate the virtual environment with Python 3.9
2. Open one of the Python files. For example: "Gradient test/gradient_ascent_Diastolic_BP_cases.py"
3. Set the working directory
4. Set three LSTM model states 1, 2, and 3
5. Set the path of normalizer_state
6. Run the Python file and it will generate gradient cases

## Generate repair units
1. Execute the Python files located in "Generate repair units". For example, Generate_Diastolic_BP_cases.py will generate diastolic blood pressure repair units for the LSTM model. 
2. Combine the "train_listfile.csv" with generated "DBP_C1_low_2k_name_list.csv".
3. Combine the train set with the repair unit set.
4. Train the LSTM model using the combined training set



