# TRUSTWORTHY-ML

## Prepare data and codes
1. MIMIC III dataset can be downloaded from https://mimic.physionet.org/
2. We construct a benchmark machine learning dataset from https://github.com/YerevaNN/mimic3-benchmarks using the MIMIC III dataset
3. In the Benchmark GitHub, please follow the six steps specified in the “Building a benchmark” subsection to create a benchmark machine learning dataset for the In-hospital mortality prediction task.
4. Replace "preprocessing.py" and "metrics.py" located in mimic3-benchmarks/mimic3models with the files located in the "Helper" directory.
5. Create two virtual environments using two files in the "Requirements" directory. 

## Train Model
Activate virtual environment 3.7

LSTM training
python -um mimic3models.in_hospital_mortality.main-fresh --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --epochs 60 --output_dir mimic3models/in_hospital_mortality

Train-Channel Wise LSTM
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/channel_wise_lstms.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train-repair-Respiratory-CW-LSTM --batch_size 8 --output_dir mimic3models/in_hospital_mortality

Train-Test-Logistic Regression 
python -um mimic3models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic3models/in_hospital_mortality/LR_trained

## Test model
Determine the best model from the epochs and load the model. In our case, the 39th epoch performed the best. 

Testing LSTM (epoch 39):
python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models/in_hospital_mortality/keras_states_LSTM_trail_1/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch39.test0.2848591662846242.state




### Trian baseline model


