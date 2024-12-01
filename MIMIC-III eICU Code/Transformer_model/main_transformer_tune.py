import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import os
from metrics import print_metrics_binary
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Masking
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Masking
import keras_tuner as kt


# Model architecture parameters
input_dim = 76
n_timesteps = 48
batch_size = 8
num_classes = 1
depth = 2
units = 16  # LSTM units
dropout = 0.3
rec_dropout = 0.0
batch_norm = False
epochs = 100


# Custom callback for calculating additional metrics after each epoch
class PrintAUC(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print AUROC after each epoch, if needed
        print(f"Epoch {epoch+1}, Val AUROC: {logs.get('val_auroc')}")

    def on_trial_end(self, trial, logs=None):
        # Print AUROC for each completed trial
        val_auroc = logs.get('val_auroc')
        print(f"Trial {trial.trial_id} - Val AUROC: {val_auroc}")

# Define the model-building function with hyperparameters
def build_transformer_model(hp):
    input_dim = 76  # Fixed input dimension
    num_classes = 1  # Binary classification

    inputs = Input(shape=(None, input_dim), name="X")

    # Masking layer for padded sequences
    x = Masking()(inputs)
    
    # Positional encoding layer
    position_embedding = layers.Embedding(input_dim=50, output_dim=input_dim)(tf.range(start=0, limit=48))
    x = x + position_embedding
    
    # Hyperparameters to tune
    num_heads = hp.Choice('num_heads', values=[2, 4, 8])
    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3)
    ff_dim = hp.Choice('ff_dim', values=[16, 64, 256])
    dropout = hp.Choice('dropout', values = [0.1, 0.3, 0.5])
    batch_norm = hp.Boolean('batch_norm')
    units = hp.Choice('units', values=[16, 32, 64])
    

    # Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim, dropout=dropout)(x1, x1)
        x2 = layers.Add()([x, attention_output])  # Residual connection

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff_output = layers.Dense(ff_dim, activation="relu")(x3)
        ff_output = layers.Dense(input_dim)(ff_output)
        x = layers.Add()([x2, ff_output])  # Residual connection

    # Global average pooling for sequence output
    x = layers.GlobalAveragePooling1D()(x)

    # Batch normalization
    x = BatchNormalization()(x) if batch_norm else x
    
    # Dropout layer
    x = Dropout(dropout)(x)
    
    # Dense layer for binary classification
    x = Dense(units, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Build and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name="auroc")])
    
    return model

# Step 2: Initialize the Tuner and Search Space
tuner = kt.Hyperband(
    build_transformer_model,
    objective=kt.Objective("val_auroc", direction="max"),  # Optimization metric
    max_epochs=20,
    factor=3,
    directory='Models',
    project_name='Transformer_tuning2_eICU'
)

# X_train = np.load("Data/Preprocessed_data/Train_X.npy")  # Random input data
# y_train = np.load("Data/Preprocessed_data/Train_Y.npy")  # Binary target

# X_val = np.load("Data/Preprocessed_data/Val_X.npy")  # Random validation data
# y_val = np.load("Data/Preprocessed_data/Val_Y.npy")  # Binary validation target

X_train = np.load("Data/eICU/Preprocessed_data/train_X.npy")  # Random input data
y_train = np.load("Data/eICU/Preprocessed_data/train_Y.npy")  # Binary target

X_val = np.load("Data/eICU/Preprocessed_data/val_X.npy")  # Random validation data
y_val = np.load("Data/eICU/Preprocessed_data/val_Y.npy")  # Binary validation target

# Step 3: Perform the Hyperparameter Search
# Replace X_train, y_train with your training data
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[PrintAUC()])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f" - num_heads: {best_hps.get('num_heads')}")
print(f" - num_transformer_blocks: {best_hps.get('num_transformer_blocks')}")
print(f" - ff_dim: {best_hps.get('ff_dim')}")
print(f" - dropout: {best_hps.get('dropout')}")
print(f" - batch_norm: {best_hps.get('batch_norm')}")
print(f" - units: {best_hps.get('units')}")








