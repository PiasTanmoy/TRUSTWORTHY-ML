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

# Model architecture parameters



# Custom callback for calculating additional metrics after each epoch
class MetricsCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, file_name='training_metrics.csv', append=True):
        super(MetricsCSVLogger, self).__init__()
        self.validation_data = validation_data
        self.training_data = training_data
        self.file_name = file_name
        self.append = append

        dummy_metrics = print_metrics_binary([0, 1, 1, 0], [[0], [0], [1], [1]])

        self.columns = [ 'epoch', 'loss', 'accuracy' ] + [m for m in dummy_metrics.keys()] + ['val_loss', 'val_accuracy'] + [f"val_{m}" for m in dummy_metrics.keys()]
    
        if not append or not os.path.exists(self.file_name):
            # Create the CSV file and write the header
            with open(self.file_name, 'w') as f:
                pass
                #f.write(','.join(self.columns) + '\n')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Predict on validation data
        val_pred = self.model.predict(self.validation_data[0])
        val_true = self.validation_data[1]
        val_metrics = print_metrics_binary(val_true, val_pred)

        training_pred = self.model.predict(self.training_data[0])
        training_true = self.training_data[1]
        training_metrics = print_metrics_binary(training_true, training_pred)
        
        val_pred_binary = (val_pred > 0.5).astype(int)
        
        auroc = roc_auc_score(val_true, val_pred)
        recall = recall_score(val_true, val_pred_binary)
        precision = precision_score(val_true, val_pred_binary)
        f1 = f1_score(val_true, val_pred_binary)
        accuracy = accuracy_score(val_true, val_pred_binary)
        
        logs['val_auroc'] = auroc
        logs['val_recall'] = recall
        logs['val_precision'] = precision
        logs['val_f1'] = f1
        logs['val_accuracy'] = accuracy
        
        # # Save metrics to logs
        print(f"\nEpoch {epoch + 1}: val_auroc: {auroc:.4f} - val_recall: {recall:.4f} - val_precision: {precision:.4f} - val_f1: {f1:.4f} - val_accuracy: {accuracy:.4f}")
        
         # Prepare the data to log to CSV
        row = [epoch + 1, logs.get('loss'), logs.get('accuracy')] + [training_metrics[m] for m in training_metrics.keys()] + [logs.get('val_loss'), logs.get('val_accuracy')] + [val_metrics[m] for m in val_metrics.keys()]

        # Write the row to the CSV file
        with open(self.file_name, 'a') as f:
            if epoch == 0 and self.append:  # Write header if appending and it's the first epoch
                f.write(','.join(self.columns) + '\n')
            f.write(','.join([str(x) for x in row]) + '\n')


st = '''
Best hyperparameters:
 - num_heads: 4
 - num_transformer_blocks: 2
 - ff_dim: 16
 - dropout: 0.5
 - batch_norm: True
 - units: 32
'''
print(st)

# Function to build the Transformer Model
def build_model(input_dim=76, units=32, dropout=0.5, num_classes=1, num_heads=4, num_transformer_blocks=2, ff_dim=16, batch_norm=True):

    inputs = Input(shape=(None, input_dim), name="X")

    # Masking layer for padded sequences
    x = Masking()(inputs)
    
    # Positional encoding layer
    position_embedding = layers.Embedding(input_dim=50, output_dim=input_dim)(tf.range(start=0, limit=48))
    x = x + position_embedding

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


input_dim = 76
n_timesteps = 48
epochs = 100
batch_size = 32


# Build the model
model = build_model()

# Model summary
model.summary()



# CSV Logger to save metrics
X_train = np.load("Data/eICU/Preprocessed_data/train_X.npy")  # Random input data
y_train = np.load("Data/eICU/Preprocessed_data/train_Y.npy")  # Binary target

X_val = np.load("Data/eICU/Preprocessed_data/val_X.npy")  # Random validation data
y_val = np.load("Data/eICU/Preprocessed_data/val_Y.npy")  # Binary validation target

################# Set path #################

# Save the model after each epoch
save_path = "Models/Transformer_2_eICU/Trial_3"

################# Set path #################

os.makedirs(save_path+'/checkpoints', exist_ok=True)
csv_file_path = os.path.join(save_path, 'training_metrics.csv')
model_checkpoint_path = os.path.join(save_path+'/checkpoints', 'model_epoch_{epoch:02d}.keras')

# Callbacks
checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=False, save_weights_only=False)

# Compile the custom metrics callback (this requires you to pass validation data during training)
metrics_csv_logger = MetricsCSVLogger(training_data = (X_train, y_train), validation_data=(X_val, y_val), file_name=csv_file_path, append=True)


# Training the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, metrics_csv_logger],
    verbose = 0
)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 25})

def plot_training_history(history, save_path="training_history.png"):
    # Extract the metrics from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history.get('accuracy')  # 'accuracy' key can change depending on the metric you use
    val_accuracy = history.history.get('val_accuracy')

    epochs = range(1, len(loss) + 1)

    # Create subplots for accuracy and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    ax1.plot(epochs, loss, 'b', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Accuracy
    if accuracy and val_accuracy:  # Check if accuracy data is available
        ax2.plot(epochs, accuracy, 'b', label='Training accuracy')
        ax2.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(save_path)

    # Show the plots
    plt.show()

plot_training_history(history, save_path=os.path.join(save_path, "training_validation_history.png"))
