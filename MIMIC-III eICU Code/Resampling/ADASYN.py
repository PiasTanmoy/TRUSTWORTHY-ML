import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import numpy as np
import collections

os.chdir('/raid/home/tanmoysarkar/Trustworthiness/MIMIC-III/mimic3-benchmarks')

GCS_motor =  {
"1 No Response": 1,
"No response": 1,
"2 Abnorm extensn": 2,
"Abnormal extension": 2,
"3 Abnorm flexion": 3,
"Abnormal Flexion": 3,
"4 Flex-withdraws": 4,
"Flex-withdraws": 4,
"5 Localizes Pain": 5,
"Localizes Pain": 5,
"6 Obeys Commands": 6,
"Obeys Commands": 6
}


GCS_motor_reverse =  {
1: "1 No Response",
2: "2 Abnorm extensn",
3: "3 Abnorm flexion",
4: "4 Flex-withdraws",
5: "5 Localizes Pain",
6: "6 Obeys Commands",
}



GCS_eye = {
"None": 0,
"1 No Response": 1,
"2 To pain": 2,
"To Pain": 2,
"3 To speech": 3,
"To Speech": 3,
"4 Spontaneously": 4,
"Spontaneously": 4
}


GCS_eye_reverse = {
0: "None",
1: "1 No Response",
2: "2 To pain",
3: "3 To speech",
4: "4 Spontaneously",
}


GCS_speech = {
"No Response-ETT": 1,
"No Response": 1,
"1 No Response": 1,
"1.0 ET/Trach": 1,
"2 Incomp sounds": 2,
"Incomprehensible sounds": 2,
"3 Inapprop words": 3,
"Inappropriate Words": 3,
"4 Confused": 4,
"Confused": 4,
"5 Oriented": 5,
"Oriented": 5
}


GCS_speech_reverse = {
1: "1 No Response",
2: "2 Incomp sounds",
3: "3 Inapprop words",
4: "4 Confused",
5: "5 Oriented",
}

default_values = {
    "Capillary refill rate": "0.0",
    "Diastolic blood pressure": "59.0",
    "Fraction inspired oxygen": "0.21",
    "Glascow coma scale eye opening": "4 Spontaneously",
    "Glascow coma scale motor response": "6 Obeys Commands",
    "Glascow coma scale total": "15",
    "Glascow coma scale verbal response": "5 Oriented",
    "Glucose": "128.0",
    "Heart Rate": "86",
    "Height": "170.0",
    "Mean blood pressure": "77.0",
    "Oxygen saturation": "98.0",
    "Respiratory rate": "19",
    "Systolic blood pressure": "118.0",
    "Temperature": "36.6",
    "Weight": "81.0",
    "pH": "7.4"
  }


# # Load the data from the CSV file

# df['Glascow coma scale eye opening'] = df['Glascow coma scale eye opening'].map(GCS_eye)
# df['Glascow coma scale motor response'] = df['Glascow coma scale motor response'].map(GCS_motor)
# df['Glascow coma scale verbal response'] = df['Glascow coma scale verbal response'].map(GCS_speech)
# df['Glascow coma scale total'] = df['Glascow coma scale total'].astype('Int64')

s = "train"
# Define paths
data_folder = 'data/task-data/in-hospital-mortality/train-filled-48H'
label_file = 'data/task-data/in-hospital-mortality/train_listfile.csv'
output_folder = 'data/task-data/in-hospital-mortality/train-filled-48H-ADASYN'
output_folder_labels = "data/task-data/in-hospital-mortality/"

df = pd.read_csv(data_folder+"/3_episode1_timeseries.csv")

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

X = np.load(output_folder_labels + "X_" + s + "_filled_48H_flattened_raw.npy")
y = np.load(output_folder_labels + "y_" + s + "_filled_48H_flattened_raw.npy")

# restore np.load for future normal usage
np.load = np_load_old

print("ADASYN Working ......")
# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

print("ADASYN Completed!!")

missing = []
c = 1



# Iterate through the resampled data and save them as new CSV files
for i, (data_row, label) in enumerate(zip(X_resampled, y_resampled)):
    # Reshape back to the original 2D format (number of rows and columns)
    reshaped_data = data_row.reshape(df.shape)
    
    # Convert to DataFrame
    resampled_df = pd.DataFrame(reshaped_data, columns=df.columns)

    if resampled_df.isnull().values.any():
        print("Missing:" , c)
        c += 1

    resampled_df['Glascow coma scale eye opening'] = resampled_df['Glascow coma scale eye opening'].map(GCS_eye_reverse)
    resampled_df['Glascow coma scale motor response'] = resampled_df['Glascow coma scale motor response'].map(GCS_motor_reverse)
    resampled_df['Glascow coma scale verbal response'] = resampled_df['Glascow coma scale verbal response'].map(GCS_speech_reverse)

    resampled_df = resampled_df.fillna(method='ffill')  # Fill missing values with previous row values
    resampled_df = resampled_df.fillna(default_values)  # Fill remaining missing values with default values

    resampled_df['Glascow coma scale total'] = resampled_df['Glascow coma scale total'].round()
    resampled_df['Glascow coma scale total'] = resampled_df['Glascow coma scale total'].astype('Int64')
    resampled_df['Hours'] = resampled_df["Hours"].astype(float)
    
    # Generate a new file name
    new_file_name = f'resampled_{i}.csv'
    new_file_path = os.path.join(output_folder, new_file_name)
    
    # Save the DataFrame as CSV
    resampled_df.to_csv(new_file_path, index=False)
    
    print(i, new_file_name)
    if resampled_df.isnull().values.any():
        missing.append(new_file_name)

 
# Save the new labels
resampled_labels_df = pd.DataFrame({
    'stay': [f'resampled_{i}.csv' for i in range(len(y_resampled))],
    'y_true': y_resampled
})

resampled_labels_file = os.path.join(output_folder_labels, s+'_ADASYN_labels.csv')
resampled_labels_df.to_csv(resampled_labels_file, index=False)

print(f"Saved resampled labels to: {resampled_labels_file}")
print(len(missing), missing)

counter = collections.Counter(y_resampled)
print(len(y_resampled))
print(counter)

