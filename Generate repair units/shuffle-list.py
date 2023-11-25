import pandas as pd

list_file = pd.read_csv("train_listfile_original_H500_L500.csv")

list_file_shuffled = list_file.sample(frac=1)

list_file_shuffled.to_csv('train_listfile_original_H500_L500_shuffled.csv', index=False)
