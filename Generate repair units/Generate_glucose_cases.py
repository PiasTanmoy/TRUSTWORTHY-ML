import os
import numpy as np
import pandas as pd
import random
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics


pathName = os.getcwd()

save_dir = 'Glucose_C1_low_2k/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_file_C1 = pd.read_csv('../train-original/train_C1.csv', sep = ',')
train_file_C1_cases = train_file_C1['file_name'].to_list()
print(len(train_file_C1_cases))


os.chdir("../../train")
train_file_path = os.path.abspath(os.curdir)


numFiles = []

print(pathName)
print(train_file_path)
print(os.getcwd())

fileNames = os.listdir(train_file_path)

count = 1
for fileName in fileNames:
    if fileName.endswith(".csv") and fileName in train_file_C1_cases:
        numFiles.append(fileName)
        count+=1

print(len(numFiles))

def create_glucose_distribution(mu, sigma = 24, count = 53):
    a = max(mu-sigma/2, 0)
    b = mu+sigma/2
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    values = dist.rvs(count)
    return values

d = {
    'name': []
}

d = {
    'stay': [],
    'y_true': []
}


for fileName in numFiles:
    file = open(os.path.join(train_file_path, fileName), "r")
    ideal_case_df = pd.read_csv(file)
    case = ideal_case_df.copy()

    #DBP_val = random.choice(DBP)
    #i = random.randint(300, 600)
    i = random.randint(10, 40)

    DBP_val = values = create_glucose_distribution(mu=i, sigma=24, count=case.shape[0])

    case['Glucose'] = values
    case['Glascow coma scale total'] = case['Glascow coma scale total'].astype('Int64')

    name = fileName[:-4] + '_glucose_' + str(i) + '.csv'
    case.to_csv(pathName+'/Glucose_C1_low_2k/' + name, index=False)
    print("SAEVD", name)

    #d['name'].append( name + ',1')

    d['stay'].append(name)
    d['y_true'].append(1)


df = pd.DataFrame(d)
# saving the dataframe
df.to_csv(pathName+'/Glucose_C1_low_2k_name_list.csv', index=False)

print("Hey")