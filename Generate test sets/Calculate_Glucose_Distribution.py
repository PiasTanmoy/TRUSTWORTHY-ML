import os
import numpy as np
import pandas as pd

pathName = os.getcwd()


numFiles = []
fileNames = os.listdir(pathName)

count = 1
for fileNames in fileNames:
    if fileNames.endswith(".csv"):
        numFiles.append(fileNames)
        print(count, fileNames)
        count+=1

seed = 'Seed 1 - 6582_episode1_timeseries.csv'
fileName = seed

print(numFiles)


glucose = np.arange(0, 800, 10).tolist()

d = {
    'name': []
}


for i in range(len(glucose)):
    file = open(os.path.join(pathName, fileName), "r")
    ideal_case_df = pd.read_csv(file)

    case = ideal_case_df.copy()

    case['Glucose'] = glucose[i]

    name = 'Glucose_' + str(glucose[i]) + '.csv'
    case.to_csv('Glucose set - Seed1/' + name, index=False)
    print("SAEVD", name)
    d['name'].append( name + ', 0')


df = pd.DataFrame(d)
# saving the dataframe
df.to_csv('Glucose set - Seed1/name_list.csv', index=False)

print("Hey")