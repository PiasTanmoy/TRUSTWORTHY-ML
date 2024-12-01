import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics

pathName = os.getcwd() + '/train-original/'
numFiles = []
fileNames = os.listdir(pathName)

save_dir = 'Low Case/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


import pandas as pd
train_listfile = pd.read_csv(pathName + 'train_listfile - 0.csv')['stay'].to_list()
#print(train_listfile)


def create_data_distribution(mu, sigma = 15, count = 53):
    a = max(mu-sigma, 0)
    b = mu+sigma
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    values = dist.rvs(count)
    return values

import random


for fileName in train_listfile:

    file = open(os.path.join(pathName, fileName), "r")
    ideal_case_df = pd.read_csv(file)
    case = ideal_case_df.copy()

    '''
    The sigma should be changed for every attributes
    The first batch of generated test data is somewhat wrong!
    '''

    high_sbp = random.sample(range(20, 70), 1)[0]
    high_sbp = create_data_distribution(mu=high_sbp, sigma=15, count=case.shape[0])
    case['Systolic blood pressure'] = high_sbp

    high_dbp = random.sample(range(10, 40), 1)[0]
    high_dbp = create_data_distribution(mu=high_dbp, sigma=15, count=case.shape[0])
    case['Diastolic blood pressure'] = high_dbp

    high_glucose = random.sample(range(10, 60), 1)[0]
    high_glucose = create_data_distribution(mu=high_glucose, sigma=15, count=case.shape[0])
    case['Glucose'] = high_glucose

    high_resp = random.sample(range(1, 10), 1)[0]
    high_resp = create_data_distribution(mu=high_resp, sigma=15, count=case.shape[0])
    case['Respiratory rate'] = high_resp

    high_Heart_rate = random.sample(range(10, 30), 1)[0]
    high_Heart_rate = create_data_distribution(mu=high_Heart_rate, sigma=15, count=case.shape[0])
    case['Heart Rate'] = high_Heart_rate

    high_body_temp = random.sample(range(34, 36), 1)[0]
    high_body_temp = create_data_distribution(mu=high_body_temp, sigma=15, count=case.shape[0])
    case['Temperature'] = high_body_temp

    case['Mean blood pressure'] = case['Diastolic blood pressure']*(2/3) + case['Systolic blood pressure']/3
    case['Glascow coma scale total'] = case['Glascow coma scale total'].astype('Int64')

    case.to_csv(save_dir + fileName, index=False)
    print(fileName)


