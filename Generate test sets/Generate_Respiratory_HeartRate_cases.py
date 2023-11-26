import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics

seed = 'seed 1 - 5482_episode1_timeseries (close to ideal).csv'
#seed = 'seed 2 - 6003_episode1_timeseries (close to ideal).csv'
#seed = 'seed 3 - 6552_episode2_timeseries (close to ideal).csv'
#seed = 'seed 4 - 12605_episode1_timeseries (close to ideal).csv'
#seed = 'seed 5 - 15645_episode1_timeseries (close to ideal).csv'
save_dir = 'Respiratory HeartRate set - Seed1 - SD 20-10/'


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fileName = seed
file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)
print(ideal_case_df)

sigma_DBP = sigma_Respiratory = 20
sigma_Glucose = sigma_Heart = 10
start = 0
end = 250

DBP = np.linspace(0, 50, 50).astype(int)
Glucose = np.linspace(0, 100, 50).astype(int)

'''
This function  generates normal distribution 
using two boundaries 
'''
def create_data_distribution(mu, sigma = 15, count = 53):
    a = max(mu-sigma, 0)
    b = mu+sigma
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    values = dist.rvs(count)
    return values

d = {
    'stay': [],
    'y_true': []
}

file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)

for i in range(0, DBP.shape[0]):

    case = ideal_case_df.copy()

    values = create_data_distribution(mu=DBP[i], sigma=sigma_DBP, count=case.shape[0])
    case['Respiratory rate'] = values
    print(DBP[i], "Respiratory rate Mean", sum(values) / len(values), "SD", statistics.pstdev(values))

    for j in range(0, Glucose.shape[0]):

        values = create_data_distribution(mu=Glucose[j], sigma=sigma_Glucose, count=case.shape[0])
        case['Heart Rate'] = values
        print(Glucose[j], "Heart Rate Mean", sum(values) / len(values), "SD", statistics.pstdev(values))

        case['Glascow coma scale total'] = case['Glascow coma scale total'].astype('Int64')
        #case['Mean blood pressure'] = case['Diastolic blood pressure']*(2/3) + case['Systolic blood pressure']/3

        name = 'Real_Seed1_' + 'RespiratoryRate_' + str(DBP[i]) + '_HeartRate_' + str(Glucose[j]) + '.csv'
        case.to_csv(save_dir + name, index=False)

        #print("SAEVD", name)
        d['stay'].append( name)

        if i == start:
            d['y_true'].append(1)
        else:
            d['y_true'].append(0)

df = pd.DataFrame(d)
# saving the dataframe
df.to_csv(save_dir + 'name_list.csv', index=False)


print("Hey")