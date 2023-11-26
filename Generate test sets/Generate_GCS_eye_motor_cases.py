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

save_dir = 'GCS-eye-motor - Seed1 - SD 0/'
file_name_pre = 'GCS-'
attr_name = 'Glascow coma scale eye opening'


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fileName = seed
file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)
print(ideal_case_df)

sigma = 15
start = 1
end = 4


'''
This function  generates normal distribution 
using two boundaries 
'''
def create_respiratory_rate_distribution(mu, sigma = 48, count = 53):
    a = max(mu-sigma, 0)
    b = mu+sigma
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    values = dist.rvs(count)
    return values

d = {
    'stay': [],
    'y_true': []
}

stats_ = {
    'name': [],
    'mu': [],
    'SD': []

}

# 'Glascow coma scale eye opening'
gcs_eye=[
'1 No Response',
'2 To pain',
'3 To speech',
'4 Spontaneously'
]
# 'Glascow coma scale motor response'
gcs_motor = [
'1 No Response',
'2 Abnorm extensn',
'3 Abnorm flexion',
'4 Flex-withdraws',
'5 Localizes Pain',
'6 Obeys Commands'
]

# Glascow coma scale verbal response
gcs_verbal=[
'1.0 ET/Trach',
'2 Incomp sounds',
'3 Inapprop words',
'4 Confused',
'5 Oriented'
]


file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)

flag = 0
verbal = 4
for eye in range(len(gcs_eye)):
    for motor in range(len(gcs_motor)):

        gcs_case = ideal_case_df.copy()
        gcs_total = eye + motor + verbal + 3

        gcs_case['Glascow coma scale total'] = gcs_total
        gcs_case['Glascow coma scale eye opening'] = gcs_eye[eye]
        gcs_case['Glascow coma scale motor response'] = gcs_motor[motor]
        gcs_case['Glascow coma scale verbal response'] = gcs_verbal[verbal]
        gcs_case['Glascow coma scale total'] = gcs_case['Glascow coma scale total'].astype('Int64')

        name = 'gcs' + '_e' + str(eye + 1) + '_m' + str(motor + 1) + '_v' + str(
            verbal + 1) + '_t' + str(gcs_total) + '.csv'

        print(name[:-4])

        gcs_case.to_csv(save_dir + name, index=False)

        # print("SAEVD", name)
        d['stay'].append(name)

        if flag == 0:
            d['y_true'].append(1)
            flag = 1
        else:
            d['y_true'].append(0)


        print('gcs_t_' + str(gcs_total) + '_' + str(eye + 1) + str(motor + 1) + str(verbal + 1) + '.csv,', 0)


df = pd.DataFrame(d)
# saving the dataframe
df.to_csv(save_dir + 'name_list.csv', index=False)


print("Hey")