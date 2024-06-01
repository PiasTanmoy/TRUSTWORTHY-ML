import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics

seed = 'seed 5 - 15645_episode1_timeseries (close to ideal).csv'
save_dir = 'Glucose set - Seed5 - SD 48/'

fileName = seed
file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)
print(ideal_case_df)

glucose = np.arange(0, 1300, 1).tolist()

'''
This function  generates normal distribution 
using two boundaries 
'''
def create_glucose_distribution(mu, sigma = 48, count = 53):
    a = max(mu-sigma/2, 0)
    b = mu+sigma/2
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


for i in range(20, 600):
    file = open(os.path.join('Seeds/' + fileName), "r")
    ideal_case_df = pd.read_csv(file)

    case = ideal_case_df.copy()

    values = create_glucose_distribution(mu=i, sigma=48, count=case.shape[0])
    print(i, "Mean", sum(values) / len(values), "SD", statistics.pstdev(values))

    stats_['mu'].append(sum(values) / len(values))
    stats_['SD'].append(statistics.pstdev(values))

    case['Glucose'] = values
    case['Glascow coma scale total'] = case['Glascow coma scale total'].astype('Int64')

    name = 'Glucose_' + str(glucose[i]) + '.csv'
    case.to_csv(save_dir + name, index=False)

    #print("SAEVD", name)
    d['stay'].append( name)
    d['y_true'].append(0)

    stats_['name'].append(name)



df = pd.DataFrame(d)
# saving the dataframe
df.to_csv(save_dir + 'name_list.csv', index=False)

df = pd.DataFrame(stats_)
df.to_csv(save_dir +'stats.csv', index=False)

print("Hey")