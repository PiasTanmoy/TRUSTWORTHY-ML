import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics

#seed = 'seed 1 - 5482_episode1_timeseries (close to ideal).csv'
#seed = 'seed 2 - 6003_episode1_timeseries (close to ideal).csv'
#seed = 'seed 3 - 6552_episode2_timeseries (close to ideal).csv'
#seed = 'seed 4 - 12605_episode1_timeseries (close to ideal).csv'
seed = 'seed 5 - 15645_episode1_timeseries (close to ideal).csv'

save_dir = 'Respiratory rate set - Seed5 - SD 5/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fileName = seed
file = open(os.path.join('Seeds/' + fileName), "r")
ideal_case_df = pd.read_csv(file)
print(ideal_case_df)


sigma = 5
start = 1
end = 101
respiratory_rate = np.arange(start, end, 1).tolist()

'''
This function  generates normal distribution 
using two boundaries 
'''
def create_respiratory_rate_distribution(mu, sigma = 5, count = 53):
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


for i in range(start, end):
    file = open(os.path.join('Seeds/' + fileName), "r")
    ideal_case_df = pd.read_csv(file)

    case = ideal_case_df.copy()

    values = create_respiratory_rate_distribution(mu=i, sigma=sigma, count=case.shape[0])
    print(i, "Mean", sum(values) / len(values), "SD", statistics.pstdev(values))

    stats_['mu'].append(sum(values) / len(values))
    stats_['SD'].append(statistics.pstdev(values))

    case['Respiratory rate'] = values
    case['Glascow coma scale total'] = case['Glascow coma scale total'].astype('Int64')

    name = 'Resp_rate_' + str(i) + '.csv'
    case.to_csv(save_dir + name, index=False)

    #print("SAEVD", name)
    d['stay'].append( name)

    if i == start:
        d['y_true'].append(1)
    else:
        d['y_true'].append(0)

    stats_['name'].append(name)



df = pd.DataFrame(d)
# saving the dataframe
df.to_csv(save_dir + 'name_list.csv', index=False)

df = pd.DataFrame(stats_)
df.to_csv(save_dir +'stats.csv', index=False)


print("Hey")