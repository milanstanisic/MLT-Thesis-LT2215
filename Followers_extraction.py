import pandas as pd 
from Scweet.user import get_users_following
import numpy as np 
import os 

#Script for extraction of followers per user and further preprocessing of data 
os.chdir(r'PATH') #replace with your local working directory
data = pd.read_csv('Data_users.csv')
data = data[data['label'].isin(['hateful', 'normal'])]
data = data.dropna(subset=['username'])
data.reset_index(drop = True, inplace = True)
print(data['username'].nunique())



for i in range(600, 3000): #customizable range
    try:
        users = data['username'][i*10:(i+1)*10]
        print('User subset selected successfully.')
        following = get_users_following(users, '.env', file_path='outputs/{}'.format(i))
    except:
        continue
    print(i)
print('Done')
