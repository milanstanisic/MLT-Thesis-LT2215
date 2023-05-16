# This script preformats the entire dataframe 
import snscrape.modules.twitter as sntwitter
import pandas as pd
from tqdm import tqdm
import nltk 
import os
from gensim.models.fasttext import FastText
import json

tqdm.pandas()

data = pd.read_csv(r'C:\Users\milan\Documents\Uni Masters\Thesis\Data\Data_users.csv')
data.dropna(subset = 'username', inplace = True)
data = data[['tweet_ID', 'username', 'label']]
data['following'] = None
data = data[data['label'].isin(['normal', 'hateful'])]
#print(data.groupby(['label']).count().to_markdown())

os.chdir(r'C:\Users\milan\Documents\Uni Masters\Thesis\Data\outputs')

#Joining JSON contents with the original dataframe 
followers = pd.DataFrame(columns = ['username', 'following'])
for f in os.listdir():
    with open(f) as file:
        current_file = json.load(file)
        for user in current_file.keys():
            followers.loc[len(followers)] = [user, current_file[user]]

followers.set_index('username', inplace = True)
data['following'] = data['username'].apply(lambda x: followers.at[x, 'following'] if x in followers.index else None)
del followers 
data['n_of_following'] = data['following'].apply(lambda x: len(x) if x is not None else None)
data['username'] = data['username'].apply(lambda x: '@'+x[20:-1])
data.reset_index(inplace = True, drop = True)
os.chdir(r'C:\Users\milan\Documents\Uni Masters\Thesis\Data')

#print(data[data['following'].notna()].groupby(['label']).count()['username'])
contents = pd.read_csv('data_subset.csv')[['tweet_ID', 'tweet_contents']]
contents.set_index('tweet_ID', inplace = True)
data['tweet_contents'] = data['tweet_ID'].progress_apply(lambda x: contents.at[x, 'tweet_contents'])
'''
#Contents extraction 
def getcontents(x):
    contents = ""
    for i,tweet in enumerate(sntwitter.TwitterTweetScraper(x).get_items()): 
        if i>1: 
            break
        try:
            contents = json.loads(tweet.json())['rawContent']
        except KeyError:
            pass
    return contents[:contents.index('#')] if '#' in contents else contents

data['tweet_contents'] = data['tweet_ID'].progress_apply(lambda x: getcontents(x))
print('Extracted contents.')'''
data.to_csv('data_subset.csv')