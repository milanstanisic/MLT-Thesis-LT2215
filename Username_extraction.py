# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import re

def getusername(x):
    REGEX = re.compile('https://twitter\.com/[^ /]+/')
    username = ""
    for i,tweet in enumerate(sntwitter.TwitterTweetScraper(x).get_items()): 
        if i>1: 
            break
        username = REGEX.search(tweet.url).group()
    return username

os.chdir(r"C:\Users\milan\Documents\Uni Masters\Thesis\Data")

tweet_id = pd.read_csv('data.csv')

urls = []
i = 0
print(tweet_id.shape)
for uid in tweet_id['tweet_id']:
    try:
        urls.append(getusername(int(uid)))
    except:
        urls.append(None)
    i += 1
    if i%100 == 0:
        print(i)

data = pd.DataFrame({'tweet_ID':tweet_id['tweet_id'], 'username':urls, 'label':tweet_id['label']})
data.to_csv('Data_users.csv')