# The second part of preprocessing: getting tweet embeddings; language model
from gensim.models.fasttext import FastText
from gensim.models import fasttext
import os 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd 
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
import re

from random import random

tqdm.pandas()



os.chdir(r'PATH') #replace with your working directory
data = pd.read_csv('data_subset.csv') #REPLACE WITH THE ACTUAL FILE PATH 
data = data[data['label'].isin(['hateful', 'normal'])]

def preprocess_raw_text(x):
    try:
        return [lemmatizer.lemmatize(y.lower()) if '@' not in y else y for y in word_tokenize(x)]
    except:
        return None
    
def seek_mentions(x):
    if x is None:
        return x
    users = []
    user = False
    for element in x:
        if element == '@':
            user = True
            continue
        if user: 
            users.append('@' + element)
            user = False
    return users 

def filter_words(x):
    filtered = []
    for y in x:
        if re.fullmatch("^[A-Za-z']*$", y) and y != 'http':
            filtered.append(y)
    return filtered

#Training embeddings
lemmatizer = WordNetLemmatizer()
print(len(data[data['label'].isin(['hateful'])]))
data['tweet_contents'] = data['tweet_contents'].apply(lambda x: preprocess_raw_text(x))
data['users_mentioned'] = data['tweet_contents'].apply(lambda x: seek_mentions(x))
data.dropna(subset = ['tweet_contents'], inplace = True)
data['tweet_contents'] = data['tweet_contents'].apply(lambda x: filter_words(x))
#wv = FastText.load_fasttext_format('FastText/cc.en.300.vec.gz')
model = FastText(vector_size=50, window=3, min_count=1)  # instantiate
model.build_vocab(corpus_iterable=list(data['tweet_contents']))
model.train(corpus_iterable=list(data['tweet_contents']), total_examples=len(list(data['tweet_contents'])), epochs=10)
data['tweet_vector'] = data['tweet_contents'].progress_apply(lambda x: np.mean(np.array([model.wv[word] for word in x]), axis = 0))
data.dropna(subset = ['tweet_vector'], inplace = True)
data['drop_this'] = data['tweet_vector'].apply(lambda x: False if x.shape[0] == 50 else True)
data = data[data['drop_this'].isin([False])]
data.drop(['drop_this'], axis = 1, inplace = True)

experimental_sample = data[data['following'].notna()]
data['i'] = data.index
print(len(experimental_sample))
data['include'] = data.apply(lambda x: True if x['i'] not in experimental_sample.index else False,axis = 1)
data = data[data['include'].isin([True])]

data = data.sample(frac=1)
data.reset_index(drop = True, inplace = True)
#data['i'] = data.index
test_data = experimental_sample
#test_data = data[data['i'].gt(int(len(data)*0.7))]
#data = data[data['i'].lt(int(len(data)*0.7))]
#Undersampling - test 
print(len(data[data['label'].isin(['normal'])]))
print(len(data[data['label'].isin(['hateful'])]))

no_hate = data[data['label'].isin(['normal'])]
no_hate['include'] = [True if random() <= (len(data[data['label'].isin(['hateful'])])/len(data))*1 else False for x in range(len(no_hate))]
no_hate = no_hate[no_hate['include'].isin([True])]
data = data[data['i'].isin(no_hate.index) | data['label'].isin(['hateful']) ]
print(len(data[data['label'].isin(['normal'])]))
print(len(data[data['label'].isin(['hateful'])]))

#Upsampling - test
''' 
positives = data[data['label'].isin(['hateful'])]

for x in range(10):
    data = pd.concat([data, positives])'''
#Training and evaluating the classifier

print('Training the classifier...')
#X = list(data['label'].apply(lambda x: 1 if x == 'hateful' else 0))
#y = list(data['tweet_vector'])
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

y_train = list(data['label'].apply(lambda x: 1 if x == 'hateful' else 0))
X_train = list(data['tweet_vector'])

y_test = list(test_data['label'].apply(lambda x: 1 if x == 'hateful' else 0))
X_test = list(test_data['tweet_vector'])


#CLASSIFIER
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
print('Fitted the classifier. Predicting test set...')
predictions_test = [classifier.predict_proba(x.reshape(1, -1)) for x in X_test]
predictions_train = [classifier.predict_proba(x.reshape(1, -1)) for x in X_train]

preds = [classifier.predict(x.reshape(1, -1)) for x in X_train]
print("Recall - train:", metrics.recall_score(y_train, preds))
print("Precision - train:", metrics.precision_score(y_train, preds))
print('ROC AUC score - training:',metrics.roc_auc_score(y_train, preds))

print(classifier.classes_)
experimental_sample['lan {}'.format(str(classifier.classes_))] = predictions_test
experimental_sample.to_csv('results_lan_model.csv')
data.to_csv('preprocessed.csv')
