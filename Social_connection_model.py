#Preprocessing part 3 and social connection model
import pandas as pd 
import numpy as np 
import os 
import scipy 
from functools import reduce 
from tqdm import tqdm
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics

tqdm.pandas()

def create_sparse_vector(x, users_followed):
    users_array = np.zeros(len(users_followed))
    for element in users_followed:
        if element in x:
            users_array[users_followed.index(element)] += 1
    return users_array

def ext(x, y):
    x.extend(y)
    return list(set(x))

def manual_eval(x):
    x = x[1:-1]
    elements = x.replace('\'', '')
    elements = elements.replace(' ', '')
    elements = elements.split(',')
    return elements

def count_up(x):
    global all_followed
    for follower in x: 
        all_followed[follower] += 1
    return x

def observes_anyone(x, y):
    for element in x:
        if element in y:
            return True
    return False

def reduce_dimensionality(X, y): #feature selector
    #Feature selection 
    def get_vector(x, selected_features):
        instance = []
        for feat_index in selected_features:
            instance.append(x[feat_index])
        return np.array(instance)
    
    importance_vector = list(mutual_info_classif(X, y))
    selected_features = importance_vector.copy()
    critical_importance = np.quantile(importance_vector, 0.8)
    for x in range(len(importance_vector)):
        if importance_vector[x] >= critical_importance:
            selected_features[x] = x
        else:
            selected_features[x] = False
            
    while True: 
        try:
            selected_features.remove(False)
        except:
            break
    X = pd.Series(X)
    X = X.apply(lambda x: get_vector(x, selected_features))
    return list(X)

os.chdir(r'PATH') #replace with your working directory
data = pd.read_csv('results_lan_model.csv')
#data = data[data['following'].notna() & data['label'].isin(['hateful', 'normal'])]
data.reset_index(inplace = True, drop = True)
data['following'] = data['following'].apply(lambda x: manual_eval(x))
all_followed = dict.fromkeys( list(reduce(lambda x, y: ext(x, y), data['following'])), 0)
data['following'] = data['following'].apply(lambda x: count_up(x))
all_followed = pd.DataFrame({"user":all_followed.keys(), "n_of_followers":all_followed.values()})
#all_followed.to_csv('follower_data.csv')
critical = np.quantile(all_followed['n_of_followers'], 0.999)
all_followed['include'] = all_followed['n_of_followers'].apply(lambda x: True if x >= critical else np.nan)
all_followed.dropna(inplace = True)
users_followed = list(all_followed['user'])
data['observes_anyone'] = data['following'].apply(lambda x: observes_anyone(x, users_followed))
data['following'] = data['following'].progress_apply(lambda x: create_sparse_vector(x, users_followed))

#Classifier - training and evaluating 
y = list(data['label'].apply(lambda x: 1 if x == 'hateful' else 0))
X = reduce_dimensionality(list(data['following']), y)
data['following'] = X

print() #This call has to stay here


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
#classifier = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')
#classifier = NearestCentroid()
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
predictions_train = [classifier.predict_proba(x.reshape(1, -1)) for x in X_train]
predictions_test = [classifier.predict_proba(x.reshape(1, -1)) for x in X_test]
    
preds = [classifier.predict(x.reshape(1, -1)) for x in X_train]
print('Precision - train: ',metrics.precision_score(y_train, preds))
print('Recall - train: ', metrics.recall_score(y_train, preds))
print('ROC AUC - train: ', metrics.roc_auc_score(y_train, preds))

print(classifier.classes_)
data['soc {}'.format(str(classifier.classes_))] = [classifier.predict_proba(x.reshape(1, -1)) for x in X]
data = data[data['following'].isin(X_test)]
data.to_csv('test_set.csv')
