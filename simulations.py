# General import and load data
import pandas as pd
import numpy as np
import os
import io

from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

# Algorithms
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import SlopeOne
from surprise import CoClustering
from surprise import NMF

""" --------------------------------------------------------------------------------------------------------------
	GENERAL FUNCTIONS
-------------------------------------------------------------------------------------------------------------- """

def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0

""" --------------------------------------------------------------------------------------------------------------
	DOWNLOADING & CREATING DATASETS
-------------------------------------------------------------------------------------------------------------- """

path_moody_lyrics1 = os.path.expanduser('~/Google Drive/TELECO/tfg/inc/MoodyLyrics4Q/MoodyLyrics4Q.csv')
path_moody_lyrics2 = os.path.expanduser('~/Google Drive/TELECO/tfg/inc/MoodyLyrics/ml_raw.xlsx')
path_million_song  = os.path.expanduser('~/Google Drive/TELECO/tfg/inc/song_data.csv')
path_triplets_msd  = os.path.expanduser('~/Google Drive/TELECO/tfg/inc/million-song-triplets.txt')

# MSD triplets
song_df_1 = pd.read_table(path_triplets_msd)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

# MSD song data
song_df_2 = pd.read_csv(path_million_song)

# MSD merge: users + song data
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
song_df["song"] = song_df["title"] + " - " + song_df["artist_name"]

song_df_3 = pd.read_csv(path_moody_lyrics1)
song_df_4 = pd.read_excel(path_moody_lyrics2, header=15).drop(['Unnamed: 4', 'Unnamed: 5'], axis=1)

# MoodyLyrics Concatenation
song_df_4.rename(columns={"Artist":"artist","Index":"index","Mood":"mood","Title":"title"}, inplace=True)
moodylyrics_df = pd.concat([song_df_3, song_df_4])
moodylyrics_df = moodylyrics_df.drop_duplicates("title")

# final dataframe merge
song_df_test = pd.merge(song_df, song_df_3, on="title", how="left")
# drop rows with NaN
song_df_test = song_df_test.dropna()
# drop columns with repeated data
song_df_test = song_df_test.drop(["index","artist"], axis=1)
song_df_test.insert(3,'rating1', 0)
song_df_test.insert(4,'rating2', 0)
song_df_test.insert(5,'rating3', 0)

# User and Song lists
users = song_df_test['user_id'].unique()
songs = song_df_test['song_id'].unique()

# Cropped dataset for recommender system
song_df_test.drop(song_df_test[song_df_test["year"] == 0].index, inplace=True)
song_df_test.drop(song_df_test[song_df_test["user_id"].isin(users[:35000])].index, inplace=True)
# 5000 total users
print("Users in dataset: " + str(song_df_test["user_id"].unique().size))

# Create a Frame for Users and the total listenings of each one
users_listenings = song_df_test.groupby(['user_id'], as_index=False).agg('listen_count')
users_data = users_listenings.sum()
users_data.rename(columns={'listen_count': 'sum_listen_count'}, inplace=True)

song_df_test = pd.merge(song_df_test, users_data, on="user_id")

# Calculating the Ratings

total_listenings = song_df_test["listen_count"].sum()
listenings_per_user = song_df_test["listen_count"].sum()/song_df_test["user_id"].unique().size

song_df_test['rating1'] = song_df_test["listen_count"]/total_listenings
song_df_test['rating2'] = song_df_test["listen_count"]/listenings_per_user
song_df_test['rating3'] = song_df_test["listen_count"]/song_df_test["sum_listen_count"]

print("Rating 1 mean: " + str(song_df_test['rating1'].mean()))
print("Rating 2 mean: " + str(song_df_test['rating2'].mean()))
print("Rating 3 mean: " + str(song_df_test['rating3'].mean()))

# Passing values to Int
#song_df_test["rating"] = song_df_test["rating"].loc[song_df_test["rating"]<10]*100
#song_df_test = song_df_test.round({'rating' : 0})

song_grouped = song_df_test.groupby(['song', 'mood']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['rating'] = song_grouped['listen_count'].div(grouped_sum)*10000
song_info = song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])

""" --------------------------------------------------------------------------------------------------------------
	RECOMENDATION ENGINE
-------------------------------------------------------------------------------------------------------------- """

""" -------------------------------------------------
	RATING 1
-------------------------------------------------- """

reader = Reader(line_format = 'user item rating',rating_scale=(0, 1))
data = Dataset.load_from_df(song_df_test[['user_id', 'song_id', 'rating1']], reader)

benchmark1 = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), NMF() ,NormalPredictor() ,BaselineOnly() ,CoClustering() ,SlopeOne()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark1.append(tmp)

####### KNN Basic #######
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark1.append(tmp)

####### KNN With Means #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithMeans(sim_options=sim_options)
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark1.append(tmp)

####### KNN With Z Score #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithZScore(sim_options=sim_options)
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark1.append(tmp)

####### KNN Pearson Baseline #######
sim_options = {'name': 'pearson_baseline',
               'user_based': True  # compute  similarities between items
               }
algo = KNNBaseline(sim_options=sim_options)
results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark1.append(tmp)

pd.DataFrame(benchmark1).set_index('Algorithm').sort_values('test_rmse')

""" -------------------------------------------------
	RATING 2
-------------------------------------------------- """

reader2 = Reader(line_format = 'user item rating',rating_scale=(0, 20))
data2 = Dataset.load_from_df(song_df_test[['user_id', 'song_id', 'rating2']], reader2)

benchmark2 = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), NMF() ,NormalPredictor() ,BaselineOnly() ,CoClustering() ,SlopeOne()]:
    # Perform cross validation
    results = cross_validate(algorithm, data2, measures=['RMSE'], cv=3, verbose=False)
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark2.append(tmp)

####### KNN Basic #######
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
results = cross_validate(algo, data2, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark2.append(tmp)

####### KNN With Means #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithMeans(sim_options=sim_options)
results = cross_validate(algo, data2, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark2.append(tmp)

####### KNN With Z Score #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithZScore(sim_options=sim_options)
results = cross_validate(algo, data2, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark2.append(tmp)

####### KNN Pearson Baseline #######
sim_options = {'name': 'pearson_baseline',
               'user_based': True  # compute  similarities between items
               }
algo = KNNBaseline(sim_options=sim_options)
results = cross_validate(algo, data2, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark2.append(tmp)

pd.DataFrame(benchmark2).set_index('Algorithm').sort_values('test_rmse')

""" -------------------------------------------------
	RATING 3
-------------------------------------------------- """

reader3 = Reader(line_format = 'user item rating',rating_scale=(0, 1))
data3 = Dataset.load_from_df(song_df_test[['user_id', 'song_id', 'rating2']], reader3)

benchmark3 = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), NMF() ,NormalPredictor() ,BaselineOnly() ,CoClustering() ,SlopeOne()]:
    # Perform cross validation
    results = cross_validate(algorithm, data3, measures=['RMSE'], cv=3, verbose=False)
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark3.append(tmp)

####### KNN Basic #######
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
results = cross_validate(algo, data3, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark3.append(tmp)

####### KNN With Means #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithMeans(sim_options=sim_options)
results = cross_validate(algo, data3, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark3.append(tmp)

####### KNN With Z Score #######
sim_options = {'name': 'cosine',
               'user_based': True  # compute  similarities between items
               }
algo = KNNWithZScore(sim_options=sim_options)
results = cross_validate(algo, data3, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark3.append(tmp)

####### KNN Pearson Baseline #######
sim_options = {'name': 'pearson_baseline',
               'user_based': True  # compute  similarities between items
               }
algo = KNNBaseline(sim_options=sim_options)
results = cross_validate(algo, data3, measures=['RMSE'], cv=3, verbose=False)

tmp = pd.DataFrame.from_dict(results).mean(axis=0)
tmp = tmp.append(pd.Series([str(algo).split(' ')[0].split('.')[-1]], index=['Algorithm']))
benchmark3.append(tmp)

pd.DataFrame(benchmark3).set_index('Algorithm').sort_values('test_rmse')
