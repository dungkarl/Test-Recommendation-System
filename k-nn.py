"""
    test recommendation system based rating
"""
import pandas as pd
import numpy as np


movies_df = pd.read_csv('movies.csv', usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
# temp = pd.read_csv('movies.csv')
# print(temp.head())
# print(temp.shape)
rating_df=pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
print(movies_df.head())
print(rating_df.shape)
df = pd.merge(rating_df,movies_df,on='movieId')
print(df.head())
