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
# print(movies_df.head())
# print(rating_df.shape)
df = pd.merge(rating_df,movies_df,on='movieId')
#print(df.head())
combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
#print(combine_movie_rating)
temp1 = combine_movie_rating.groupby(by=['title'])
#print(temp1['rating'].count().reset_index())

movie_ratingCount = (combine_movie_rating.groupby(by=['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])
#print(movie_ratingCount.head())
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
print(rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())
popularity_threshold = 50
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
print(rating_popular_movie.head())
print(rating_popular_movie.shape)
movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
print(movie_features_df.head())

from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)
#print(movie_features_df_matrix)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)
print(movie_features_df.shape)
query_index = np.random.choice(movie_features_df.shape[0]) # thay doi bang user_id khi su dung vao bai toan baygolf
print('index:',query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
print(movie_features_df.head())
print(distances)
print('indices:', indices)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
