# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.recommend as rc
import time
from sklearn import datasets

#简单数据
X0=np.array([[4,0,3,0],[0,2,0,0],[3,0,3,4],[5,0,0,0]]).astype('float64')

#电影评分数据
#下载地址：https://grouplens.org/datasets/movielens/
filedir='D:\\training_data\\used\\ml-latest-small\\'
links=pd.read_csv(open(filedir+'links.csv'))
movies=pd.read_csv(open(filedir+'movies.csv'))
ratings=pd.read_csv(open(filedir+'ratings.csv'))
tags=pd.read_csv(open(filedir+'tags.csv'))

#重塑评论数据
ratings_=ratings.pivot(index='userId',columns='movieId',values='rating')
ratings_=ratings_.fillna(0)
movies_=movies[movies['movieId'].isin(ratings_.columns)]

#协同过滤
config=[('item','cosine'),('item','distance'),
        ('user','cosine'),('user','distance')]
R_list=[]
for i in range(len(config)):
    cf0=rc.CollaborativeFiltering(mode=config[i][0],similarity=config[i][1])
    R0=cf0.fit_predict(X0)
    R_list.append(R0)

cf1_1=rc.CollaborativeFiltering(mode='item',similarity='distance')
R1_1=cf1_1.fit_predict(ratings_.values)
r1_1=cf1_1.fit_predict(ratings_.values,ratings_.values[0])
r1_1_=cf1_1.recommend_list(r1_1,movies_.values,to_list=True)

cf1_2=rc.CollaborativeFiltering(mode='user',similarity='distance')
R1_2=cf1_2.fit_predict(ratings_.values)
r1_2=cf1_2.fit_predict(ratings_.values,ratings_.values[0])
r1_2_=cf1_2.recommend_list(r1_2,movies_.values,n=10,to_list=True)