# -*- coding: utf-8 -*-
import user_lib.bayes as by
import pandas as pd
import numpy as np
import user_lib.data_prep as dp

#小麦种子数据集
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

#分割训练集和测试集
X1,y1,test_X1,test_y1=dp.split_train_test(buf)
X1=X1.round(4)
#生成离散化区间
rg=dp.discret_reference(X1,10)
#进行特征离散化
X1_=dp.discret(X1,rg)
test_X1_=dp.discret(test_X1,rg)

#拟合朴素贝叶斯分类器
#朴素贝叶斯只能分类，且不能处理连续特征
nb=by.NaiveBayes()
nb.fit(X1_,y1)
pred_y1=nb.predict(X1_)
score1=nb.access(y1,pred_y1)
print('\nuser train score:%f'%score1)
pred_y1=nb.predict(test_X1_)
score1=nb.access(test_y1,pred_y1)
print('\nuser test score:%f'%score1)
