# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.regression as rg
import user_lib.data_prep as dp

#简单数据集，分类结果的图像展示
#注：x取值范围过大可能会导致simgond计算超精度上限，视情况进行缩放
f = open('D:\\training_data\\used\\simple_data1.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

x=buf.iloc[:,:2]
y=buf.iloc[:,2]

logR=rg.LogisticRegression()
dp_tool=dp.DataPreprocessing()

x=dp_tool.minmax_scaler(x)

init_dir={'a':3}
gd_type='SGD'
iter_max=500

theta=logR.fit(x,y,init_dir,iter_max,gd_type)
theta_h=logR.theta_h
cost_h=logR.cost_h

x1_r=pd.Series(np.linspace(x['x1'].min(),x['x1'].max(),101))
x2_r=-(theta[0]+theta[1]*x1_r)/theta[2]

print('\nscore:%f'%logR.score)
plt.scatter(x['x1'][y==1],x['x2'][y==1],c='b',marker='o')
plt.scatter(x['x1'][y==0],x['x2'][y==0],c='r',marker='x')
plt.plot(x1_r,x2_r,c='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(labels = ['bound','y=1', 'y=0'],loc='best')
plt.show()

#非线性
f = open('D:\\training_data\\used\\simple_data2.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

x=buf.iloc[:,:2]
y=buf.iloc[:,2]

logR=rg.LogisticRegression()
dp_tool=dp.DataPreprocessing()

#x=dp_tool.minmax_scaler(x)
x=dp_tool.feature_mapping(x,6,cross=True)

init_dir={'a':3}
gd_type='SGD'
iter_max=500

theta=logR.fit(x,y,init_dir,iter_max,gd_type)
theta_h=logR.theta_h
cost_h=logR.cost_h

x_r=pd.DataFrame()
x1_r,x2_r=np.mgrid[
        x['x1'].min():x['x1'].max():101j,
        x['x2'].min():x['x2'].max():101j
        ]
x_r['x1']=x1_r.reshape(101*101)
x_r['x2']=x2_r.reshape(101*101)
x_r=dp_tool.feature_mapping(x_r,6,cross=True)
y_r=pd.DataFrame(logR.model(dp.DataPreprocessing.fill_x0(x_r),
                            theta).reshape(101,101))

print('\nscore:%f'%logR.score)
plt.scatter(x['x1'][y==1],x['x2'][y==1],c='b',marker='o')
plt.scatter(x['x1'][y==0],x['x2'][y==0],c='r',marker='x')
c=plt.contour(x1_r[:,0],x2_r[0,:],y_r,1, colors='black')
plt.clabel(c, inline=1, fontsize=10)  
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(labels = ['y=1', 'y=0'],loc='best')
plt.show()

#钞票数据集：类1为假钞，0为真钞
f = open('D:\\training_data\\used\\bill.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['小波变换图像','小波偏斜变换图像','小波峰度变换图像','图像熵','类']
describe=buf.describe()

train=buf.sample(frac=0.8,random_state=1)
test=buf[~buf.index.isin(train.index)]

x=train.iloc[:,:4]
y=train.iloc[:,4]

logR=rg.LogisticRegression()
dp_tool=dp.DataPreprocessing()

dp_tool.scaler_reference(x)
x=dp_tool.minmax_scaler(x)

k=len(x.columns)+1
init_dir={'a':10}
gd_type='SGD'
iter_max=200

theta=logR.fit(x,y,init_dir,iter_max,gd_type)
theta_h=logR.theta_h
cost_h=logR.cost_h

cost_h.plot()
plt.show()
print('\nuser train score:%f'%logR.score)

test_x=test.iloc[:,:4]
test_y=test.iloc[:,4]
test_x=dp_tool.minmax_scaler(test_x)
test_p,test_score,test_a_result=logR.predict_test(test_x,test_y)
print('\nuser test score:%f'%test_score)

#与sklearn对照
from sklearn.linear_model import LogisticRegression
#建模
#sklearn默认设置是使用liblinear库进行优化，内部使用了坐标轴下降法
#设置solver='sag'可以改为随机梯度下降
lgModel = LogisticRegression()
#训练模型
lgModel.fit(x, y)
#评分
print('\nsklearn train score:%f'%lgModel.score(x, y))
#查看参数
lgModel.coef_
#查看截距
lgModel.intercept_
#预测
sk_test_p=lgModel.predict(test_x)
sk_test_score=lgModel.score(test_x, test_y)
print('\nsklearn test score:%f'%sk_test_score)

#迭代次数与拟合程度关系
logR2=rg.LogisticRegression()

iter_max=0
iter_add=10
add_times=60
iter_max_list=[]
score_list=[]

print('\n\n<iter_max/score>')
for i in range(add_times):
    iter_max+=iter_add
    print('iter_max:%d'%(iter_max))
    logR2.fit(x,y,init_dir,iter_max,gd_type)
    temp_score=logR2.predict_test(test_x,test_y)[1]
    score_list.append([logR2.score,temp_score])
    iter_max_list.append(iter_max)

plt.xlabel('iter')
plt.ylabel('score')
plt.plot(iter_max_list,score_list)
plt.legend(labels = ['train', 'test'],loc='best')

#多分类
#one vs rest
#小麦种子数据集
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

train=buf.sample(frac=0.8,random_state=1)
test=buf[~buf.index.isin(train.index)]

x=train.iloc[:,:7]
y=train.iloc[:,7]

logRmuti=rg.LogisticRegression()
dp_tool=dp.DataPreprocessing()

dp_tool.scaler_reference(x)
x=dp_tool.minmax_scaler(x)

k=len(x.columns)+1
init_dir={'a':10}
gd_type='SGD'
iter_max=200

theta=logRmuti.fit(x,y,init_dir,iter_max,gd_type)

p,score,pred_dist=logRmuti.predict_test(x,y)
print('\nuser train score:%f'%score)

test_x=test.iloc[:,:7]
test_y=test.iloc[:,7]
test_x=dp_tool.minmax_scaler(test_x)
test_p,test_score,test_pred_dist=logRmuti.predict_test(test_x,test_y)
print('\nuser test score:%f'%test_score)
print('\nclassificaion:')
print(logRmuti.classificaion)
print('\npredict distribution:')
print(logRmuti.pred_dist)