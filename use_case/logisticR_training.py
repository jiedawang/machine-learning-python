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

X1=buf.iloc[:,:2]
y1=buf.iloc[:,2]
ref1=dp.scaler_reference(X1)
X1_=dp.minmax_scaler(X1,ref1)

logR1=rg.LogisticRegression(learning_rate=1.0,iter_max=100)

theta1=logR1.fit(X1_,y1,output=True,show_time=True)
theta_h1=logR1.theta_h
cost_h1=logR1.cost_h
theta1=theta1.iloc[:,0]

x1_r=pd.Series(np.linspace(X1_['x1'].min(),X1_['x1'].max(),101))
x2_r=-(theta1[0]+theta1[1]*x1_r)/theta1[2]

print('\nscore:%f'%logR1.score)
plt.scatter(X1_['x1'][y1==1],X1_['x2'][y1==1],c='b',marker='o')
plt.scatter(X1_['x1'][y1==0],X1_['x2'][y1==0],c='r',marker='x')
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

X2=buf.iloc[:,:2]
y2=buf.iloc[:,2]
#X2_=dp.minmax_scaler(X2)
X2_=dp.feature_mapping(X2,6,cross=True)

logR2=rg.LogisticRegression(learning_rate=1.0,iter_max=100)

theta2=logR2.fit(X2_,y2,output=True,show_time=True)
theta_h2=logR2.theta_h
cost_h2=logR2.cost_h
theta2=theta2.iloc[:,0]

x_r=pd.DataFrame()
x1_r,x2_r=np.mgrid[
        X2_['x1'].min():X2_['x1'].max():101j,
        X2_['x2'].min():X2_['x2'].max():101j
        ]
x_r['x1']=x1_r.reshape(101*101)
x_r['x2']=x2_r.reshape(101*101)
x_r=dp.feature_mapping(x_r,6,cross=True)
y_r=logR2.predict(x_r,return_proba=True)
y_r=pd.DataFrame(y_r.iloc[:,1].values.reshape(101,101))

print('\nscore:%f'%logR2.score)
plt.scatter(X2_['x1'][y2==1],X2_['x2'][y2==1],c='b',marker='o')
plt.scatter(X2_['x1'][y2==0],X2_['x2'][y2==0],c='r',marker='x')
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

X3,y3,test_X3,test_y3=dp.split_train_test(buf)

ref3=dp.scaler_reference(X3)
X3_=dp.minmax_scaler(X3,ref3)
test_X3_=dp.minmax_scaler(test_X3,ref3)

logR3=rg.LogisticRegression(learning_rate=1.0,iter_max=1000)

theta3=logR3.fit(X3_,y3,output=True,show_time=True)
theta_h3=logR3.theta_h
cost_h3=logR3.cost_h

cost_h3[0].plot()
plt.show()
print('\nuser train score:%f'%logR3.score)

result3=logR3.predict(test_X3_)
result3_2=logR3.predict(test_X3_,return_proba=True)
score3,dist3=logR3.assess(test_y3,result3,return_dist=True)
print('\nuser test score:%f'%score3)

#与sklearn对照
from sklearn.linear_model import LogisticRegression
#建模
#sklearn默认设置是使用liblinear库进行优化，内部使用了坐标轴下降法
#设置solver='sag'可以改为随机梯度下降
lgModel = LogisticRegression()
#训练模型
lgModel.fit(X3_, y3)
#评分
print('\nsklearn train score:%f'%lgModel.score(X3_, y3))
#查看参数
lgModel.coef_
#查看截距
lgModel.intercept_
#预测
sk_test_p=lgModel.predict(test_X3_)
sk_test_score=lgModel.score(test_X3_, test_y3)
print('\nsklearn test score:%f'%sk_test_score)

#迭代次数与拟合程度关系
logR4=rg.LogisticRegression(learning_rate=1.0)

iter_max,iter_add,add_times=0,20,50
iter_max_list,score_list=[],[]

print('\n\n<the relation of iter_max and score>')
for i in range(add_times):
    iter_max+=iter_add
    logR4.iter_max=iter_max
    print('\niter_max:%d'%(iter_max))
    logR4.fit(X3_,y3)
    temp_p_y=logR4.predict(test_X3_)
    temp_score=logR4.assess(test_y3,temp_p_y)
    score_list.append([logR4.score,temp_score])
    iter_max_list.append(iter_max)

plt.xlabel('iter')
plt.ylabel('score')
plt.plot(iter_max_list,score_list)
plt.legend(labels = ['train', 'test'],loc='best')
plt.show()

#多分类
#one vs rest/one vs one 两种模式测试
#小麦种子数据集
f = open('D:\\training_data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()

X4,y4,test_X4,test_y4=dp.split_train_test(buf)

ref4=dp.scaler_reference(X4)
X4_=dp.minmax_scaler(X4,ref4)
test_X4_=dp.minmax_scaler(test_X4,ref4)

iter_max,learning_rate=1000,1.0
multi_class_list=['ovo','ovr']

for multi_class in multi_class_list:

    print('\n<multi_class: '+multi_class+'>')
    logR4=rg.LogisticRegression(learning_rate=learning_rate,
                                iter_max=iter_max,multi_class=multi_class)
    
    theta4=logR4.fit(X4_,y4,output=True,show_time=True)
    theta_h4=logR4.theta_h
    cost_h4=logR4.cost_h
    
    print('\nuser train score:%f'%logR4.score)
    
    result4=logR4.predict(test_X4_)
    score4,dist4=logR4.assess(test_y4,result4,return_dist=True)
    print('\nuser test score:%f'%score4)
    print('\nclasses:')
    print(logR4.classes)
    print('\npredict distribution:')
    print(dist4)