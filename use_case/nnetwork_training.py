# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import user_lib.neural_network as nn
from user_lib.image_data import MnistManager,CifarManager
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neural_network import MLPClassifier
import time
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.datasets import make_moons
import user_lib.data_prep as dp

#1.非线性可分简单数据集(圆形)
f = open('D:\\training_data\\used\\simple_data2.txt')
buf = pd.read_table(f,header=None,sep=',')
buf.columns=['x1','x2','y']
describe=buf.describe()

X=buf.iloc[:,:2].values
y=buf.iloc[:,2].values

#2.非线性可分自动生成数据(半月形)
X,y=make_moons(n_samples=1000,noise=0.3)

#MLP训练
mlp0=nn.MultilayerPerceptron(
        hidden_layers=(8,8),activation='relu',softmax=False,
        optimizer='nagd',batch_size=200,iter_max=100,learning_rate=0.1,
        l2_alpha=0.0001,dropout_p=0.,early_stop=10,relu_a=0.25)
mlp0.fit(X,y,monitor_cost=True,monitor_score=True,show_time=True)

#X值域
x1_min,x1_max=X[:,0].min(),X[:,0].max()
x2_min,x2_max=X[:,1].min(),X[:,1].max()

#生成网格数据，用于绘制等高线图或曲面图
X0=np.zeros((101*101,2))
X0_1,X0_2=np.mgrid[
        x1_min-0.1:x1_max+0.1:101j,
        x2_min-0.1:x2_max+0.1:101j
        ]
X0[:,0]=X0_1.reshape(101*101)
X0[:,1]=X0_2.reshape(101*101)
    
#设置图像大小
plt.figure(figsize=(6,6),dpi=80)

#模型预测等高线图
y0=mlp0.predict(X0,return_a=True)
y0=y0.reshape(101,101,2)
plt.contourf(X0_1,X0_2,y0[:,:,1],cmap=plt.cm.RdYlBu) 

#数据集散点图
plt.scatter(X[:,0][y==1],X[:,1][y==1],c='b',edgecolors='black',label='y=1')
plt.scatter(X[:,0][y==0],X[:,1][y==0],c='r',edgecolors='black',label='y=0')
plt.legend()

plt.xlabel('x1')
plt.ylabel('x2')
plt.suptitle('Fitting result',fontsize=14,y=0.96)
plt.show()

#神经元作用的曲面图
def plot_surface(fig,X,Y,Z,nrows=1,ncols=1,index=1,title='',elev=None,azim=None):
    
    axes1=fig.add_subplot(nrows,ncols,index,projection='3d')
    axes1.set_title(title,fontsize=12)
    
    #拟合曲面图
    axes1.plot_surface(X,Y,Z,color='lightgrey',alpha=1.0)

    #axes1.set_xlabel('x1')
    #axes1.set_ylabel('x2')
    #axes1.set_zlabel('a')
    axes1.set_xticks([])
    axes1.set_yticks([])
    axes1.set_zticks([])
    
    axes1.view_init(elev=elev, azim=azim)

#绘制所有神经元的作用曲面图
def plot_neurons_action(model,X):
    x1,x2=X[:,0].reshape(101,101),X[:,1].reshape(101,101)
    al,al_grad=model.forward_prop_(X,return_al=True)
    
    for k in range(1,len(al)):
        size=al[k].shape[1]
        row=math.ceil(size/4)
        subplot_idx=[]
        for i in range(1,row+1):
            for j in range(1,5):
                idx=4*(i-1)+j
                if idx>size:
                    break
                if row==1:
                    subplot_idx.append((1,size,idx))
                else:
                    subplot_idx.append((row,4,idx))
        
        fig=plt.figure(figsize=(7,2.0+1.0*row),dpi=100)
        
        a=al[k].reshape((101,101,size))
        
        for rol,col,idx in subplot_idx:
            plot_surface(fig,x1,x2,a[:,:,idx-1],rol,col,idx,
                         'neuron %d'%idx,elev=20, azim=40)
       
        if k==len(al)-1:
            plt.suptitle('[Output Layer]\n(x,y,z)=(x1,x2,a)',fontsize=14,y=1.12)
        else:
            plt.suptitle('[Hidden Layer %d]\n(x,y,z)=(x1,x2,a)'%k,fontsize=14,y=1.10)
        plt.tight_layout()
        plt.show()
 
plot_neurons_action(mlp0,X0)

#图像识别

#读取mnist手写数字数据集
mnist=MnistManager()
#mnist.to_images(rewrite=True)
start=time.clock()
train_images,train_labels,test_images,test_labels=mnist.read_as_array()
print('\ntime used for loading: %f'%(time.clock()-start))

#MLP训练
#此处测试两种优化器: nesterov,adam
mlps=[]
mlps.append(nn.MultilayerPerceptron(
        hidden_layers=(100,),activation='relu',softmax=False,
        optimizer='nagd',batch_size=100,iter_max=100,learning_rate=0.1,
        l2_alpha=1e-4,dropout_p=0.,early_stop=10))
mlps.append(nn.MultilayerPerceptron(
        hidden_layers=(100,),activation='relu',softmax=False,
        optimizer='adam',batch_size=100,iter_max=100,learning_rate=0.001,
        l2_alpha=1e-4,dropout_p=0.,early_stop=10))

mlps[0].fit(train_images/255,train_labels,test_images/255,test_labels,
            monitor_cost=True,monitor_score=True,show_time=True)
#mlp0.fit(train_images/255,train_labels,monitor_cost=True,show_time=True)
mlps[1].fit(train_images/255,train_labels,test_images/255,test_labels,
            monitor_cost=True,monitor_score=True,show_time=True)

#耗时细节
mlp0=mlps[0]
mlp0.time_cost

#cost下降曲线
mlp0.plot_cost_h()

#score上升曲线
mlp0.plot_score_h()

#预测测试
#a=mlp0.predict(test_images/255,return_a=True)
pred=mlp0.predict(test_images/255,show_time=True)
score=mlp0.assess(test_labels,pred)
print('\nuser test score:%f'%score)

#单图预测
prob0=mlp0.predict(test_images[0]/255,return_a=True)
pred0=mlp0.prob_to_label_(prob0,mlp0.classes)

#平均图像
fig=plt.figure(figsize=(8,6))
for i in range(10):
    axes=plt.subplot(3,4,i+1)
    mean_image=train_images[train_labels==str(i)].mean(axis=0)
    mean_image=Image.fromarray(np.uint8(mean_image))
    axes.imshow(mean_image)
    axes.set_xticks([])
    axes.set_yticks([])
plt.show()

#保存加载测试
mlp0.save('D:\\Model\\mlp0.txt')
mlp0.load('D:\\Model\\mlp0.txt')
mlp0=nn.load('D:\\Model\\mlp0.txt')

#sklearn对照
#sklearn默认优化算法使用adam，激活函数使用ReLu
sk_mlp=MLPClassifier(hidden_layer_sizes=(100,),max_iter=100,verbose=True)
start=time.clock()
sk_mlp.fit(train_images.reshape((-1,28*28))/255,train_labels)
print('\ntime used for training: %f'%(time.clock()-start))
sk_pred=sk_mlp.predict(test_images.reshape((-1,28*28))/255)
sk_score=mlp0.assess(test_labels,sk_pred)
print('\nsklearn test score:%f'%sk_score)

#神经元关注图像（绿色正相关，红色负相关）
def w_plot(model,layer,w_p):
    if layer==0:
        print('input layer can not be visualized')
        return None
    elif layer==len(model.biases):
        layer_type='output'
    else:
        layer_type='hidden'
    row=math.ceil(model.weights[layer-1].shape[1]/4)
    plt.figure(figsize=(10,2.4*row))
    plt.suptitle('[weights of layer %d (%s)]'%(layer,layer_type),
                 fontsize=16,y=0.95-0.0025*row)
    if layer==1:
        w_p=model.weights[layer-1]-model.biases[layer-1]/(28*28)
    else:
        w_p=np.dot(w_p,model.weights[layer-1])-model.biases[layer-1]/(28*28)
    for i in range(model.weights[layer-1].shape[1]):
        w_=w_p[:,i].reshape(28,28)
        scalar=256/np.abs(w_).max()
        image=Image.new('RGB',(28,28))
        for y in range(28):
            for x in range(28):
                if w_[x,y]>0:
                    r,g=0,int(w_[x,y]*scalar)
                else:
                    r,g=int(-w_[x,y]*scalar),0
                image.putpixel((x,y),(r,g,0))
        #image.resize((28,28))
        plt.subplot(row,4,i+1)
        plt.imshow(image)
    plt.show()
    return w_p

#除第一层隐含层，其他层用这种方式计算出的图像可能并不正确
w_p=None
for i in range(len(mlp0.biases)):
    w_p=w_plot(mlp0,i+1,w_p)

'''
def predict(X,w):
    prob=np.dot(X,w)
    pred=mlp0.prob_to_label_(prob,mlp0.classes)
    return pred

prob=np.dot(test_images.reshape((-1,28*28)),w_p)
pred_=predict(test_images.reshape((-1,28*28)),w_p)
score_=mlp0.assess(test_labels,pred_)
print('\nuser test score:%f'%score_)
'''

#绘制结构图
mlp0.plot_network()

#加载cifar10数据集
cifar=CifarManager()
#cifar.to_images(rewrite=True,chinese_label=True)
start=time.clock()
train_images,train_labels,test_images,test_labels=\
    cifar.read_as_array(chinese_label=True)
print('\ntime used for loading: %f'%(time.clock()-start))

#MLP训练
mlp0=nn.MultilayerPerceptron(
        hidden_layers=(100,),activation='relu',softmax=False,
        optimizer='nagd',batch_size=100,iter_max=100,learning_rate=0.01,
        l2_alpha=0.0,dropout_p=0.0,early_stop=10)
'''
mlp0=nn.MultilayerPerceptron(
        hidden_layers=(100,),activation='relu',softmax=True,
        optimizer='nagd',batch_size=100,iter_max=100,learning_rate=0.01,
        l2_alpha=0.0,dropout_p=0.0,early_stop=10)
'''
mlp0.fit(train_images/255,train_labels,test_images/255,test_labels,
         monitor_cost=True,monitor_score=True,show_time=True)

#耗时细节
mlp0.time_cost

#cost下降曲线
mlp0.plot_cost_h()

#score上升曲线
mlp0.plot_score_h()

#保存和加载
mlp0.save('D:\\Model\\mlp0.txt')
mlp0.load('D:\\Model\\mlp0.txt')
mlp0=nn.load('D:\\Model\\mlp0.txt')

#预测测试
#a=mlp0.predict(test_images/255,return_a=True)
pred=mlp0.predict(test_images/255,show_time=True)
score=mlp0.assess(test_labels,pred)
print('\nuser test score:%f'%score)

#单图预测
prob0=mlp0.predict(test_images[0]/255,return_a=True)
pred0=mlp0.prob_to_label_(prob0,mlp0.classes)

#平均图像
mean_images=np.zeros((10,32,32,3))
labels=['飞机','汽车','鸟','猫','鹿',
        '狗','青蛙','马','船','卡车']
fig=plt.figure(figsize=(8,6))
for i in range(10):
    axes=plt.subplot(3,4,i+1)
    mean_image=train_images[train_labels==labels[i]].mean(axis=0)
    mean_images[i]=mean_image
    mean_image=Image.fromarray(np.uint8(mean_image))
    axes.imshow(mean_image)
    axes.set_xticks([])
    axes.set_yticks([])
plt.show()

probs1=mlp0.predict(mean_images/255,return_a=True)
preds1=mlp0.prob_to_label_(probs1,mlp0.classes)

#回归
#波士顿房价数据集
f = open('D:\\training_data\\used\\boston_house_price.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
describe=buf.describe()

X,y,test_X,test_y=dp.split_train_test(buf)

#归一化是必须的，不然很难优化
ref=dp.scaler_reference(X)
X_=dp.minmax_scaler(X,ref)
test_X_=dp.minmax_scaler(test_X,ref)

mlp1=nn.MultilayerPerceptron(
        mode='r',hidden_layers=(100,),activation='relu',softmax=False,
        optimizer='nagd',batch_size=205,iter_max=100,learning_rate=0.01,
        l2_alpha=0.0001,dropout_p=0.,early_stop=10,relu_a=0.0)
mlp1.fit(X_.values,y.values,test_X_.values,test_y.values,
         monitor_cost=True,monitor_score=True,show_time=True)

#绘制结构图
mlp1.plot_network()
#fig,ax=mlp1.plot_network(return_fig=True)

