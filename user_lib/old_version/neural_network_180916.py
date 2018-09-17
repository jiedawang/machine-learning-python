# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from user_lib.check import check_type,check_limit,\
    check_index_match,check_items_match
import user_lib.data_prep as dp
import time
import user_lib.statistics as stats
import matplotlib.pyplot as plt
#from numba import jit,prange
import math
import json
#from numexpr import evaluate
#from concurrent.futures import ThreadPoolExecutor,as_completed
#import cupy as cp

#多层感知器MLP
#全连接前馈神经网络
#最基础的神经网络
class MultilayerPerceptron:
    '''\n  
    Note: 多层感知器(全连接前馈网络)，不支持离散特征输入
    
    Parameters
    ----------
    input_shape: 输入形状，tuple类型，默认(28,28)，默认值是用于mnist数据集的， 
                 一般分类问题->(features_n,)，
                 灰度图片->(height,width)， 彩色图片->(height,width,3)，
                 输入数据会根据input_shape自动展开，降为2维矩阵(samples_n,features_n)
    output_shape: 输出形状，tuple类型，默认(10,)，分类问题->(classes_n,)
    hidden_layers: 隐含层设置，tuple类型，默认(100,)，每一项代表一层，值代表神经元数量
    mode: 模式，'c'->分类，'r'->回归，默认'c'
    activation: 隐含层/输出层激活函数，str类型或tuple(str)类型，
                'sigm'->sigmoid函数，1./(1.+exp(-z))，值域:0~1
                'tanh'->双曲正切函数，(exp(z)-exp(-z))/(exp(z)+exp(-z))，值域:-1~1
                'relu'->线性整流函数，max(0,z)，值域:0~正无穷
                'soft'->柔性最大值，exp(z)/sum_j(exp(z))，值域0~1，输出层总和为1
                'none'->不作处理
                默认值'sigm'           
    cost: 代价函数，str类型，
          'mse'->均方误差，0.5*mean_i(sum_j((a-y)**2))，
          'ce'->交叉熵，mean_i(sum_j(y*ln(a)+(1-y)*ln(1-a)))，
          'log'->对数似然，mean_i(sum_j(y*ln(a)))
    optimizer: 优化器，str类型，
               'sgd'->随机梯度下降，最基础的优化方案，可以和其他一些策略结合用于搜索最优解
               'magd'->动量加速梯度下降，引入动量项，能够更快地收敛
               'nagd'->nesterov加速梯度下降，引入动量修正，比magd更稳定
               'adam'->自适应矩估计，一种自适应学习率的算法， 通常有着很好的效果
    batch_size: mini-batch大小，即每次优化使用的样本量，int类型(>0),
                目前的机制是将训练集按batch_size随机分割后使用每个子集进行优化，
                使用完所有mini-batch算做一次迭代，然后重复
    iter_max: 迭代优化轮数上限，int类型(>0)，默认值100
    learning_rate: 迭代优化学习率，float类型(>0.0)，默认值1.0
    L2_alpha: L2正则化系数，float类型(>0.0)，默认值为0.0001，
          该系数用于平衡代价中关于模型复杂度的惩罚项，用于防止过拟合
    dropout_p: 神经元弃权概率，float类型(>=0.0,<1.0)，默认0.0
    early_stop: 用于提前结束的代价没有进一步下降的迭代次数上限，int类型(>0)，默认10
    lr_atten_max: 学习率衰减次数上限，int类型(>0)，默认10
    momentum_p: 动量因子，float类型(>=0.0,<=1.0)，控制历史梯度的利用率，
                仅对magd/nagd优化器有效，默认0.9
    adam_beta1: adam优化器参数1，float(>0.0,<1.0)，控制历史梯度的利用率，默认0.9
    adam_beta2: adam优化器参数2，float(>0.0,<1.0)，控制历史梯度平方的利用率，默认0.999
    adam_eps: adam优化器参数3，float(>0.0)，防止除零错误的容错因子，默认1e-8
    relu_a: relu激活函数参数，float(>=0.0,<=1.0)，自变量小于0时函数的系数，
            =0.0时是常规relu，=1.0是线性函数，默认0.0
    ----------
    
    Attributes
    ----------
    input_size: 输入层大小，int类型(>0)
    output_size: 输出层大小，int类型(>0)
    weights: 权重，list(ndarray(input_size,output_size)<float64>)类型
    biases: 偏置，list(ndarray(output_size,)<float64>)类型
    classes: 分类标签，list<str>类型
    time_cost: 训练耗时，Series类型
    cost_h: 历史代价，DataFrame类型，fit时需设置monitor_cost=True
    score_h: 历史评分，DataFrame类型，fit时需设置monitor_score=True
    iter_total: 总训练迭代次数，int类型
    ----------
    '''
    
    def __init__(self,input_shape=(28,28),output_shape=(10,),hidden_layers=(100,),
                 mode='c',activation='sigm',cost='mse',optimizer='sgd',
                 batch_size=256,iter_max=100,learning_rate=0.1,
                 L2_alpha=0.0001,dropout_p=0.0,
                 early_stop=10,lr_atten_rate=0.9,lr_atten_max=10,
                 momentum_p=0.9,adam_beta1=0.9,adam_beta2=0.999,adam_eps=1e-8,
                 relu_a=0.0):
        check_type('input_shape',type(input_shape),type(()))
        check_type('output_shape',type(output_shape),type(()))
        check_type('hidden_layers',type(hidden_layers),type(()))
        check_type('mode',type(mode),type(''))
        check_type('batch_size',type(batch_size),type(0))
        check_type('iter_max',type(iter_max),type(0))
        check_type('learning_rate',type(learning_rate),type(0.0))
        check_type('L2_alpha',type(L2_alpha),type(0.0))
        check_type('dropout_p',type(dropout_p),type(0.0))
        if type(early_stop)==type(True):
            if early_stop==True:
                early_stop=20
            else:
                early_stop=iter_max
        check_type('early_stop',type(early_stop),type(0))
        check_type('lr_atten_rate',type(lr_atten_rate),type(0.0))
        check_type('lr_atten_max',type(lr_atten_max),type(0))
        check_type('momentum_p',type(momentum_p),type(0.0))
        check_type('adam_beta1',type(adam_beta1),type(0.0))
        check_type('adam_beta2',type(adam_beta2),type(0.0))
        check_type('adam_eps',type(adam_eps),type(0.0))
        check_type('relu_a',type(relu_a),type(0.0))
        for item in input_shape:
            check_type('item in input_shape',type(item),type(0))
        for item in input_shape:
            check_type('item in output_shape',type(item),type(0))
        for item in input_shape:
            check_type('item in hidden_layers',type(item),type(0))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        check_limit('batch_size',batch_size>0,'value>0')
        check_limit('iter_max',iter_max>0,'value>0')
        check_limit('learning_rate',learning_rate>0.0,'value>0.0')
        check_limit('L2_alpha',L2_alpha>=0.0,'value>=0.0')
        check_limit('dropout_p',(dropout_p>=0.)&(dropout_p<1.),'0.<=value<1.')
        check_limit('early_stop',early_stop>0,'value>0')
        check_limit('lr_atten_rate',(lr_atten_rate>0.)&(lr_atten_rate<=1.),'0.<value<=1.')
        check_limit('lr_atten_max',lr_atten_max>=0,'value>=0')
        check_limit('momentum_p',(momentum_p>0.)&(momentum_p<1.),'0<value<1')
        check_limit('adam_beta1',(adam_beta1>0.)&(adam_beta1<1.),'0<value<1')
        check_limit('adam_beta2',(adam_beta2>0.)&(adam_beta2<1.),'0<value<1')
        check_limit('adam_eps',adam_eps>0.,'value>0')
        check_limit('relu_a',(relu_a>=0.)&(relu_a<=1.),'0<=value<=1')
        self.mode=mode
        if mode=='r':
            print('\nwarning: output_shape has been changed to (1,)')
            output_shape=(1,)
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.hidden_layers=hidden_layers
        self.input_size=int(np.array(input_shape).prod())
        self.output_size=int(np.array(output_shape).prod())
        self.layers=(self.input_size,)+hidden_layers+(self.output_size,)
        self.bind_func_(activation,cost,optimizer)
        self.random_init_()
        self.batch_size=batch_size
        self.iter_max=iter_max
        self.learning_rate=learning_rate
        self.L2_alpha=L2_alpha
        self.dropout_p=dropout_p
        self.early_stop=early_stop
        self.lr_atten_rate=lr_atten_rate
        self.lr_atten_max=lr_atten_max
        self.momentum_p=momentum_p
        self.adam_beta1=adam_beta1
        self.adam_beta2=adam_beta2
        self.adam_eps=adam_eps
        self.relu_a=relu_a
        self.time_cost=pd.Series(
                np.zeros(9),index=['Total','input check','mini batch','forward prop',
                        'back prop','--cost','--grad','--delta','monitor'],name='time_cost') 
        self.classes=[]
        self.iter_total=0
     
    #绑定函数
    #注：在激活函数/代价函数/优化器有多种选择时用于根据参数配置绑定函数，
    #    一开始实现时可以略去该部分
    def bind_func_(self,activation,cost,optimizer):
        #参数类型校验
        check_type('activation',type(activation),[type(''),type(())])
        check_type('cost',type(cost),type(''))
        check_type('optimizer',type(optimizer),type(''))
        #参数值校验
        activation_list=['sigm','tanh','relu','soft','none']
        layers=self.layers
        activation_config=[]
        if type(activation)==type(''):
            check_limit('activation',activation in activation_list,str(activation_list))
            for i in range(len(layers)-1):
                activation_config.append(activation)
        else:
            for item in activation:
                check_limit('item in activation',item in activation_list,str(activation_list))
            #将激活函数设置分配到各个层
            if len(activation)==1:
                for i in range(len(layers)-1):
                    activation_config.append(activation[0])
            elif len(activation)==2:
                for i in range(len(layers)-2):
                    activation_config.append(activation[0])
                activation_config.append(activation[1])
            elif len(activation)==len(layers)-1:
                for i in range(len(layers)-1):
                    activation_config.append(activation[i])
            else:
                raise ValueError('\nconfig error: layers do not match')
        if (activation_config[-1] in ('tanh','relu','none'))&(self.mode=='c'):
            print('\nwarning: you set tanh/relu/none activation of output for classification')
        if (activation_config[-1] in ('sigm','soft'))&(self.mode=='r'):
            print('\nwarning: you set sigmoid/softmax activation of output for regression')
        cost_list=['mse','ce','log']
        check_limit('cost',cost in cost_list,str(cost_list))
        if (cost in ['ce','log'])&(self.mode=='r'):
            print('\nwarning: you set cross entropy/log like cost for regression')
        optimizer_list=['sgd','magd','nagd','adam']
        check_limit('optimizer',optimizer in optimizer_list,str(optimizer_list))
        #绑定函数
        activations_=[]
        for i in range(len(activation_config)):
            if activation_config[i]=='sigm':
                activations_.append(self.sigmoid_)
            elif activation_config[i]=='tanh':
                activations_.append(self.tanh_)
            elif activation_config[i]=='relu':
                activations_.append(self.relu_)
            elif activation_config[i]=='soft':
                activations_.append(self.softmax_)
            elif activation_config[i]=='none':
                activations_.append(self.identity_)
            else:
                raise ValueError('Unknown activation function')
        self.activation=activation_config
        self.activations_=activations_
        if cost=='mse':
            self.cost_=self.mean_sqr_err_
        elif cost=='ce':
            self.cost_=self.cross_ent_
        elif cost=='log':
            self.cost_=self.log_like_
        else:
            raise ValueError('Unknown cost function')
        self.cost=cost
        if optimizer=='sgd':
            self.optimizer_=self.sgd_
        elif optimizer=='magd':
            self.optimizer_=self.momentum_
        elif optimizer=='nagd':
            self.optimizer_=self.nesterov_
        elif optimizer=='adam':
            self.optimizer_=self.adam_
        else:
            raise ValueError('Unknown optimizer')
        self.optimizer=optimizer
    
    #随机初始化权重和偏置        
    def random_init_(self):
        layers=self.layers
        w,b=[],[]
        for i in range(len(layers)-1):
            w_shape=(layers[i],layers[i+1])
            b_shape=layers[i+1]
            #w不能初始化为0，会导致输入对后方神经元无影响，
            #反向传播时梯度全部一致，无法正常优化
            #w初始化为均值为 0 标准差为 1/sqr(input_n)
            #的高斯随机分布可以加快收敛，有时也能带来性能的提升
            w_=np.random.randn(w_shape[0],w_shape[1])/np.sqrt(w_shape[0])
            b_=np.zeros(b_shape)
            w.append(w_)
            b.append(b_)
        self.weights,self.biases=w,b
    
    #重置
    def reset(self):
        self.random_init_()
        self.time_cost=pd.Series(
                np.zeros(9),index=['Total','input check','mini batch','forward prop',
                        'back prop','--cost','--grad','--delta','monitor'],name='time_cost') 
        self.classes=[]
        self.iter_total=0
    
    #激活函数：恒等式
    #相当于不使用激活函数
    def identity_(self,z,grad=False):
        if grad==False:
            return z
        else:
            return z,np.ones_like(z)
    
    #激活函数：sigmoid  
    #逻辑函数，又称S型生长曲线
    #通过设置grad=True可以返回梯度信息
    def sigmoid_(self,z,grad=False):
        a=1./(1.+np.exp(-1.*z))
        if grad==False:
            return a
        else:
            return a,a*(1-a)
    
    #激活函数：tanh
    #双曲正切函数
    def tanh_(self,z,grad=False):
        e=np.exp(z)
        a=(e-1./e)/(e+1./e)
        if grad==False:
            return a
        else:
            return a,1-a*a
    
    #激活函数：ReLU
    #线性整流函数，又称修正线性单元
    def relu_(self,z,grad=False):
        a=z.copy()
        a[a<0.]*=self.relu_a
        if grad==False:
            return a
        else:
            g=np.zeros_like(a)
            g[a>0.]=1.
            g[a<=0.]=self.relu_a
            return a,g
    
    #l2惩罚项
    def l2_penalty_(self):
        c=0.
        for w_ in self.weights:
            c+=0.5*self.L2_alpha*np.dot(w_.ravel(),w_.ravel())
        return c
    
    #代价函数：均方误差
    def mean_sqr_err_(self,a,Y,grad=False):
        re=a-Y
        c=0.5*(re*re).sum()/Y.shape[0]
        if grad==False:
            return c
        else:
            return c,re
    
    #代价函数：交叉熵
    #注：此处设置了一个容错因子，防止计算log(0)溢出
    def cross_ent_(self,a,Y,grad=False):
        tol=1e-10
        c=-(Y*np.log(a+tol)+(1-Y)*np.log(1-a+tol)).sum()/Y.shape[0]
        if grad==False:
            return c
        else:
            return c,(a-Y)/(a+tol)/(1-a+tol)

    #前向传播
    #a为输入神经元激活值，shape=(sample_n,input_size)
    #return_al=True时，返回所有层的激活值和梯度信息
    def forward_prop_(self,a,return_al=False):
        if return_al==False:
            for w_,b_,activation_ in zip(
                    self.weights,self.biases,self.activations_):
                a=activation_(np.dot(a,w_)+b_)
            return a
        else:
            al,al_grad=[a],[]
            layer_idx=1
            for w_,b_,activation_ in zip(
                    self.weights,self.biases,self.activations_):
                a,a_grad=activation_(np.dot(a,w_)+b_,grad=True)
                #dropout
                #原本需要在预测时每层激活值乘上p，此处将该操作提前了，
                #训练时每层激活值除以了p进行缩放
                if (self.dropout_p>0)&(layer_idx<len(self.layers)-1):
                    mark=np.random.rand(a.shape[1])<self.dropout_p
                    a*=(mark/self.dropout_p)
                    a_grad*=(mark/self.dropout_p)
                al.append(a)
                al_grad.append(a_grad)
                layer_idx+=1
            return al,al_grad
            
    #反向传播
    #反向逐层计算w和b的梯度
    def back_prop_(self,al,al_grad,Y):
        delta_w,delta_b=[],[]
        start=time.clock()
        cost,cost_grad=self.cost_(al[-1],Y,grad=True)
        self.time_cost['--cost']+=time.clock()-start
        for i in range(len(al)-1,0,-1):
            start=time.clock()
            if i==len(al)-1:
                #输出层的计算不一样
                #delta_z=(al[i]-Y)*al[i]*(1-al[i])
                delta_z=cost_grad*al_grad[i-1]
            else:
                delta_z=np.dot(delta_z,self.weights[i].T)*al_grad[i-1]
            self.time_cost['--grad']+=time.clock()-start
            start=time.clock()
            delta_w.append(np.dot(al[i-1].T,delta_z)/delta_z.shape[0])
            delta_b.append(delta_z.mean(axis=0))
            self.time_cost['--delta']+=time.clock()-start
        delta_w.reverse()
        delta_b.reverse()
        return delta_w,delta_b
    
    #将分类概率转换为分类标签
    def prob_to_label_(self,prob,classes):
        prob_max=np.argmax(prob.T,axis=0)
        if len(classes)>0:
            if type(prob_max)==type(np.array(1)):
                classes=np.array(classes)
                output=np.full(len(prob_max),'').astype(classes.dtype)
                for i in range(len(classes)):
                    output[prob_max==i]=classes[i]
            else:
                output=classes[prob_max]
        else:
            output=prob_max
        return output
    
    #计算代价
    def compute_cost_(self,X,Y):
        output_a=self.predict(X,return_a=True,check_input=False)
        cost=self.cost_(output_a,Y)+self.l2_penalty_()
        return cost
    
    #计算评分
    def compute_score_(self,X,y):
        pred_y=self.predict(X,check_input=False)
        score=self.assess(y,pred_y,check_input=False)
        return score
    
    #优化方案：随机梯度下降
    def sgd_(self,X_,Y_):
        #前向传播计算全部神经元的激活值和激活函数梯度信息
        start=time.clock()
        al,al_grad=self.forward_prop_(X_,return_al=True)
        self.time_cost['forward prop']+=time.clock()-start
        #反向传播计算全部权重和偏置的更新量
        start=time.clock()
        delta_w,delta_b=self.back_prop_(al,al_grad,Y_)
        self.time_cost['back prop']+=time.clock()-start
        #考虑学习率和L2正则化更新权重和偏置
        for j in range(len(delta_b)):
            delta_w[j]+=self.L2_alpha*self.weights[j]
            self.weights[j]-=self.learning_rate*delta_w[j]
            self.biases[j]-=self.learning_rate*delta_b[j]

    #优化方案：动量梯度下降
    #该算法在更新时考虑上一次的更新量，使用动量因子p控制比重
    #相当于增加了惯性，而动量因子等同于摩擦力带来的速度衰减率
    def momentum_(self,X_,Y_):
        #初始化
        if hasattr(self, 'v_w')==False:
            self.v_w,self.v_b=[],[]
            for j in range(len(self.weights)): 
                self.v_w.append(np.zeros_like(self.weights[j]))
                self.v_b.append(np.zeros_like(self.biases[j]))
        #前向传播计算全部神经元的激活值和激活函数梯度信息
        start=time.clock()
        al,al_grad=self.forward_prop_(X_,return_al=True)
        self.time_cost['forward prop']+=time.clock()-start
        #反向传播计算全部权重和偏置的更新量
        start=time.clock()
        delta_w,delta_b=self.back_prop_(al,al_grad,Y_)
        self.time_cost['back prop']+=time.clock()-start
        #考虑学习率和L2正则化更新权重和偏置
        for j in range(len(self.weights)):
            delta_w[j]+=self.L2_alpha*self.weights[j]
            self.v_w[j]=self.momentum_p*self.v_w[j]+self.learning_rate*delta_w[j]
            self.v_b[j]=self.momentum_p*self.v_b[j]+self.learning_rate*delta_b[j]
            self.weights[j]-=self.v_w[j]
            self.biases[j]-=self.v_b[j]
    
    #优化方案：nesterov加速梯度下降
    #在动量梯度下降基础上根据动量项修正当前梯度的计算
    #相当于预估了累积速度会将自己带到哪个位置，再计算该位置的梯度
    #从另一个角度看，也相当于先根据累积速度更新，再根据梯度对前进方向做一个修正
    def nesterov_(self,X_,Y_):
        #初始化
        if hasattr(self, 'v_w')==False:
            self.v_w,self.v_b=[],[]
            for j in range(len(self.weights)): 
                self.v_w.append(np.zeros_like(self.weights[j]))
                self.v_b.append(np.zeros_like(self.biases[j]))
        else:
            #根据动量项修正当前权重和偏置
            for j in range(len(self.weights)):
                self.weights[j]-=self.momentum_p*self.v_w[j]
                self.biases[j]-=self.momentum_p*self.v_b[j]
        #前向传播计算全部神经元的激活值和激活函数梯度信息
        start=time.clock()
        al,al_grad=self.forward_prop_(X_,return_al=True)
        self.time_cost['forward prop']+=time.clock()-start
        #反向传播计算全部权重和偏置的更新量
        start=time.clock()
        delta_w,delta_b=self.back_prop_(al,al_grad,Y_)
        self.time_cost['back prop']+=time.clock()-start
        #考虑学习率和L2正则化更新权重和偏置
        for j in range(len(self.weights)):
            delta_w[j]+=self.L2_alpha*self.weights[j]
            self.v_w[j]=self.momentum_p*self.v_w[j]+self.learning_rate*delta_w[j]
            self.v_b[j]=self.momentum_p*self.v_b[j]+self.learning_rate*delta_b[j]
            self.weights[j]-=self.learning_rate*delta_w[j]
            self.biases[j]-=self.learning_rate*delta_b[j]
    
    #优化方案：自适应矩估计
    #本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
    #Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
    #作者建议beta1取默认值为0.9，beta2为0.999，eps为10−8。
    #从经验上表明Adam在实际中表现很好，同时，与其他的自适应学习算法相比，其更有优势。
    #注：此段解释摘自https://blog.csdn.net/u012759136/article/details/52302426/
    #    其实对照一下其他算法更好理解，
    #    AdaGrad对不同参数应用不同的学习率，更新量的分母是所有历史梯度平方的和，
    #    即越频繁更新的参数分母会越大，学习率会变小，防止更新过头，
    #    更新不频繁的参数的学习率则会变大，加速更新，
    #    但有个问题是无限制的累积会导致分母整体都越来越大，学习率最后变得过小导致更新困难
    #    RMSprop对此进行了改进，历史梯度平方的记忆有一个衰减率，这样就不会无限制地累积了
    #    这实际上是在求一个历史梯度的加权平均数，权重随时间衰减
    #    Adam在RMSprop的基础上又引入了动量项，即更新量的分子和momentum是差不多的，
    #    除此之外还引入了偏差校正，导致一开始更新量过小
    def adam_(self,X_,Y_):
        #初始化
        if hasattr(self, 'v_w')==False:
            self.t,self.m_w,self.m_b,self.v_w,self.v_b=1,[],[],[],[]
            for j in range(len(self.weights)): 
                self.m_w.append(np.zeros_like(self.weights[j]))
                self.m_b.append(np.zeros_like(self.biases[j]))
                self.v_w.append(np.zeros_like(self.weights[j]))
                self.v_b.append(np.zeros_like(self.biases[j]))
        #前向传播计算全部神经元的激活值和激活函数梯度信息
        start=time.clock()
        al,al_grad=self.forward_prop_(X_,return_al=True)
        self.time_cost['forward prop']+=time.clock()-start
        #反向传播计算全部权重和偏置的更新量
        start=time.clock()
        delta_w,delta_b=self.back_prop_(al,al_grad,Y_)
        self.time_cost['back prop']+=time.clock()-start
        #考虑学习率和L2正则化更新权重和偏置
        for j in range(len(self.weights)):
            delta_w[j]+=self.L2_alpha*self.weights[j]
            self.m_w[j]=self.adam_beta1*self.m_w[j]+(1.-self.adam_beta1)*delta_w[j]
            self.m_b[j]=self.adam_beta1*self.m_b[j]+(1.-self.adam_beta1)*delta_b[j]
            self.v_w[j]=self.adam_beta2*self.v_w[j]+(1.-self.adam_beta2)*(delta_w[j]**2)
            self.v_b[j]=self.adam_beta2*self.v_b[j]+(1.-self.adam_beta2)*(delta_b[j]**2)
            #一开始m和v会初始化为0向量，而beta1和beta2趋近于1，
            #因此从计算中可以看出，当前梯度只会以(1-beta)的比率进行利用,
            #在迭代初期，没有足量的历史梯度累积，m和v的值会很小
            #通过计算偏差校正的一阶矩和二阶矩估计来抵消偏差
            m_w_=self.m_w[j]/(1-self.adam_beta1**self.t)
            m_b_=self.m_b[j]/(1-self.adam_beta1**self.t)
            v_w_=self.v_w[j]/(1-self.adam_beta2**self.t)
            v_b_=self.v_b[j]/(1-self.adam_beta2**self.t)
            self.weights[j]-=self.learning_rate/(np.sqrt(v_w_)+self.adam_eps)*m_w_
            self.biases[j]-=self.learning_rate/(np.sqrt(v_b_)+self.adam_eps)*m_b_
            self.t+=1

    #优化
    #注：external_monitor是外部监视方法
    #   在使用其他程序调用算法并希望每次迭代都能反馈信息时通过该参数传入一个外部方法
    #   每次迭代后都会调用一次该方法，并将反馈信息通过参数传入方法
    def optimize_(self,X,Y,test_X=None,test_Y=None,
             monitor_cost=False,monitor_score=False,external_monitor=None):
        #是否有测试数据/训练数据集大小/小批量子集大小/子集数量
        if type(test_X)!=type(None):
            test=True
        else:
            test=False
        self.samples_n=len(X)
        batch_size=self.batch_size
        batches_n=math.ceil(self.samples_n/batch_size)
        #y转换
        if self.mode=='c':
            y=self.prob_to_label_(Y,self.classes)
        else:
            y=Y.reshape((-1,))
        if test==True:
            if self.mode=='c':
                test_y=self.prob_to_label_(test_Y,self.classes)
            else:
                test_y=test_Y.reshape((-1,))
        #计算初始的代价和评分，需要设置monitor
        if monitor_cost==True:
            cost1=self.compute_cost_(X,Y)
            if test==True:
                cost2=self.compute_cost_(test_X,test_Y)
                cost_h=[[cost1,cost2]]
            else:
                cost_h=[[cost1]]
        else:
            cost_h=[]
        if monitor_score==True:
            score1=self.compute_score_(X,y)
            if test==True:
                score2=self.compute_score_(test_X,test_y)
                score_h,test_score_best=[[score1,score2]],score2
            else:
                score_h=[[score1]]
        else:
            score_h=[]
        #学习率衰减次数/测试集无提升次数/代价连续上升次数
        lr_atten,test_no_improve,cost_keep_rise=0,0,0
        #迭代优化
        for i in range(self.iter_max):
            message=''
            #随机排序数据集
            start=time.clock()
            random_idx=np.random.permutation(self.samples_n)
            X,Y,y=X[random_idx],Y[random_idx],y[random_idx]
            self.time_cost['mini batch']+=time.clock()-start
            #使用每个小批量子集进行更新
            for j in range(batches_n):
                #按batch_size抽取小批量子集
                start=time.clock()
                X_,Y_=X[j*batch_size:(j+1)*batch_size,:],Y[j*batch_size:(j+1)*batch_size,:]
                self.time_cost['mini batch']+=time.clock()-start
                #应用优化方案
                self.optimizer_(X_,Y_)
            print('\niter: %d'%(i+1))
            message+='\niter: %d'%(i+1)
            #对当前网络进行评估
            start=time.clock()
            if monitor_cost==True:
                cost1=self.compute_cost_(X,Y)
                if test==True:
                    cost2=self.compute_cost_(test_X,test_Y)
                    cost_h.append([cost1,cost2])
                    print('train cost: %f test cost: %f'%(cost1,cost2))
                    message+='\ntrain cost: %f\ntest cost: %f'%(cost1,cost2)
                else:
                    cost_h.append([cost1])
                    print('train cost: %f'%cost1)
                    message+='\ntrain cost: %f'%cost1
                #记录代价连续上升的迭代次数
                if cost_h[-1][0]>cost_h[-2][0]:
                    cost_keep_rise+=1
                else:
                    cost_keep_rise=0
            if monitor_score==True:
                score1=self.compute_score_(X,y)
                if test==True:
                    score2=self.compute_score_(test_X,test_y)
                    score_h.append([score1,score2])
                    print('train score: %f test score: %f'%(score1,score2))
                    message+='\ntrain score: %f\ntest score: %f'%(score1,score2)
                    #记录测试集表现没有提升的迭代次数
                    if score2<test_score_best:
                        test_no_improve+=1
                        print('(test score has not been improved for %d iters)'%test_no_improve)
                        message+='\n(test score has not been improved for %d iters)'%test_no_improve
                    else:
                        test_no_improve=0
                        test_score_best=score2
                else:
                    score_h.append([score1])
                    print('train score: %f'%score1)
                    message+='\ntrain score: %f'%score1
            self.time_cost['monitor']+=time.clock()-start
            #提前结束迭代和学习率衰减
            if monitor_cost==True:
                #学习率太大，代价计算溢出，提前结束迭代
                if (cost1==0.)|(cost1==np.inf):
                    print('\nwarning: early stopping due to too large learning rate')
                    message+='\nwarning: early stopping due to too large learning rate'
                    break
                #代价上升，且衰减次数未达上限，学习率衰减,
                #代价连续上升，衰减强度会增加
                if (lr_atten<self.lr_atten_max)&(cost_keep_rise>=1):
                    self.learning_rate*=(self.lr_atten_rate**cost_keep_rise)
                    lr_atten+=1
                    print('\nwarning: learning rate is attenuated to %f'%self.learning_rate)
                    message+='\nwarning: learning rate is attenuated to %f'%self.learning_rate
                if (lr_atten==self.lr_atten_max)&(cost_keep_rise>=1):
                    print('\nwarning: learning rate attenuation has reached the limit')
                    message+='\nwarning: learning rate attenuation has reached the limit'
            if (monitor_score==True)&(test==True):
                #测试集表现没有提升的迭代次数超设定阈值，提前结束迭代
                if test_no_improve>=self.early_stop:
                    print('\nwarning: early stopping by %d no improve iters limit'%self.early_stop)
                    message+='\nwarning: early stopping by %d no improve iters limit'%self.early_stop
                    break
            if type(external_monitor)!=type(None):
                external_monitor((message,score_h,cost_h))
        #记录迭代次数
        self.iter_total+=(i+1)
        #cost_h,score_h转换为DataFrame,方便查看和快速绘图
        if test==True:
            self.cost_h=pd.DataFrame(cost_h,columns=['train','test'])
            self.score_h=pd.DataFrame(score_h,columns=['train','test'])
        else:
            self.cost_h=pd.DataFrame(cost_h,columns=['train'])
            self.score_h=pd.DataFrame(score_h,columns=['train'])
    
    #X输入校验
    #X shape=(sample_n,input_shape)或(sample_n,input_size)
    def check_input_X_(self,X,name='X'):
        #类型校验
        check_type(name,type(X),type(np.array(0)))
        type_list=[np.int64,np.float64]
        check_type(name,X.dtype,type_list)
        #shape调整
        X_D,input_D=len(X.shape),len(self.input_shape)
        input_size=self.input_size
        error_info='The shape of '+name+' does not match to input_shape'
        #(m,k)->(1,m*k)
        if X_D==input_D:
            if X.shape==self.input_shape:
                X=X.reshape((1,input_size))
            else:
                raise ValueError(error_info)
        #(n,m,k)->(n,m*k)
        elif X_D==1+input_D:
            if X.shape[1:]==self.input_shape:
                X=X.reshape((-1,input_size))
            else:
                raise ValueError(error_info)
        else:
            raise ValueError(error_info)
        return X
    
    #y输入校验
    def check_input_y_(self,y,name='y',transform=True):
        #类型校验
        check_type(name,type(y),type(np.array(0)))
        if self.mode=='c':
            y=y.astype('str')
        if transform==False:
            return y
        #对目标向量进行one-hot编码
        #(单列离散变量转换为多列01变量)
        if self.mode=='c':
            if len(y.shape)==1:
                Y,classes=dp.dummy_var(y)
            else:
                Y,classes=y,[i for i in range(y.shape[1])]
        else:
            Y,classes=y.reshape((-1,1)),[]
        #shape调整
        Y_D,output_D=len(Y.shape),len(self.output_shape)
        output_size=self.output_size
        error_info='The shape of '+name+' does not match to output_shape'
        #(m,k)->(1,m*k)
        if Y_D==output_D:
            if Y.shape==self.output_shape:
                Y=Y.reshape((1,output_size))
            else:
                raise ValueError(error_info)
        #(n,m,k)->(n,m*k)
        elif Y_D==1+output_D:
            if Y.shape[1:]==self.output_shape:
                Y=Y.reshape((-1,output_size))
            else:
                raise ValueError(error_info)
        else:
            raise ValueError(error_info)
        return Y,classes
    
    #拟合
    def fit(self,X,y,test_X=None,test_y=None,show_time=False,
            monitor_cost=False,monitor_score=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合神经网络
        
        Note: 输入数据必须全部是连续数值类型，其他类型自行预处理
        
        Parameters
        ----------
        X: 特征矩阵,ndarray(samples_n,input_shape)<float64,int64>类型
        y: 目标向量,ndarray(samples_n,)<str,float64,int64>类型
        test_X: 测试特征矩阵,ndarray(samples_n,input_shape)<float64,int64>类型
        test_y: 测试目标向量,ndarray(samples_n,)<str,float64,int64>类型
        show_time: 是否显示时间开销，bool类型，默认False
        monitor_cost: 监控cost变化，bool类型，默认值False
        monitor_score: 监控score变化，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        '''
        start=time.clock()
        #输入校验
        start1=time.clock()
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('show_time',type(show_time),type(True))
            check_type('monitor_cost',type(monitor_cost),type(True))
            check_type('monitor_score',type(monitor_score),type(True))
            X=self.check_input_X_(X)
            y,self.classes=self.check_input_y_(y)
            if (len(self.classes)<2)&(self.mode=='c'):
                raise ValueError('too few classes,should >1')
            check_index_match(X,y,'X','y',only_len=True)
            if type(test_X)!=type(None):
                test_X=self.check_input_X_(test_X,name='test_X')
                test_y,test_classes=self.check_input_y_(test_y,name='test_y')
                check_index_match(test_X,test_y,'test_X','test_y',only_len=True)
        self.time_cost['input check']+=time.clock()-start1
        #优化
        self.optimize_(X,y,test_X,test_y,monitor_cost,monitor_score)
        if show_time==True:
            print('\ntime used for training: %f'%(time.clock()-start))
        self.time_cost['Total']+=time.clock()-start
        
    #预测
    def predict(self,X,return_a=False,show_time=False,check_input=True):
        '''\n
        Function: 对输入数据进行预测
        
        Parameters
        ----------
        X: 特征矩阵,ndarray类型
        return_a: 是否返回输出层激活值，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测值向量，ndarray类型
        -------
        '''
        start=time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('show_time',type(show_time),type(True))
            check_type('return_a',type(return_a),type(True))
            X=self.check_input_X_(X)
        #前向传播
        a=self.forward_prop_(X)
        #整合结果
        if self.mode=='c':
            if return_a==False:
                output=self.prob_to_label_(a,self.classes)
            else:
                output=a.reshape((-1,)+self.output_shape)
        else:
            if return_a==False:
                output=a.reshape((-1,))
            else:
                output=a
        if show_time==True:
            print('\ntime used for predict: %f'%(time.clock()-start))
        if output.shape[0]==1:
            output=output[0]
        return output
    
    #评估
    def assess(self,y,pred_y,return_dist=False,check_input=True):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y: 观测值，ndarray类型
        pred_y: 预测值，ndarray类型
        return_dist: 是否返回预测分布，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('return_dist',type(return_dist),type(True))
            y=self.check_input_y_(y,'y',transform=False)
            pred_y=self.check_input_y_(pred_y,'pred_y',transform=False)
            check_index_match(y,pred_y,'y','pred_y',only_len=True)
        if self.mode=='c':
            return stats.accuracy(y,pred_y,return_dist,self.classes)
        else:
            return stats.r_sqr(y,pred_y)
    
    #保存
    def save(self,file_path):
        '''\n
        Function: 保存模型
        
        Notes: 以json格式保存参数
        
        Parameters
        ----------
        file_path: 文件路径，str类型
        ----------
        '''
        data = {"input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "hidden_layers": self.hidden_layers,
                "mode": self.mode,
                "activation": self.activation,
                "cost": self.cost,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "iter_max": self.iter_max,
                "learning_rate": self.learning_rate,
                "L2_alpha": self.L2_alpha,
                "dropout_p": self.dropout_p,
                "early_stop": self.early_stop,
                "lr_atten_rate": self.lr_atten_rate,
                "lr_atten_max": self.lr_atten_max,
                "momentum_p": self.momentum_p,
                "adam_beta1": self.adam_beta1,
                "adam_beta2": self.adam_beta2,
                "adam_eps": self.adam_eps,
                "relu_a": self.relu_a,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "classes": self.classes,
                "input_size": self.input_size,
                "output_size": self.output_size,}
        f = open(file_path, "w")
        json.dump(data, f)
        f.close()
    
    #加载
    def load(self,file_path):
        '''\n
        Function: 加载模型
        
        Notes: 只加载预测用参数，完整加载请使用外部方法
        
        Parameters
        ----------
        file_path: 文件路径，str类型
        ----------
        '''
        f = open(file_path, "r")
        data = json.load(f)
        f.close()
        self.mode=data['mode']
        self.input_shape=tuple(data['input_shape'])
        self.output_shape=tuple(data['output_shape'])
        self.hidden_layers=tuple(data['hidden_layers'])
        self.bind_func_(tuple(data['activation']),data['cost'],data['optimizer'])
        self.weights=[np.array(w) for w in data['weights']]
        self.biases=[np.array(b) for b in data['biases']]
        self.classes=data['classes']
        self.input_size=int(data['input_size'])
        self.output_size=int(data['output_size'])
        
    #神经网络结构可视化
 
    #绘制一层神经元
    def plot_layer_(self,ax,weights,biases,layer,height,show_max,
                    layer_offset,neuron_offset,connect_offset,first):
        #当前层神经元数量/前一层神经元数量
        nn,pnn=weights.shape[1],weights.shape[0]
        #屏蔽大部分神经元的显示
        unshow,unshow_p=0,0
        show_max=show_max
        if nn>show_max:
            unshow=nn-show_max
            nn=show_max+1
        if pnn>show_max:
            unshow_p=pnn-show_max
            pnn=show_max+1
        if height>show_max:
            height=show_max+1
        #计算偏移(x是横向轴，y是纵向轴，左下角是原点)
        x_off=layer*layer_offset
        y_off=(height-nn)/2
        y_off_p=(height-pnn)/2
        #绘制第一层隐含层前先绘制输入层
        if first==True:
            for i in range(pnn):
                self.plot_neuron_(ax,None,1,y_off_p,
                                  (x_off-layer_offset,neuron_offset*(y_off_p+pnn-i-1)),
                                  layer_offset,neuron_offset,connect_offset)
                if (i==show_max/2)&(pnn==show_max+1):
                    ax.text(x_off-layer_offset,neuron_offset*(y_off_p+pnn-i-1),
                            str(unshow_p),va="center",ha="center")
        #遍历该层神经元
        if pnn==show_max+1:
            weights=np.r_[weights[:int(show_max/2+1)],weights[-int(show_max/2):]]
        if nn==show_max+1:
            weights=np.c_[weights[:,:int(show_max/2+1)],weights[:,-int(show_max/2):]]
            biases=np.r_[biases[:int(show_max/2+1)],biases[-int(show_max/2):]]
            biases[int(show_max/2)]=1
        weights=np.abs(weights)
        weights=weights/weights.max()
        biases=np.abs(biases)
        if biases.max()==0:
            biases[:]=1
        else:
            biases=biases/biases.max()
        for j in range(nn):
            self.plot_neuron_(ax,weights[:,j],biases[j],y_off_p,
                              (x_off,neuron_offset*(y_off+nn-j-1)),
                              layer_offset,neuron_offset,connect_offset)
            if (j==show_max/2)&(nn==show_max+1):
                ax.text(x_off,neuron_offset*(y_off+nn-j-1),str(unshow),va="center",ha="center",fontsize=14)
    
    #绘制一个神经元           
    def plot_neuron_(self,ax,w,b,y_off_p,xy,layer_offset,neuron_offset,connect_offset,text='    '):
        #绘制该神经元
        style_neuron = dict(boxstyle="circle", color='white', ec='black',lw=1) 
        ax.annotate(text,xy=(0,0),xycoords='axes fraction',
                    xytext=(xy[0],xy[1]),
                    textcoords='axes fraction',va="center",ha="center",
                    bbox=style_neuron,fontsize=15)
        #绘制该神经元的所有输入连接
        if type(w)!=type(None):
            pnn=w.shape[0]
            for i in range(pnn):
                style_connect = dict(arrowstyle="<-", color='black',lw=0.2+w[i])
                ax.annotate('',xy=(xy[0]-layer_offset+connect_offset,neuron_offset*(y_off_p+pnn-i-1)), 
                             xycoords='axes fraction',
                             xytext=(xy[0]-connect_offset,xy[1]),
                             textcoords='axes fraction',
                             va="center",ha="center",arrowprops=style_connect)
    
    #绘制神经网络
    def plot_network(self,return_fig=False):
        layers=self.layers
        height=max(layers)
        layer_offset=0.95
        neuron_offset=0.2
        connect_offset=0.08
        show_max=10
        fig=plt.figure(figsize=(3,4))
        axprops=dict(xticks=[], yticks=[])
        ax=fig.add_subplot(111,frameon=False,**axprops)
        for i in range(len(self.weights)):
            if i==0:
                self.plot_layer_(ax,self.weights[i],self.biases[i],i,height,show_max,
                                 layer_offset,neuron_offset,connect_offset,first=True)
            else:
                self.plot_layer_(ax,self.weights[i],self.biases[i],i,height,show_max,
                                 layer_offset,neuron_offset,connect_offset,first=False)
        if return_fig==False:
            plt.show()
        else:
            plt.close()
            return fig,ax

#加载文件并返回一个model    
def load(file_path):
    '''\n
    Function: 加载模型
        
    Parameters
    ----------
    file_path: 文件路径，str类型
    ----------
    
    Returns
    -------
    0: 加载的模型，MultilayerPerceptron类型
    -------
    '''
    f = open(file_path, "r")
    data = json.load(f)
    f.close()
    model=MultilayerPerceptron(input_shape=tuple(data['input_shape']),
                               output_shape=tuple(data['output_shape']),
                               hidden_layers=tuple(data['hidden_layers']),
                               mode=data['mode'],
                               activation=tuple(data['activation']),
                               cost=data['cost'],
                               optimizer=data['optimizer'],
                               batch_size=data['batch_size'],
                               iter_max=data['iter_max'],
                               learning_rate=data['learning_rate'],
                               L2_alpha=data['L2_alpha'],
                               dropout_p=data['dropout_p'],
                               early_stop=data['early_stop'],
                               lr_atten_rate=data['lr_atten_rate'],
                               lr_atten_max=data['lr_atten_max'],
                               momentum_p=data['momentum_p'],
                               adam_beta1=data['adam_beta1'],
                               adam_beta2=data['adam_beta2'],
                               adam_eps=data['adam_eps'],
                               relu_a=data['relu_a'])
    model.weights=[np.array(w) for w in data['weights']]
    model.biases=[np.array(b) for b in data['biases']]
    model.classes=data['classes']
    return model