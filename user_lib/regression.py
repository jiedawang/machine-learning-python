# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.statistics as stats
from user_lib.check import check_type,check_limit,check_index_match,check_feats_match
import time

#线性回归
class LinearRegression:
    '''\n  
    Note: 线性回归，只能用于回归，对于非线性问题的拟合需要进行特征映射
    
    Parameters
    ----------
    fit_mode: 拟合模式，str类型(ne,sgd),
              'ne'->正规方程法，直接求得最优解，在特征数量很多时速度会很慢
              'sgd'->随机梯度下降，迭代优化，没有限制但需要选择合适的a和iter_max
              默认值'ne'
    learning_rate: 迭代优化学习率，float类型(>0.0)，默认值0.001
    iter_max: 迭代优化迭代次数上限，int类型(>0)，默认值1000
    mini_batch: 迭代优化每次使用的样本数，int类型(>0)，0为自动选择，默认值0
    L2_n: L2正则化系数，float类型(>0.0)，默认值为0，
          该系数在代价中加入了关于模型复杂度的惩罚项，
          作用是防止过拟合，一般用于非线性模式
    early_stop: 是否允许超过一定次数迭代后代价没有进一步降低时提前结束训练，
                bool类型，默认值True
    ----------
    '''
    
    def __init__(self,fit_mode='ne',learning_rate=0.001,iter_max=1000,
                 mini_batch=256,L2_n=0.0,early_stop=True):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('fit_mode',type(fit_mode),type(''))
        check_type('learning_rate',type(learning_rate),type(0.0))
        check_type('iter_max',type(iter_max),type(0))
        check_type('mini_batch',type(mini_batch),type(0))
        check_type('L2_n',type(L2_n),type(0.0))
        check_type('early_stop',type(early_stop),type(True))
        fit_mode=fit_mode.lower()
        mode_list=['ne','sgd']
        check_limit('fit_mode',fit_mode in mode_list,str(mode_list))
        check_limit('learning_rate',learning_rate>0.0,'value>0.0')
        check_limit('iter_max',iter_max>0,'value>0')
        check_limit('mini_batch',mini_batch>=0,'value>=0')
        check_limit('L2_n',L2_n>=0.0,'value>=0.0')
        self.fit_mode=fit_mode
        self.learning_rate=learning_rate
        self.iter_max=iter_max
        self.mini_batch=mini_batch
        self.L2_n=L2_n
        self.early_stop=early_stop
        self.dp_tool=None
    
    #线性内核函数
    #fx=theta0*1+theta1*x1+theta2*x2+...
    def linear_(self,X,theta):
        '''
        return
        0: 矩阵相乘结果，即预测值向量，narray(m,1)类型
        '''
        return np.dot(X,theta)
    
    #代价函数,用于衡量拟合的偏差程度
    def cost_(self,y,p_y):
        '''
        return
        0: 平方误差，float类型
        '''
        re=p_y-y
        return np.dot(re.T,re)/2/len(y)
        #旧版代码：return (np.sum((p_y-y)**2)/2/len(y))
    
    #正规方程
    def norm_equa_(self,X,y,L2_n):
        '''
        return
        0: 求解的参数向量，narray(m,1)类型
        '''
        I=np.eye(len(X.columns))
        theta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)-L2_n*I),X.T),y)
        return theta
        
    #单次梯度下降
    def stoc_grad_desc_(self,X,y,theta,L2_n,learning_rate):
        '''
        return
        0: 依据学习率和梯度变化后的参数向量，narray(m,1)类型
        '''
        temp=theta
        #计算theta向量依据学习率和梯度变化后的值
        p_y=self.linear_(X,theta)
        temp=theta*(1-L2_n/len(y))-learning_rate*(np.dot(X.T,p_y-y))/len(y)
        return temp
        #旧版代码（非矩阵运算，性能相差6倍）
        #for i in range(len(theta)):
            #temp[i]=theta[i]-a*np.sum((p_y-y)*X.iloc[:,i])/len(y)
  
    #用正规方程进行拟合
    #注：直接求得最优解，在特征数少时建议采用该方法，特征数很多时该方法效率很差
    #L2_n不为0时应用L2正则化（权重衰减），避免过拟合
    def fit_by_ne_(self,X,y):
        '''
        return
        0: 求解的参数向量，Series类型
        '''
        theta=self.norm_equa_(X,y,self.L2_n)
        return pd.Series(theta)
    
    #用梯度下降法进行拟合
    #注：迭代优化逼近最优解，在特征数很多时也有较好的效率
    #梯度下降需要求代价函数的一阶连续导数，无法应用L1正则化，只能使用L2正则化
    def fit_by_sgd_(self,X,y):
        '''
        return
        0: 结束迭代后得到的参数向量，Series类型
        1: 历史参数向量，DataFrame类型
        2: 迭代中达到的最小代价，float类型
        3: 历史代价，Series类型
        '''
        n,m=len(y),len(X.columns)
        #校正mini-batch
        if self.mini_batch==0:
            if n<256:
                mini_batch=n
            elif 256*self.iter_max<n:
                mini_batch=int(n/self.iter_max)
            else:
                mini_batch=256
        else:
            if self.mini_batch>n:
                print('\nwarning: mini-batch is too big')
                mini_batch=n
            else:
                mini_batch=self.mini_batch
        #初始化变量
        theta=np.zeros(m)
        theta_h,cost_h=[],[]
        theta_h.append(tuple(theta))
        p_y=self.linear_(X,theta)
        cost_min=self.cost_(p_y,y)
        cost_h.append(cost_min)
        no_desc=0
        self.iter_num=0
        #迭代计算
        for i in range(self.iter_max):
            #进行一次梯度下降
            sp_X=X.sample(n=mini_batch)
            sp_y=y[sp_X.index]
            theta=self.stoc_grad_desc_(sp_X,sp_y,theta,self.L2_n,self.learning_rate)
            #记录本次结果
            self.iter_num+=1
            theta_h.append(tuple(theta))
            p_y=self.linear_(X,theta)
            cost=self.cost_(p_y,y)
            cost_h.append(cost)
            #如果超过10次迭代没有进一步降低cost_，提前结束迭代
            if cost<cost_min:
                cost_min=cost
                no_desc=0
            else:
                no_desc+=1
            if (no_desc>=10)&(self.early_stop==True):
                print('\nwarning: early stopping')
                break
            #cost值溢出时停止迭代
            if cost==float("inf"):
                print('\nwarning: cost value overflow')
                break
        #cost和theta的历史值
        theta=pd.Series(theta)
        theta_h=pd.DataFrame(theta_h)
        cost_h=pd.Series(cost_h)
        #异常提示：cost变化曲线严重发散
        if cost_h[len(cost_h)-1]/cost_h[0]>=1e+10:
            print('\nwarning: seriously divergence')
        #异常提示：cost变化曲线强烈振荡
        if cost_h[cost_h>cost_h[0]].count()>0:
            print('\nwarning: strong oscillation')
        #异常提示：cost变化曲线末梢不稳定且出现反弹回升
        if cost_h[cost_h.index>(self.iter_max/2)].mean()/cost_min>2:
            print('\nwarning: later costs were unstable and rebounded')
        return theta,theta_h,cost_min,cost_h
    
    #X输入校验
    def check_input_X_(self,X):
        '''
        return
        0: 补齐常数列的X，DataFrame类型
        '''
        if type(X)==type(pd.Series()):
            X=X.to_frame()
        check_type('X',type(X),type(pd.DataFrame()))
        type_list=[np.int64,np.float64]
        for i in range(len(X.columns)):
            check_type('column %d in X'%i,X.dtypes[i],type_list)
        return dp.fill_x0(X)
    
    #y/p_y输入校验
    def check_input_y_(self,y,name='y'):
        check_type(name,type(y),type(pd.Series()))
        type_list=[np.int64,np.float64]
        check_type(name,y.dtype,type_list)
    
    #theta输入校验
    def check_input_t_(self,theta):
        check_type('theta',type(theta),type(pd.Series()))
        type_list=[np.float64]
        check_type('theta',theta.dtype,type_list)

    #拟合
    def fit(self,X,y,output=False,show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合线性回归
        
        Note: 线性回归的输入数据必须全部是数值类型，其他类型自行预处理
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        y: 观测值向量,Series类型
        output: 是否返回求解的参数向量，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 返回求解的参数向量，Series类型
        -------
        '''
        start=time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('output',type(output),type(True))
            X=self.check_input_X_(X)
            self.check_input_y_(y)
            check_index_match(X,y,'X','y')
        #归一化校验
        range_=X.iloc[:,1:].max()-X.iloc[:,1:].min()
        if (range_.max()<1.1)&(range_.min()>0.9):
            if (self.learning_rate<0.1)&(self.fit_mode=='sgd'):
                print('\nit is recommended to change learning_rate over 0.1 for scaled X')
        else:
            print('\nit is recommended to scale X')
        #选择不同的拟合方式
        print('\nfitting ---')
        if self.fit_mode=='ne':
            theta=self.fit_by_ne_(X,y)
            self.theta=theta
            p_y=self.predict(X,check_input=False)
            a_result=self.assess(y,p_y,detailed=True,check_input=False)
            self.cost=a_result.loc['cost']
            self.cost_min=self.cost
            self.score=a_result.loc['r_sqr']
        elif self.fit_mode=='sgd':
            theta,theta_h,cost_min,cost_h=self.fit_by_sgd_(X,y)
            self.theta=theta
            self.theta_h=theta_h
            self.cost=cost_h.iloc[-1]
            self.cost_min=cost_min
            self.cost_h=cost_h
            try:
                p_y=self.predict(X,check_input=False)
                self.score=self.assess(y,p_y,check_input=False)
            except:
                print('\nwarning: fail to assess on train')
        time_cost=time.clock()-start
        if show_time==True:
            print('\ntime used for training: %f'%time_cost)
        #返回求得的参数
        if output==True:
            return theta
    
    #快速绘制cost,theta的变化曲线
    def plot_change_h(self):
        '''\n
        Function: 快速绘制cost,theta的变化曲线
        
        Note: 梯度下降拟合后可用，即fit_mode='sgd'
        
        Print
        -------
        0: 基于matplotlib绘制的关于cost/iter和theta/iter关系的曲线图
        -------
        '''
        try:
            self.cost_h.plot()
            plt.xlabel('iteration')
            plt.ylabel('cost')   
            plt.suptitle("[history of cost]")
            plt.show()
            self.theta_h.plot()
            plt.xlabel('iteration')
            plt.ylabel('theta')
            plt.suptitle("[history of theta]")
            plt.legend(loc='right')
            plt.show()
        except:
            print('\nfail to plot: this method can only use after fit by sgd')

    #预测    
    def predict(self,X,theta=None,show_time=False,check_input=True):
        '''\n
        Function: 对输入数据进行预测
        
        Note: theta参数不提供时直接使用内部存储
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        theta: 参数向量,Series类型
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测值向量，Series类型
        -------
        '''
        start=time.clock()
        #外部传入theta或使用内部缓存
        if type(theta)==type(None):
            theta=self.theta
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            X=self.check_input_X_(X)
            self.check_input_t_(theta)
            check_feats_match(X.columns,theta,'features in X','theta',mode='len')
        #预测
        p_y=pd.Series(self.linear_(X,theta),index=X.index)
        time_cost=time.clock()-start
        if show_time==True:
            print('\ntime used for predict: %f'%time_cost)
        return p_y

    #执行模型评估
    def assess(self,y,p_y,theta=None,detailed=False,check_input=True):
        '''\n
        Function: 执行模型评估
        
        Note: 拟合后关于训练集的r2和cost已保存在内部属性中，
              通过.score和.cost查看
        
        Parameters
        ----------
        y: 观测值向量,Series类型
        p_y: 预测值向量,Series类型
        detailed: 是否返回详细评估，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: r2或评估结果表，float类型或Series类型
        -------
        '''
        #外部传入theta或使用内部缓存
        if type(theta)==type(None):
            theta=self.theta
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('detailed',type(detailed),type(True))
            self.check_input_y_(y)
            self.check_input_y_(p_y,'p_y')
            check_index_match(y,p_y,'y','p_y')
        #r2计算
        r_sqr=stats.r_sqr(y,p_y)
        #是否进行详细评估
        if detailed==False:
            return r_sqr
        else:
            k,n=len(theta),len(y)
            cost=self.cost_(y,p_y)
            #计算调整r2和代价值
            adj_r_sqr=stats.adj_r_sqr(r_sqr,n,k)
            a_result=[]
            #f_value=self.f_test(p_y,y,len(x),len(theta))
            a_result.append(('r_sqr',r_sqr))
            a_result.append(('adj_r_sqr',adj_r_sqr))
            a_result.append(('cost',cost))
            a_result=pd.DataFrame(a_result,columns=['index','value'])
            a_result=a_result.set_index('index').iloc[:,0]
            return a_result
        

#逻辑回归    
class LogisticRegression:
    '''\n  
    Note: Note: 逻辑回归，只能用于分类（虽然名字叫回归），对于非线性问题的拟合需要进行特征映射
    
    Parameters
    ----------
    fit_mode: 拟合模式，str类型,目前只有sgd,
              'sgd'->随机梯度下降，迭代优化，没有限制但需要选择合适的a和iter_max
              默认值'sgd'
    multi_class: 多分类模式，str类型，默认'ovr'，(下面的n指类的数量)
                'ovr'-> one vs rest，一个分类作为正样本，其余分类作为负样本，
                        共训练n个分类器
                'ovo'-> one vs one，一个分类作为正样本，另一个分类作为负样本，
                        共训练n(n-1)/2个分类器，耗时更长，准确率更高
                除训练多个分类器，每个分类器一个参数向量的方式外，
                还可以直接训练单个分类器，该分类器拥有一个参数矩阵，
                但要求训练集大小一致，ovr容易实现该种方式
    learning_rate: 迭代优化学习率，float类型(>0.0)，默认值0.001
    iter_max: 迭代优化迭代次数上限，int类型(>0)，默认值1000
    mini_batch: 迭代优化每次使用的样本数，int类型(>0)，默认值256
    L2_n: L2正则化系数，float类型(>0.0)，默认值为0，
          该系数在代价中加入了关于模型复杂度的惩罚项，
          作用是防止过拟合，一般用于非线性模式
    early_stop: 是否允许超过一定次数迭代后代价没有进一步降低时提前结束训练，
                bool类型，默认值True
    ----------
    '''
    
    def __init__(self,fit_mode='sgd',multi_class='ovo',learning_rate=0.001,
                 iter_max=1000,mini_batch=0,L2_n=0.0,early_stop=True):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('fit_mode',type(fit_mode),type(''))
        check_type('multi_class',type(multi_class),type(''))
        check_type('learning_rate',type(learning_rate),type(0.0))
        check_type('iter_max',type(iter_max),type(0))
        check_type('mini_batch',type(mini_batch),type(0))
        check_type('L2_n',type(L2_n),type(0.0))
        check_type('early_stop',type(early_stop),type(True))
        fit_mode=fit_mode.lower()
        mode_list,mode_list2=['sgd'],['ovr','ovo']
        check_limit('fit_mode',fit_mode in mode_list,str(mode_list))
        check_limit('multi_class',multi_class in mode_list2,str(mode_list2))
        check_limit('learning_rate',learning_rate>0.0,'value>0.0')
        check_limit('iter_max',iter_max>0,'value>0')
        check_limit('mini_batch',mini_batch>=0,'value>=0')
        check_limit('L2_n',L2_n>=0.0,'value>=0.0')
        self.fit_mode=fit_mode
        self.learning_rate=learning_rate
        self.iter_max=iter_max
        self.mini_batch=mini_batch
        self.L2_n=L2_n
        self.early_stop=early_stop
        self.multi_class=multi_class
        
    #线性内核函数
    def linear_(self,X,theta):
        '''
        return
        0: 矩阵相乘结果，即预测值向量，narray(m,1)类型
        '''
        return np.dot(X,theta)
    
    #sigmoid函数
    def sigmoid_(self,array):
        '''
        return
        0: 映射后的概率值，narray(m,1)类型，范围0.0~1.0
        '''
        return 1.0/(1.0+np.e**(-1.0*array))
    
    #代价函数
    def cost_(self,p_y,y):
        '''
        return
        0: log误差，float类型
        '''
        return np.sum(y*np.log(p_y)+(1-y)*np.log(1-p_y))*(-1.0/len(y))
    
    #单次梯度下降
    #注：逻辑回归的梯度计算和线性回归是一样的
    def stoc_grad_desc_(self,X,y,theta,L2_n,learning_rate):
        '''
        return
        0: 依据学习率和梯度变化后的参数向量，narray(m,n)类型
        '''
        temp=theta
        p_y=self.sigmoid_(self.linear_(X,theta))
        temp=theta*(1-L2_n/len(y))-learning_rate*(np.dot(X.T,p_y-y))/len(y)
        return temp
    
    #使用梯度下降拟合
    #注：迭代优化逼近最优解，在特征数很多时也有较好的效率
    #梯度下降需要求损失函数的一阶连续导数，无法应用L1正则化，只能使用L2正则化
    def fit_by_sgd_(self,X,y):
        '''
        return
        0: 结束迭代后得到的参数向量，Series类型
        1: 历史参数向量，DataFrame类型
        2: 迭代中达到的最小代价，float类型
        3: 历史代价，Series类型
        '''
        #数据集大小，特征数量
        n,m=len(y),len(X.columns)
        #校正mini-batch
        if self.mini_batch==0:
            if n<256:
                mini_batch=n
            elif 256*self.iter_max<n:
                mini_batch=int(n/self.iter_max)
            else:
                mini_batch=256
        else:
            if self.mini_batch>n:
                print('\nwarning: mini-batch is too big')
                mini_batch=n
            else:
                mini_batch=self.mini_batch
        #初始化变量
        theta=np.zeros(m)
        theta_h=np.zeros((self.iter_max,m))
        cost_h=np.zeros(self.iter_max)
        p_y=self.sigmoid_(self.linear_(X,theta))
        cost_min=self.cost_(p_y,y)
        cost_h[0]=cost_min
        no_desc,self.iter_num=0,0
        #迭代计算
        for i in range(self.iter_max):
            #进行一次梯度下降
            sp_X=X.sample(n=mini_batch)
            sp_y=y[sp_X.index]
            theta=self.stoc_grad_desc_(sp_X,sp_y,theta,self.L2_n,self.learning_rate)
            #记录本次结果
            self.iter_num+=1
            theta_h[i]=theta
            p_y=self.sigmoid_(self.linear_(X,theta))
            cost=self.cost_(p_y,y)
            cost_h[i]=cost
            #如果超过10次迭代没有进一步降低cost，提前结束迭代
            if cost<cost_min:
                cost_min=cost
                no_desc=0
            else:
                no_desc+=1
            if (no_desc>=10)&(self.early_stop==True):
                print('\nwarning: early stopping')
                break
            #cost值溢出时停止迭代
            if cost==float("inf"):
                print('\nwarning: cost value overflow')
                break
        #cost和theta的历史值
        theta=pd.Series(theta)
        theta_h=pd.DataFrame(theta_h)
        cost_h=pd.Series(cost_h)
        #异常提示：cost变化曲线严重发散
        if cost_h[len(cost_h)-1]/cost_h[0]>=1e+10:
            print('\nwarning: seriously divergence')
        #异常提示：cost变化曲线强烈振荡
        if cost_h[cost_h>cost_h[0]].count()>0:
            print('\nwarning: strong oscillation')
        #异常提示：cost变化曲线后半部分不稳定且出现反弹回升
        if cost_h[cost_h.index>(self.iter_max/2)].mean()/cost_min>2:
            print('\nwarning: later costs were unstable and rebounded')
        return theta,theta_h,cost_min,cost_h
    
    #X输入校验
    def check_input_X_(self,X):
        '''
        return
        0: 补齐常数列的X，DataFrame类型
        '''
        if type(X)==type(pd.Series()):
            X=X.to_frame()
        check_type('X',type(X),type(pd.DataFrame()))
        type_list=[np.int64,np.float64]
        for i in range(len(X.columns)):
            check_type('column %d in X'%i,X.dtypes[i],type_list)
        return dp.fill_x0(X)
    
    #y/p_y输入校验
    def check_input_y_(self,y,name='y'):
        '''
        return
        0: 转换为str的y，Series类型
        '''
        check_type(name,type(y),type(pd.Series()))
        return y.astype('str')
    
    #theta输入校验
    def check_input_t_(self,theta):
        if type(theta)==type(pd.Series()):
            theta=theta.to_frame()
        check_type('theta',type(theta),type(pd.DataFrame()))
        type_list=[np.float64]
        for i in range(len(theta.columns)):
            check_type('theta',theta.dtypes[i],type_list)
    
    #拟合
    def fit(self,X,y,output=False,show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合逻辑回归
        
        Note: 逻辑回归的特征输入为连续型数值，分类输出为离散标签
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        y: 观测值向量,Series类型
        output: 是否返回求解的参数向量，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 返回求解的参数向量，Series类型
        -------
        '''
        start=time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('output',type(output),type(True))
            X=self.check_input_X_(X)
            y=self.check_input_y_(y)
            check_index_match(X,y,'X','y')
        #判断类别数量
        values=y.sort_values().drop_duplicates().tolist()
        features_n,classes_n=len(X.columns),len(values)
        if classes_n<=1:
            raise ValueError('classes_n in y should >=2')
        if classes_n>=0.5*len(y):
            print('\nwarning: too many classes in y')
        self.classes=values
        #归一化校验
        range_=X.iloc[:,1:].max()-X.iloc[:,1:].min()
        if (range_.max()<1.1)&(range_.min()>0.9):
            if self.learning_rate<0.1:
                print('\nit is recommended to change learning_rate over 0.1 for scaled X')
        else:
            print('\nit is recommended to scale X')
        #将单列的多类别分类值转换为多列的01类别判断，索引(记录，类)->属于该类
        Y=dp.dummy_var(y)
        theta_h,cost_h=[],[]
        #多分类模式ovr
        if self.multi_class=='ovr':
            theta=np.zeros((features_n,classes_n))
            cost_min=np.zeros(classes_n)
            cost=np.zeros(classes_n)
            for i in range(classes_n):
                print('\nfitting classifier %d ---'%i)
                theta_,theta_h_,cost_min_,cost_h_=self.fit_by_sgd_(X,Y.iloc[:,i])
                theta[:,i],cost_min[i],cost[i]=theta_,cost_min_,cost_h_.iloc[-1]
                theta_h.append(theta_h_)
                cost_h.append(cost_h_)
            self.classes_paired=None
        #多分类模式ovo
        elif self.multi_class=='ovo':
            #正负样本选取矩阵,索引(组合，类)->取用
            class_p,class_n=dp.combine_enum_paired(list(range(classes_n)))
            #应用正负样本选取矩阵后的分类情况，索引(记录，组合)->分类判断
            #1->正样本分类，0->负样本分类，0.5->无法判别
            Y_=(np.dot(Y,class_p.T)-np.dot(Y,class_n.T)+1.0)/2.0
            Y_=pd.DataFrame(Y_,index=Y.index)
            combines_n=len(Y_.columns)
            theta=np.zeros((features_n,combines_n))
            cost_min=np.zeros(combines_n)
            cost=np.zeros(combines_n)
            for i in range(combines_n):
                print('\nfitting classifier %d ---'%i)
                theta_,theta_h_,cost_min_,cost_h_=self.fit_by_sgd_(X,Y_.iloc[:,i])
                theta[:,i],cost_min[i],cost[i]=theta_,cost_min_,cost_h_.iloc[-1]
                theta_h.append(theta_h_)
                cost_h.append(cost_h_)
            self.classes_paired=class_p-class_n
        theta=pd.DataFrame(theta)
        cost_min=pd.Series(cost_min)
        cost=pd.Series(cost)
        self.theta=theta
        self.theta_h=theta_h
        self.cost_min=cost_min
        self.cost_h=cost_h
        self.cost=cost
        p_y=self.predict(X,check_input=False)
        self.score=self.assess(y,p_y,check_input=False)
        time_cost=time.clock()-start
        if show_time==True:
            print('\ntime used for training: %f'%time_cost)
        #返回求得的参数
        if output==True:
            return theta
    
    #所有分类器的预测结果
    def predict_(self,X,theta,classes_paired=None,return_proba=False):
        '''
        return
        0: 预测矩阵，narray(m,n)类型
        '''
        #在各个分类器判定为正分类的概率，索引（记录，分类器）->正分类概率
        p_y=self.sigmoid_(self.linear_(X,theta))
        #进一步转化为各个分类的概率,索引(记录，类)
        if type(classes_paired)!=type(None):
            class_p=classes_paired.copy()
            class_p[class_p<0]=0
            class_n=-classes_paired.copy()
            class_n[class_n<0]=0
            p_y=np.dot(p_y,class_p)+np.dot(1-p_y,class_n) 
        #归一化
        p_y=(p_y.T/p_y.sum(axis=1)).T
        #转化为离散值
        if return_proba==False:
            p_y_max=p_y.max(axis=1)
            max_idx=(p_y.T==p_y_max).T.astype('int')
            classes_idx=np.array(range(p_y.shape[1]))
            p_y_=np.dot(max_idx,classes_idx).astype('int')
            return p_y_
        else:
            return p_y        
    
    #预测
    def predict(self,X,theta=None,classes=None,classes_paired=None,
                return_proba=False,show_time=False,check_input=True):
        '''\n
        Function: 对输入数据进行预测
        
        Note: theta,classes,classes_paired参数不提供时直接使用内部存储
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        theta: 参数向量,Series类型
        classes: 类标签，list(str)类型
        classes_paired: 正负样本类选取，narray(m,n)类型
        return_proba: 是否返回分类概率，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测值向量，Series类型
        -------
        '''
        start=time.clock()
        #外部传入参数或使用内部缓存
        if type(theta)==type(None):
            theta=self.theta
        if type(classes)==type(None):
            classes=self.classes
        if type(classes_paired)==type(None):
            classes_paired=self.classes_paired
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            X=self.check_input_X_(X)
            self.check_input_t_(theta)
            check_feats_match(X.columns,theta,'features in X','theta',mode='len')
        #预测
        p_y=self.predict_(X,theta,classes_paired,return_proba)
        if return_proba==False:
            p_y=pd.Series(p_y,name='classify',index=X.index)
            for i in range(len(classes)):
                p_y[p_y==i]=classes[i]
            time_cost=time.clock()-start
            if show_time==True:
                print('\ntime used for predict: %f'%time_cost)
            return p_y
        else:
            time_cost=time.clock()-start
            if show_time==True:
                print('\ntime used for predict: %f'%time_cost)
            return pd.DataFrame(p_y,columns=classes,index=X.index)
    
    #评估
    def assess(self,y,p_y,return_dist=False,check_input=True):
        '''\n
        Function: 执行模型评估
        
        Note: 拟合后关于训练集的accuracy和cost已保存在内部属性中，
              通过.score和.cost查看
        
        Parameters
        ----------
        y: 观测值向量,Series类型
        p_y: 预测值向量,Series类型
        return_dist: 是否返回预测分布，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 准确率，float类型
        1: 预测分布，DataFrame类型
        -------
        '''
        #输入校验
        if check_input==True:
            y=self.check_input_y_(y)
            p_y=self.check_input_y_(p_y,'p_y')
            check_index_match(y,p_y,'y','p_y')
        #返回准确率和预测分布
        return stats.accuracy(y,p_y,return_dist,self.classes)