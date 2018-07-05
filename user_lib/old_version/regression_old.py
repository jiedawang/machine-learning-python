# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from user_lib.data_prep import DataPreprocessing

class LinearRegression:
    
    #线性回归模型：theta0*1+theta1*x1+theta2*x2+...
    #theta:参数向量,Series类型,theta[i]对应thetai
    #x:特征矩阵,DataFrame类型,column[i]对应xi，row对应每一条数据记录
    def model(self,x,theta):
        #返回矩阵相乘结果
        return np.dot(x,theta)
    
    #代价函数,用于衡量拟合的偏差程度       
    #fx:模型预测值（向量）
    #y:真实观测值（向量）
    def cost(self,fx,y):
        re=fx-y
        return np.dot(re.T,re)/2/len(y)
        #旧版代码：return (np.sum((fx-y)**2)/2/len(y))
    
    #拟合方法1： 正规方程求解 
    #注：直接求得最优解，在特征数少时建议采用该方法，特征数很多时该方法效率很差
    #L2_n不为0时应用L2正则化（权重衰减），避免过拟合
    def normal_equation(self,x,y):
        I=np.eye(len(x.columns))
        theta=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)-self.L2_n*I),x.T),y)
        return theta
        
    #拟合方法2： 梯度下降法，计算下一组theta
    #注：用于逐步逼近最优解，在特征数很多时也有较好的效率
    #根据每次给到的数据集大小可对应批量梯度下降BGD和随机梯度下降SGD
    #注：梯度下降需要求损失函数的一阶连续导数，无法应用L1正则化，只能使用L2正则化
    #(此处代码仅作对照，实际没有使用)
    def gradient_descent(self,theta,x,y):
        temp=theta
        #计算theta向量按步长变化后的值
        fx=self.model(x,theta)
        temp=theta*(1-self.L2_n/len(y))-self.gd_a*(np.dot(x.T,fx-y))/len(y)
        #旧版代码（非矩阵运算，效率相差6倍）
        #for i in range(len(theta)):
            #temp[i]=theta[i]-step_len*np.sum((fx-y)*x.iloc[:,i])/len(y)
        return temp
  
    #直接输入数据进行拟合： 正规方程法
    #x,y:源数据,L2_n：L2正则化强度
    def fit_by_ne(self,x_,y,L2_n=0):
        #补全源数据
        x=DataPreprocessing.fill_x0(x_)
        self.L2_n=L2_n
        #求解
        self.theta=pd.Series(self.normal_equation(x,y))
        fx=self.model(x,self.theta)
        self.score,self.a_result=self.assessment(fx,y,len(self.theta))
        return self.theta
    
    #直接输入数据进行拟合： 梯度下降法
    #x,y:源数据
    #init_dir:梯度下降设置参数（类型不同，参数不同）
    #iter_max:迭代次数上限
    #gd_type:梯度下降类型，默认随机梯度下降
    #L2_n:L2正则化强度
    #early_stop:是否允许在cost没有继续下降的趋势时提前结束迭代
    #feedback：反馈信息
    #sample_n：每次迭代选取的mini-batch大小
    def fit_by_gd(self,x_,y,init_dir={'a':0.1},iter_max=200,gd_type='SGD',
                  L2_n=0,early_stop=True,sample_n=256):
        #校正mini-batch
        if sample_n>len(x_):
            sample_n=len(x_)
        #补全源数据
        x=DataPreprocessing.fill_x0(x_)
        #初始化变量
        theta=pd.Series(np.zeros(len(x.columns))).astype('float')
        theta_h,cost_h=[],[]
        theta_h.append(tuple(theta))
        fx=self.model(x,theta)
        cost_min=self.cost(fx,y)
        cost_h.append(cost_min)
        no_desc=0
        self.iter_num=0
        #迭代计算
        gd_mng=GradDesc(init_dir,gd_type,self.model,L2_n)
        for i in range(iter_max):
            #进行一次梯度下降
            sp_x=x.sample(n=sample_n)
            sp_y=y[sp_x.index]
            theta=gd_mng.exec_one_iter(theta,sp_x,sp_y)
            #记录本次结果
            self.iter_num+=1
            theta_h.append(tuple(theta))
            fx=self.model(x,theta)
            cost=self.cost(fx,y)
            cost_h.append(cost)
            #如果超过10次迭代没有进一步降低cost，提前结束迭代
            if cost<cost_min:
                cost_min=cost
                no_desc=0
            else:
                no_desc+=1
            if no_desc>=10 & early_stop==True:
                print('early stopping')
            #cost值溢出时停止迭代
            if cost==float("inf"):
                print('cost value overflow')
                break
        #保存结果
        self.theta=theta
        self.theta_h=pd.DataFrame(theta_h)
        self.cost_h=pd.Series(cost_h)
        self.cost_min=self.cost_h[self.cost_h==cost_min]
        #cost严重发散时不进行评估
        if cost_h[len(cost_h)-1]/cost_h[0]<=1e+10:
            self.score,self.a_result=self.assessment(fx,y,len(self.theta))
        #异常提示
        if self.cost_h[self.cost_h>self.cost_h[0]].count()>0:
            print('Strong oscillation')
        if self.cost_h[self.cost_h.index>(iter_max/2)].mean()/cost_min.values[0]>2:
            print('Later costs were unstable and rebounded')
        return theta
    
    #快速绘制cost,theta的变化曲线
    #注：仅梯度下降法求解后可用
    def plot_change_h(self):
        self.cost_h.plot()
        plt.xlabel('iteration')
        plt.ylabel('cost')    
        plt.show()
        self.theta_h.plot()
        plt.xlabel('iteration')
        plt.ylabel('theta')
        plt.show()

    #执行模型评估
    #k:theta参数个数
    #return 首要评分，全部指标
    def assessment(self,fx,y,k):
        a_result=[]
        cost=self.cost(fx,y)
        r_sqr=Statistics.get_r_sqr(fx,y)
        adj_r_sqr=Statistics.get_adj_r_sqr(r_sqr,len(fx),k)
        #f_value=self.f_test(fx,y,len(x),len(theta))
        a_result.append(('r_sqr',r_sqr))
        a_result.append(('adj_r_sqr',adj_r_sqr))
        a_result.append(('cost',cost))
        a_result=pd.DataFrame(a_result,columns=['index','value'])
        a_result.set_index('index',inplace=True)
        return r_sqr,a_result
    
    #预测测试 
    #return 预测值，首要评分，全部指标
    def predict_test(self,x_,y):
        x=DataPreprocessing.fill_x0(x_)
        fx=pd.Series(self.model(x,self.theta))
        fx.index=x.index
        score,a_result=self.assessment(fx,y,len(self.theta))
        return fx,score,a_result
    
    #预测    
    def predict(self,x_):
        x=DataPreprocessing.fill_x0(x_)
        fx=pd.Series(self.model(x,self.theta))
        fx.index=x.index
        return fx
    
class LogisticRegression:
    
    #模型函数
    def model(self,x,theta):
        return self.sigmoid(np.dot(x,theta))
    
    #将目标函数结果值映射为概率值，范围0~1
    def sigmoid(self,fx):
        return 1.0/(1.0+np.e**(-1.0*fx))
    
    #代价函数
    def cost(self,hx,y):
        return np.sum(y*np.log(hx)+(1-y)*np.log(1-hx))*(-1.0/len(y))
    
    #拟合一个二分类的分类器
    def fit_binary(self,x,y,init_dir,iter_max,gd_type,
                   L2_n,early_stop,sample_n):
        print('fitting...')
        #初始化变量
        theta=pd.Series(np.zeros(len(x.columns))).astype('float')
        theta_h,cost_h=[],[]
        no_desc=0
        iter_num=0
        hx=self.model(x,theta)
        cost_min=self.cost(hx,y) 
        cost_h.append(cost_min)
        theta_h.append(tuple(theta))
        #初始化并执行梯度下降
        gd_mng=GradDesc(init_dir,gd_type,self.model,L2_n)
        for i in range(iter_max):
            #选取mini-batch
            sp_x=x.sample(n=sample_n)
            sp_y=y[sp_x.index]
            #单次迭代
            theta=gd_mng.exec_one_iter(theta,sp_x,sp_y)
            #记录本次执行结果
            iter_num+=1
            theta_h.append(tuple(theta))
            hx=self.model(x,theta)
            cost=self.cost(hx,y)
            cost_h.append(cost)
            #超过10次没有进一步降低cost，提前结束迭代
            if cost<cost_min:
                cost_min=cost
                no_desc=0
            else:
                no_desc+=1
            if no_desc>=10 & early_stop==True:
                print('Early stopping')
                break
            #cost过大溢出
            if cost==float("inf"):
                print('Cost value overflow')
                break
        #转换结果
        cost_h=pd.Series(cost_h)
        theta_h=pd.DataFrame(theta_h)
        cost_min=cost_h[cost_h==cost_min]
        #异常提示
        if cost_h[cost_h>cost_h[0]].count()>0:
            print('Strong oscillation')
        if cost_h[cost_h.index>(iter_max/2)].mean()/cost_min.values[0]>2:
            print('Later costs were unstable and rebounded')
        return theta,theta_h,cost_min,cost_h
    
    #直接输入数据拟合
    def fit(self,x_,y_,init_dir={'a':1},iter_max=200,gd_type='SGD',
            L2_n=0,early_stop=True,sample_n=256):
        #校正mini-batch大小
        if sample_n>len(x_):
            sample_n=len(x_)
        #补全x
        x=DataPreprocessing.fill_x0(x_)
        #判断是二分类还是多分类
        values=y_.drop_duplicates().sort_values().tolist()
        if len(values)<=1:
            raise NameError('class num in y should >=2')
        elif len(values)==2:
            y=y_
            theta,theta_h,cost_min,cost_h=\
                self.fit_binary(x,y,init_dir,iter_max,gd_type,
                    L2_n,early_stop,sample_n)
        else:
            #按one vs rest规则拟合多个分类器
            y=DataPreprocessing.dummy_var(y_)
            theta,theta_h,cost_min,cost_h=[],[],[],[]
            for i in range(len(y.columns)):
                theta_,theta_h_,cost_min_,cost_h_=\
                    self.fit_binary(x,y.iloc[:,i],init_dir,iter_max,gd_type,
                             L2_n,early_stop,sample_n)
                theta.append(theta_)
                theta_h.append(theta_h_)
                cost_min.append(cost_min_)
                cost_h.append(cost_h_)
        #保存结果
        self.theta=theta
        self.theta_h=theta_h
        self.cost_min=cost_min
        self.cost_h=cost_h
        self.classificaion=values
        #评估模型
        p=self.predict(x)
        self.score,self.pred_dist=self.assessment(p,y_) 
        return theta
    
    #二分类预测
    def predict_(self,x_,theta,discrete_p=True):
        x=DataPreprocessing.fill_x0(x_)
        hx=self.model(x,theta)
        p=pd.Series(hx)
        p.index=x.index
        #转化为离散值
        if discrete_p==True:
            p[p>0.5]=1
            p[p<=0.5]=0 
        return p        
    
    #完整预测
    def predict(self,x_):
        values=self.classificaion
        if len(values)==2:
            return self.predict_(x_,self.theta)
        else:          
            p_=pd.DataFrame()
            for i in range(len(values)):
                theta=self.theta[i]
                p_[i]=self.predict_(x_,theta,discrete_p=False)
            p_max=p_.max(axis=1)
            p=pd.Series(np.zeros(len(p_)))
            p.index=p_.index
            if type(values[0])=='str':
                p=p.astype('str')
            for j in range(len(values)):
                p[p_.iloc[:,j]==p_max]=values[j]
        return p
    
    #评估
    def assessment(self,p,y):
        cp=pd.DataFrame()
        cp['y'],cp['p']=y,p
        #整体准确率
        accuracy=len(cp[cp['y']==cp['p']])*1.0/len(y)
        #计算观测值/预测值矩阵
        values=self.classificaion
        pred_dist=np.zeros((len(values),len(values)))
        for i in range(len(values)):
            for j in range(len(values)):
                bool_index=(cp['y']==values[j])&(cp['p']==values[i])
                pred_dist[i][j]=len(cp[bool_index])*1.0/len(y)
        pred_dist=pd.DataFrame(pred_dist,
                               columns='y_'+pd.Series(values).astype('str'),
                               index='p_'+pd.Series(values).astype('str'))
        return accuracy,pred_dist
    
    #预测测试 
    def predict_test(self,x,y):
        p=self.predict(x)
        score,pred_dist=self.assessment(p,y)
        return p,score,pred_dist
  
#各类梯度下降算法
class GradDesc:
    
    #此处model为外部传入的模型函数，参数只能是x和theta
    def __init__(self,init_dir,gd_type,model,L2_n):
        self.type_list=('SGD','Momentum','Nesterov',
                   'Adagrad','RMSProp','Adadelta',
                   'Adam','Adamax','Nadam')
        if gd_type not in self.type_list:
            print('gd_type should in:')
            print(self.type_list)
            raise TypeError('Unknown type')
        else:
            self.gd_type=gd_type
            self.model=model
            self.gd_init(init_dir,gd_type)
            self.L2_n=L2_n
        
    def exec_one_iter(self,theta,sp_x,sp_y):
        if self.gd_type=='SGD':
            theta=self.gd_sgd(theta,sp_x,sp_y) 
        elif self.gd_type=='Momentum':
            theta=self.gd_momentum(theta,sp_x,sp_y)  
        elif self.gd_type=='Nesterov':
            theta=self.gd_nesterov(theta,sp_x,sp_y)  
        elif self.gd_type=='Adagrad':
            theta=self.gd_adagrad(theta,sp_x,sp_y)  
        elif self.gd_type=='RMSProp':
            theta=self.gd_rmsprop(theta,sp_x,sp_y)  
        elif self.gd_type=='Adadelta':
            theta=self.gd_adadelta(theta,sp_x,sp_y)  
        elif self.gd_type=='Adam':
            theta=self.gd_adam(theta,sp_x,sp_y)  
        elif self.gd_type=='Adamax':
            theta=self.gd_adamax(theta,sp_x,sp_y)  
        elif self.gd_type=='Nadam':
            theta=self.gd_nadam(theta,sp_x,sp_y)
        return theta
    
    #随机梯度下降（基础）
    def gd_sgd(self,theta,x,y):
        temp=theta
        fx=self.model(x,theta)
        temp=theta*(1-self.L2_n/len(y))-self.gd_a*(np.dot(x.T,fx-y))/len(y)
        return temp
    
    #动量加速梯度下降
    #注：p是动量因子，一般<=0.9
    def gd_momentum(self,theta,x,y):
        temp=theta
        #theta更新量=上一次的更新量last_m乘上动量因子p，再加上当前梯度g
        fx=self.model(x,theta)
        g=(np.dot(x.T,fx-y))/len(y)
        self.gd_m=self.gd_p*self.gd_m+g
        temp=theta*(1-self.L2_n/len(y))-self.gd_a*self.gd_m
        return temp
        
    #nesterov在momentum的基础上引入动量项修正当前梯度的计算
    def gd_nesterov(self,theta,x,y):
        temp=theta
        #theta更新量=上一次的更新量last_m乘上动量因子p，再加上经过动量项修正的当前梯度g
        fx=self.model(x,theta-self.gd_a*self.gd_p*self.gd_m)
        g=(np.dot(x.T,fx-y))/len(y)
        self.gd_m=self.gd_p*self.gd_m+g
        temp=theta*(1-self.L2_n/len(y))-self.gd_a*self.gd_m
        return temp
        
    #adagrad根据历史梯度的变化来修正学习率
    def gd_adagrad(self,theta,x,y):
        temp=theta
        #约束项regularizer:
        #分母=(历史梯度平方和n+防止除零情况的平滑项e)的平方根
        fx=self.model(x,theta)
        g=(np.dot(x.T,fx-y))/len(y)
        self.gd_n=self.gd_n+g**2
        delta=-(self.gd_a/np.sqrt(self.gd_n+self.gd_e))*g
        temp=theta*(1-self.L2_n/len(y))+delta
        return temp
    
    #rmsprop是对adagrad的改进
    #同样是根据历史梯度变化来修正学习率，但是历史梯度累计限制到固定范围，
    #且不存储历史项，而是通过近似法求动态平均值
    def gd_rmsprop(self,theta,x,y):
        temp=theta
        #约束项regularizer:
        #分母=(历史梯度g**2的动态平均值E+防止除零情况的平滑项e)的平方根
        #历史梯度g**2的动态平均值E又可称为历史梯度的均方根误差RMS
        fx=self.model(x,theta)
        g=(np.dot(x.T,fx-y))/len(y)
        self.gd_E=self.gd_p*self.gd_E+(1-self.gd_p)*(g**2)
        delta=-(self.gd_a/np.sqrt(self.gd_E+self.gd_e))*g
        temp=theta*(1-self.L2_n/len(y))+delta
        return temp
    
    #adadelta在rmsprop的基础上替换掉了全局学习率,以历史变化率的RMS替代
    def gd_adadelta(self,theta,x,y):
        temp=theta
        #约束项regularizer:
        #分母=(历史梯度g**2的动态平均值E+防止除零情况的平滑项e)的平方根
        #分子=(历史变化量delta**2的动态平均值E)的平方根
        fx=self.model(x,theta)
        g=(np.dot(x.T,fx-y))/len(y)
        self.gd_Eg2=self.gd_p*self.gd_Eg2+(1-self.gd_p)*(g**2)
        delta=-(np.sqrt(self.gd_Ed2+self.gd_e)/np.sqrt(self.gd_Eg2+self.gd_e))*g
        temp=theta*(1-self.L2_n/len(y))+delta
        self.gd_Ed2=self.gd_p*self.gd_Ed2+(1-self.gd_p)*(delta**2)
        return temp
    
    #Adam本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率
    def gd_adam(self,theta,x,y):
        temp=theta
        fx=self.model(x,theta)
        g=np.dot(x.T,fx-y)/len(y)
        self.gd_t=self.gd_t+1
        self.gd_m=self.gd_u*self.gd_m+(1-self.gd_u)*g
        self.gd_n=self.gd_v*self.gd_n+(1-self.gd_v)*(g**2)
        m_h=self.gd_m/(1-self.gd_u**self.gd_t)
        n_h=self.gd_n/(1-self.gd_v**self.gd_t)
        delta=-self.gd_a*m_h/(np.sqrt(n_h)+self.gd_e)
        temp=theta*(1-self.L2_n/len(y))+delta
        return temp
    
    #Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围
    def gd_adamax(self,theta,x,y):
        temp=theta
        fx=self.model(x,theta)
        g=np.dot(x.T,fx-y)/len(y)
        self.gd_t=self.gd_t+1
        self.gd_m=self.gd_u*self.gd_m+(1-self.gd_u)*g
        self.gd_n=pd.DataFrame([self.gd_v*self.gd_n,np.abs(g)]).max()
        m_h=self.gd_m/(1-self.gd_u**self.gd_t)
        delta=-self.gd_a*m_h/(self.gd_n+self.gd_e)
        temp=theta*(1-self.L2_n/len(y))+delta
        return temp
    
    #Nadam类似于带有Nesterov动量项的Adam
    def gd_nadam(self,theta,x,y):
        temp=theta
        fx=self.model(x,theta)
        g=np.dot(x.T,fx-y)/len(y)
        self.gd_t=self.gd_t+1
        g_h=g/(1-self.gd_u**((1+self.gd_t)*self.gd_t/2))
        self.gd_m=(self.gd_u**self.gd_t)*self.gd_m+(1-self.gd_u**self.gd_t)*g
        self.gd_n=self.gd_v*self.gd_n+(1-self.gd_v)*(g**2)
        m_h=self.gd_m/(1-self.gd_u**((2+self.gd_t)*(1+self.gd_t)/2))
        n_h=self.gd_n/(1-self.gd_v**self.gd_t)
        m_m=(1-self.gd_u**self.gd_t)*g_h+(self.gd_u**(self.gd_t+1))*m_h
        delta=-self.gd_a*m_m/(np.sqrt(n_h)+self.gd_e)
        temp=theta*(1-self.L2_n/len(y))+delta
        return temp
        
    ##参数初始化,以字典类型传入参数
    ## SGD : a
    ## Momentum,Nesterov : a,p
    ## Adagrad : a,e,k
    ## RMSProp : a,p,e,k
    ## Adadelta : p,e,k
    ## Adam,Adamax,Nadam : a,u,v,e,k
    ##(a为学习率，e为防止除零的平滑项，k为theta参数个数)
    def gd_init(self,init_dir,gd_type):
        try:
            if gd_type in ('SGD'):
                self.gd_a=init_dir['a']
            elif gd_type in ('Momentum','Nesterov'):
                self.gd_a=init_dir['a']
                self.gd_p=init_dir['p']
                self.gd_m=np.zeros(init_dir['k'])
            elif gd_type in ('Adagrad'):   
                self.gd_a=init_dir['a']
                self.gd_e=init_dir['e']
                self.gd_n=np.zeros(init_dir['k'])
            elif gd_type in ('RMSProp'):   
                self.gd_a=init_dir['a']
                self.gd_p=init_dir['p']
                self.gd_e=init_dir['e']
                self.gd_E=np.zeros(init_dir['k'])
            elif gd_type in ('Adadelta'): 
                self.gd_p=init_dir['p']
                self.gd_e=init_dir['e']
                self.gd_Eg2=np.zeros(init_dir['k'])
                self.gd_Ed2=np.zeros(init_dir['k'])
            elif gd_type in ('Adam','Adamax','Nadam'): 
                #参考值:a=0.001,u=0.9,v=0.999,e=1e-8
                self.gd_a=init_dir['a']
                self.gd_u=init_dir['u']
                self.gd_v=init_dir['v']
                self.gd_e=init_dir['e']
                self.gd_m=np.zeros(init_dir['k'])
                self.gd_n=np.zeros(init_dir['k'])
                self.gd_t=0
        except:
            print('gd init error,check parameters')
        
class Statistics:
    
    #源数据评估： 相关系数 (目前只实现了pearson一种)
    #correlation_coefficient
    
    #计算x与y的相关系数向量
    def corr_xy(x,y):
        k = len(x.columns)
        correl=np.empty(k, dtype=float)
        for ia,ca in enumerate(x.columns):
            correl[ia]=Statistics.corrf(x[ca],y)
        return pd.Series(correl,index=x.columns,name='y')
    
    #计算x各特征之间的相关系数矩阵
    def corr_xx(x):
        k=len(x.columns)
        correl=np.empty((k,k),dtype=float)
        for ia,ca in enumerate(x.columns):
            for ib,cb in enumerate(x.columns):
                correl[ia][ib]=Statistics.corrf(x[ca],x[cb])
        return pd.DataFrame(correl,index=x.columns,columns=x.columns)
 
    #计算两个向量的相关系数
    def corrf(a,b):
        if len(a)!=len(b):
            raise TypeError('a,b should have the same length')
        da=a-a.mean()
        db=b-b.mean()
        n=len(a)
        Da=np.dot(da.T,da)/n
        Db=np.dot(db.T,db)/n
        Covab=np.dot(da.T,db)/n
        return Covab/(np.sqrt(Da)*np.sqrt(Db))
    
    #模型评估-模型解释力：R方(判定系数)
    #范围0~1，越大拟合结果越好
    def get_r_sqr(fx,y):
        #总平方和
        buf1=y-y.mean()
        SST=np.dot(buf1.T,buf1)
        #SST=np.sum((y-y.mean())**2)
        #残差平方和
        buf2=fx-y
        SSE=np.dot(buf2.T,buf2)
        #SSE=np.sum((fx-y)**2)
        #回归平方和=总平方和-残差平方和
        #R方=回归平方和/总平方和
        return (SST-SSE)/SST
    
    #模型评估-模型解释力：调整R方
    #（消除样本数和参数个数带来的影响）
    #r2:R方
    #n:样本容量
    #k:theta参数个数
    def get_adj_r_sqr(r2,n,k):
        if n-k==0:
            print('adj_r_sqr error:n=k')
            return 0
        else:
            return 1-(1-r2)*(n-1)/(n-k)
              
    '''
    #（这个没搞明白什么意思）
    #模型评估-回归显著性：F检验
    #(用于判断回归模型是否能真实反映数据之间的关系)
    #n:样本容量
    #k:theta参数个数
    def f_test(self,fx,y,n,k):
        #回归平方和
        #（回归平方和+残差平方和=总平方和）
        SSR=np.sum((fx-y.mean())**2)
        #残差平方和
        SSE=np.sum((fx-y)**2)
        #F统计量
        return (SSR/(k-1))/(SSE/(n-k))
    ''' 
        
