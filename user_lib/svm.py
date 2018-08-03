# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.optimize as opt
import random
import user_lib.statistics as stats
import user_lib.data_prep as dp
from user_lib.check import check_type,check_limit,check_index_match,check_feats_match
import time

#支持向量机
class SupportVectorMachine:
    '''\n  
    Note: 支持向量机，支持分类和回归
    
    Parameters
    ----------
    mode: 模式，分类->'c'，回归->'r'，默认'c'
    multi_class : 多分类模式，仅分类模式有效，str类型，默认'ovr'，(下面的k指类的数量)
                'ovr'-> one vs rest，一个分类作为正样本，其余分类作为负样本，
                        共训练k个分类器
                'tree'-> 二叉树，每个节点均分类别集合，共训练k-1个分类器
    iter_max: 迭代优化迭代次数上限，int类型(>0)，默认值10
    C: 惩罚系数，float类型(C>0.0)，默认10.0，值越高，模型对误差的容忍度越低
    k_type: 核函数类型，str类型，默认'lin'，
            'lin'->线性核，<x1,x2>
            'pol'->多项式核，(<x1,x2>+R)^d
            'rbf'->高斯核(也叫径向基核)，exp(-||x1-x2||^2/(2*sigma^2))
            核函数用于计算高维映射后的内积，使模型能够解决非线性问题
    k_args: 核函数参数，dict类型，
            k_type='lin' -> None
            k_type='pol' -> R,d
            k_type='rbf' -> sigma
    relax: 松弛变量，float类型(>0.0)，默认0.001，
           用于处理离群点，放宽了kkt条件的判定
    eps: 回归间隔带宽度，仅回归模式有效，float类型(>0.0)，默认1.0，
         该值越大，对回归误差的容忍度越高
    ----------
    
    Attributes
    ----------
    w: 超平面法线向量，narray(m,)类型
    b: 偏置量，float类型
    a: 拉格朗日乘子，narray(m,)类型
    sv_X: 支持向量特征列，narray(m,n)类型
    sv_y: 支持向量目标列，narray(m,)类型
    ----------
    '''
    
    def __init__(self,mode='c',multi_class='ovr',iter_max=10,C=10.0,
                 k_type='lin',k_args=None,relax=0.001,eps=1.0):
        check_type('mode',type(mode),type(''))
        check_type('multi_class',type(multi_class),type(''))
        check_type('iter_max',type(iter_max),type(0))
        check_type('C',type(C),type(0.0))
        check_type('k_type',type(k_type),type(''))
        check_type('relax',type(relax),type(0.0))
        check_type('eps',type(eps),type(0.0))
        mode_list,mode_list2=['c','r'],['ovr','tree']
        check_limit('mode',mode in mode_list,str(mode_list))
        check_limit('multi_class',multi_class in mode_list2,str(mode_list2))
        check_limit('iter_max',iter_max>0,'value>0')
        check_limit('C',C>0.0,'value>0.0')
        check_limit('relax',relax>0.0,'value>0.0')
        check_limit('eps',eps>0.0,'value>0.0')
        type_list=['lin','pol','rbf']
        check_limit('k_type',k_type in type_list,str(type_list))
        if k_type!='lin':
            check_type('k_args',type(k_args),type({}))
        if k_type=='pol':
            if ('R' in k_args.keys())&('d' in k_args.keys()):
                check_type('k_args:R',type(k_args['R']),type(0.0))
                check_type('k_args:d',type(k_args['d']),type(0))
                check_limit('k_args:d',k_args['d']>0,'value>0')
            else:
                raise ValueError('k_args should provide R and d for k_type: pol')
        if k_type=='rbf':
            if 'sigma' in k_args.keys():
                check_type('k_args:sigma',type(k_args['sigma']),type(0.0))
                check_limit('k_args:sigma',k_args['sigma']>0.0,'value>0.0')
            else:
                raise ValueError('k_args should provide sigma for k_type: rbf')
        self.C=C
        self.k_type=k_type
        self.k_args=k_args
        self.iter_max=iter_max
        self.relax=relax
        self.mode=mode
        self.multi_class=multi_class
        self.eps=eps
                
    #超平面(这是原始定义，但为了使用smo优化和核函数，通常会被转化为下面一个函数)
    #当该式等于0时，X在超平面上；大于0时，X在法线正方向上;小于0时，X在法线负方向上
    # w为法线向量，所有位于超平面上的向量与该向量的点积为0（两向量垂直时点积为0）
    # b为偏置量，b为0时超平面经过原点
    #回归模式下则是拟合超平面去贴合所有数据点，与超平面的函数距离小于eps时回归准确
    def hyperplane_(self,w,X,b):
        return np.dot(X,w)+b
    
    #使用支持向量进行决策
    #该表达式是通过拉格朗日乘子法变换w后得到的
    #a,sv_y,sv_X是支持向量的乘子,y和X
    def decision_(self,X,a,sv_y,sv_X,b,k_type,k_args):
        return (a*sv_y*self.kernel_(X,sv_X,k_type,k_args)).T.sum(axis=0)+b
    
    #代价函数：铰链损失（回归模式下可能不叫这个名字）
    #此处相当于引入拉格朗日乘子后的目标优化函数
    #分类模式下分类正确且在间隔带外的点无损失
    #回归模式下在间隔带内的点无损失
    def cost_(self,y,u,w,C,mode='c'):
        if mode=='c':
            loss=1-y*u
        elif mode=='r':
            loss=np.abs(y-u)
        loss[loss<0]=0
        return C*loss.sum()+0.5*(w**2).sum()
    
    #划分
    #注：函数值大于等于0划分为1，函数值小于0划分为-1
    #    在逻辑回归中函数值通过sigmond转化为0~1的概率值，
    #    并将大于等于0.5的划分为1，小于0.5的划分为0，
    #    相比之下，svm相当于简化了划分方式
    def devide_(self,u):
        p_y=np.ones(len(u))
        p_y[u<0]=-1.
        return p_y.astype('int')
    
    #向量模长
    #注：即向量的长度，将该向量除以自身模长后会得到单位向量
    def module_(self,vector):
        return np.sqrt((vector**2).sum())
    
    #与超平面的距离    
    def margin_(self,w,u,y):
        return y*u/self.module_(w)
    
    #直接进行二次规划问题求解(此种解法无法应用核函数)(回归模式此处没有实现)
    #min(0.5*||w||**2),s.t.,yi*u≥1,i=1,…,n
    #注：此处使用scipy的约束优化方法,由于只能求解一个向量，
    #    所以把所有需要求解的参数合并到一个向量中求解
    def qp_optimize_(self,X,y,C):
        X_=np.ones((X.shape[0],X.shape[1]+1))
        X_[:,1:]=X
        n,m=X_.shape
        if C==np.inf:
            func=lambda thetas: 0.5*(thetas[1:]**2).sum()
            thetas0=np.ones(m)
            cons=({'type':'ineq','fun':lambda thetas: y*np.dot(X_,thetas)-1})
            result=opt.minimize(func,thetas0,constraints=cons)
            w,b=result.x[1:],result.x[0]
            return w,b
        else:
            func=lambda thetas: 0.5*(thetas[1:m]**2).sum()+C*thetas[m:].sum()
            thetas0=np.ones(m+n)
            cons=({'type':'ineq','fun':lambda thetas: y*np.dot(X_,thetas[:m])-1+thetas[m:]},
                  {'type':'ineq','fun':lambda thetas: thetas[m:]})
            result=opt.minimize(func,thetas0,constraints=cons)
            w,b=result.x[1:m],result.x[0]
            return w,b
    
    #使用SMO算法优化
    #该算法是坐标下降法的一个变种，每次选两个拉格朗日乘子进行优化    
    #p是一次项系数，y0是原目标值(在回归模式中,y会被替换)
    def smo_optimize_(self,X,y,C,p,relax,iter_max,k_type,k_args,mode,y0=None):
        #随机选取j (j!=i)
        def random_select(i,j_list):
            j,n=i,len(j_list)
            while j==i:
                j=j_list[int(random.uniform(0,n))]
            return j
        #初始化变量：
        #n(数据集大小),m(特征数量),
        #a(乘子序列),b(偏置),k(<x1,x2>内积矩阵),
        #optimize_h(a优化历史)，cost_h(代价值历史),sv_idx(支持向量索引)
        #w(权重),u(预测值),E(偏差),yu(y*u,预测正确时>0)
        n=len(X)
        if len(X.shape)==1:
            m=1
        else:
            m=X.shape[1] 
        a,b=np.zeros(n),0
        k=self.kernel_(X,X,k_type,k_args)
        optimize_h,cost_h=[],[]
        sv_idx=(a>0)
        w=np.zeros(m) 
        u=np.zeros(n)
        E=u-y*p
        yE=y*E
        if mode=='c':
            cost_h.append(self.cost_(y,u,w,C,mode))
        else:
            cost_h.append(self.cost_(y0,u[:n//2],w,C,mode))
        #print('current cost: %f'%(cost_h[-1]))
        #kkt条件
        a_kkt_1,a_kkt_2=((yE<-relax)&(a<C)),((yE>relax)&(a>0))
        a_kkt_3=((yE>=-relax)&(yE<=relax)&((a==C)|(a==0)))
        a_kkt=a_kkt_1|a_kkt_2|a_kkt_3
        #迭代更新
        entire=True
        for iter_id in range(iter_max):
            #首次迭代或是上次迭代未有a更新则遍历全部a，
            #否则优先遍历非边界的a
            #注：此方法能加快收敛
            if entire:
                i_list=np.where(a_kkt)[0]
            else:
                i_list=np.where((a>0)&(a<C)&a_kkt)[0]
            #遍历每个a作为a1
            a_change_n=0
            a_pass_n=0
            a_traverse_n=0
            #a_,b_=a.copy(),b
            for i in i_list:
                #选取不满足KKT条件的a1
                a_traverse_n+=1
                a1,y1,E1=a[i],y[i],E[i]
                #筛选可以作为a2的a
                #条件1：eta>0, eta=k11+k22-2*k12
                k11,k22_,k12_=k[i,i],np.diag(k),k[i,:]
                eta_=k11+k22_-2*k12_
                eta_ok=(eta_>0)
                #条件2：L!=H, L和H为a的下界和上界
                L,H=np.zeros(n),np.zeros(n)
                y_eq=(y1==y)
                L[~y_eq],H[~y_eq]=a[~y_eq]-a1,a[~y_eq]-a1+C
                L[y_eq],H[y_eq]=a[y_eq]+a1-C,a[y_eq]+a1
                L[L<0],H[H>C]=0,C
                LH_ok=(L<H)
                #条件3：|a-a_new|>=1e-5
                #a2_new如果超上下界需要截断
                eta_[eta_==0]=1
                a2_new_=a+y*(E1-E)/eta_
                a2_new_[a2_new_>H]=H[a2_new_>H]
                a2_new_[a2_new_<L]=L[a2_new_<L]
                change_ok=(np.abs(a2_new_-a)>=1e-5)
                valid_j=(eta_ok&LH_ok&change_ok)
                #a2不能和a1取同一个
                valid_j[i]=False
                #不存在可供选择的a2，跳过当前a1
                if ~valid_j.any():
                    a_pass_n+=1
                    continue
                #选取|E1-E2|最大的a2
                #E=u-y,u=wx+b=sum(a*sv_y*k(sv_x,x))+b
                #注：此方法能加快收敛
                E_n0=(E!=0)&valid_j
                if E_n0.any():
                    j_list=np.where(E_n0)[0]
                    E1_E2=np.abs(E1-E[E_n0])
                    j_=np.where(E1_E2==E1_E2.max())[0][0]
                    j=j_list[j_]
                #没有E2>0的对象时随机选取a2
                else:
                    #优先从违反kkt的a中选择a2，否则只从可行的a选择a2
                    if (a_kkt&valid_j).any():
                        j_list=np.where(a_kkt&valid_j)[0]
                        j=random_select(i,j_list)
                    elif valid_j.any():
                        j_list=np.where(valid_j)[0]
                        j=random_select(i,j_list)
                    else:
                        a_pass_n+=1
                        continue
                #更新a
                a2_new,a2,y2,E2=a2_new_[j],a[j],y[j],E[j]
                a1_new=a1+y1*y2*(a2-a2_new)
                a[i],a[j]=a1_new,a2_new
                a_change_n+=1
                #更新b
                k22,k12=k22_[j],k12_[j]
                b1=b-E1-y1*(a1_new-a1)*k11-y2*(a2_new-a2)*k12
                b2=b-E2-y1*(a1_new-a1)*k12-y2*(a2_new-a2)*k22
                #b_old=b
                if (a1_new<C)&(a1_new>0):
                    b=b1
                elif (a2_new<C)&(a2_new>0):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                #print("\na[%d]: %f->%f a[%d]: %f->%f b: %f->%f"
                    #%(i,a1,a1_new,j,a2,a2_new,b_old,b))
                #更新sv_idx,u,E,yu,a_kkt
                #注：计算函数输出时，通过sv_idx筛选支持向量，可以大幅减少运算量
                sv_idx=(a>0)
                u=self.decision_(X,a[sv_idx],y[sv_idx],X[sv_idx],b,k_type,k_args)
                E=u-y*p
                yE=y*E
                a_kkt_1,a_kkt_2=((yE<-relax)&(a<C)),((yE>relax)&(a>0))
                a_kkt_3=((yE>=-relax)&(yE<=relax)&((a==C)|(a==0)))
                a_kkt=a_kkt_1|a_kkt_2|a_kkt_3
                #遍历0<a<C子集时如果成功更新a则重新开始子集遍历
                #注：该方法能加快收敛
                '''
                if (entire==False)&(a_change_n>0):
                    break
                '''
            #a更新的数量，当前代价值
            optimize_h.append([len(i_list),a_traverse_n,a_pass_n,a_change_n])
            w=(a[sv_idx]*y[sv_idx]*X[sv_idx].T).sum(axis=1)
            if mode=='c':
                cost_h.append(self.cost_(y,u,w,C,mode))
            else:
                cost_h.append(self.cost_(y0,u[:n//2],w,C,mode))
            #print('current cost: %f'%(cost_h[-1]))
            #不存在违反kkt的a，提前结束
            if (entire==True)&(a_change_n==0):
                #print("\nwarning: no a can be changed")
                break
            #遍历过全集后遍历子集，遍历子集未能有a更新时重新遍历全集
            if entire:
                entire=False
            elif a_change_n==0:
                entire=True
        #print('\nfinal iter num is %d'%(iter_id+1))
        cost_h=pd.Series(cost_h)
        optimize_h=pd.DataFrame(optimize_h,columns=
                                     ['valid','traverse','pass','change'])
        return w,b,a,X,y,cost_h,optimize_h
    
    #核函数
    #每种核函数对应一种高维映射
    #核函数用于简化映射空间中的内积运算
    def kernel_(self,X1,X2,k_type,k_args):
        if k_type=='rbf':
            return self.rbf_kernel_(X1,X2,k_args['sigma'])
        elif k_type=='pol':
            return self.pol_kernel_(X1,X2,k_args['R'],k_args['d'])
        else:
            return self.lin_kernel_(X1,X2)
    
    #高斯核
    #exp(-||x1-x2||^2/(2*sigma^2))
    #即exp(-(<x1,x1>+<x2,x2>-2*<x1,x2>)/(2*sigma^2))
    #x1是中心点，sigma是宽度，在x1确定时关于x2的函数图像是一个凸包
    def rbf_kernel_(self,X1,X2,sigma):
        X1_2=(X1**2).T.sum(axis=0)
        X2_2=(X2**2).T.sum(axis=0)
        X1_X2=np.dot(X1,X2.T)
        if (len(X1.shape)>1)&(len(X2.shape)>1):
            X1_2=X1_2.reshape((len(X1_2),1))
            X2_2=X2_2.reshape((1,len(X2_2)))
        return np.e**(-(X1_2+X2_2-2*X1_X2)/(2*sigma**2))
    
    #多项式核
    #(<x1,x2>+R)^d
    def pol_kernel_(self,X1,X2,R,d):
        return (np.dot(X1,X2.T)+R)**d
    
    #线性核
    #<x1,x2>
    def lin_kernel_(self,X1,X2):
        return np.dot(X1,X2.T)
    
    #X输入校验
    def check_input_X_(self,X):
        if type(X)==type(pd.Series()):
            X=X.to_frame()
        check_type('X',type(X),type(pd.DataFrame()))
        type_list=[np.int64,np.float64]
        for i in range(len(X.columns)):
            check_type('column %d in X'%i,X.dtypes[i],type_list)
        return X
    
    #y,p_y输入校验
    def check_input_y_(self,y,name='y',mode='c'):
        check_type(name,type(y),type(pd.Series()))
        if mode=='r':
            type_list=[np.int64,np.float64]
            check_type(name,y.dtype,type_list)
            return y
        elif mode=='c':
            return y.astype('str')
    
    #拟合
    def fit(self,X,y,keep_nonsv=False,show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合支持向量机
        
        Note: 输入数据必须全部是数值类型，其他类型自行预处理
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        y: 目标向量,Series类型
        keep_nonsv: 是否保留非支持向量点，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        '''
        start=time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('show_time',type(show_time),type(True))
            X=self.check_input_X_(X)
            y=self.check_input_y_(y,mode=self.mode)
            check_index_match(X,y,'X','y')
        #分类
        n,m=len(y),len(X.columns)
        if self.mode=='c':
            #根据类别数量处理y
            values=y.drop_duplicates().sort_values().tolist()
            classes_n=len(values)
            if classes_n>=0.5*len(y):
                print('\nwarning: too many classes in y')
            if classes_n<=1:
                raise ValueError('classes_n in y should >=2')
            self.classes=values
            #smo优化
            #二分类
            if classes_n==2:
                y_=np.ones(n)
                y_[y==values[0]]=-1
                p=np.ones(n)
                w,b,a,sv_X,sv_y,cost_h,optimize_h=\
                    self.smo_optimize_(X.values,y_,iter_max=self.iter_max,C=self.C,p=p,mode=self.mode,
                                       k_type=self.k_type,k_args=self.k_args,relax=self.relax)
                if keep_nonsv==False:
                    sv_idx=(a!=0)
                    self.a,self.sv_X,self.sv_y=[a[sv_idx]],[sv_X[sv_idx]],[sv_y[sv_idx]]
                else:
                    self.a,self.sv_X,self.sv_y=[a],[sv_X],[sv_y]
                self.w,self.b=[w],[b]
                self.cost_h,self.optimize_h=[cost_h],[optimize_h]
            #多分类
            else:
                Y=dp.dummy_var(y)
                self.w,self.b,self.a,self.sv_X,self.sv_y=[],[],[],[],[]
                self.cost_h,self.optimize_h=[],[]
                p=np.ones(n)
                if self.multi_class=='ovr':
                    for i in range(classes_n):
                        print('\nfitting classifier %d ---'%i)
                        y_=Y.iloc[:,i].values
                        y_[y_==0]=-1
                        w,b,a,sv_X,sv_y,cost_h,optimize_h=\
                            self.smo_optimize_(X.values,y_,iter_max=self.iter_max,C=self.C,p=p,mode=self.mode,
                                               k_type=self.k_type,k_args=self.k_args,relax=self.relax)
                        if keep_nonsv==False:
                            sv_idx=(a!=0)
                            self.a.append(a[sv_idx]),self.sv_X.append(sv_X[sv_idx]),self.sv_y.append(sv_y[sv_idx])
                        else:
                            self.a.append(a),self.sv_X.append(sv_X),self.sv_y.append(sv_y)
                        self.w.append(w),self.b.append(b),
                        self.cost_h.append(cost_h),self.optimize_h.append(optimize_h)
                elif self.multi_class=='tree':
                    #生成所有节点
                    def split_classes(svm_id,classes):
                        classes_n=len(classes)
                        if classes_n==2:
                            left_cid,right_cid=-1,-1
                        elif classes_n==3:
                            left_cid,right_cid=-1,10*svm_id+1
                        else:
                            left_cid,right_cid=10*svm_id+1,10*svm_id+2
                        nodes_info=[[svm_id,classes[:classes_n//2],1,left_cid],
                                    [svm_id,classes[classes_n//2:],-1,right_cid]]
                        if classes_n==3:
                            nodes_info+=split_classes(10*svm_id+1,classes[classes_n//2:])
                        elif classes_n>3:
                            nodes_info+=split_classes(10*svm_id+1,classes[:classes_n//2])
                            nodes_info+=split_classes(10*svm_id+2,classes[classes_n//2:])
                        if svm_id==1:
                            nodes_info=pd.DataFrame(nodes_info,columns=['svm','classes','y','next'])
                            svm_ids=nodes_info['svm'].drop_duplicates().sort_values().tolist()
                            for i in range(len(svm_ids)):
                                nodes_info.loc[nodes_info['svm']==svm_ids[i],'svm']=i
                                nodes_info.loc[nodes_info['next']==svm_ids[i],'next']=i
                            return nodes_info
                        else:
                            return nodes_info
                    tree=split_classes(1,self.classes)
                    #每个节点训练一个分类器
                    for i in range(tree['svm'].max()+1):
                        print('\nfitting classifier %d ---'%i)
                        left_classes=tree.loc[(tree['svm']==i)&(tree['y']==1),'classes'].values[0]
                        right_classes=tree.loc[(tree['svm']==i)&(tree['y']==-1),'classes'].values[0]
                        classes_=left_classes+right_classes
                        in_classes=(y.isin(classes_))
                        X_=X[in_classes].values
                        y_=y[in_classes].values
                        y__=np.ones(len(y_))
                        y__[np.isin(y_,right_classes)]=-1
                        p_=np.ones(len(y_))
                        w,b,a,sv_X,sv_y,cost_h,optimize_h=\
                            self.smo_optimize_(X_,y__,iter_max=self.iter_max,C=self.C,p=p_,mode=self.mode,
                                               k_type=self.k_type,k_args=self.k_args,relax=self.relax)
                        if keep_nonsv==False:
                            sv_idx=(a!=0)
                            self.a.append(a[sv_idx]),self.sv_X.append(sv_X[sv_idx]),self.sv_y.append(sv_y[sv_idx])
                        else:
                            self.a.append(a),self.sv_X.append(sv_X),self.sv_y.append(sv_y)
                        self.w.append(w),self.b.append(b)
                        self.cost_h.append(cost_h),self.optimize_h.append(optimize_h)
                    self.tree=tree    
                else:
                    raise ValueError('unsupported multi_class')
        #回归
        #参考了libsvm的实现，将原本要优化的两组a合并到一组进行优化
        elif self.mode=='r':
            X_,y_,p=np.zeros((2*n,m)),np.zeros(2*n),np.zeros(2*n)
            X_[:n,:],X_[n:,:]=X.values,X.values
            y_[:n],y_[n:]=1,-1
            p[:n],p[n:]=-self.eps+y.values,-self.eps-y.values
            w,b,a_,sv_X_,sv_y_,cost_h,optimize_h=\
                self.smo_optimize_(X_,y_,iter_max=self.iter_max,C=self.C,p=p,mode=self.mode,
                                   k_type=self.k_type,k_args=self.k_args,relax=self.relax,y0=y.values)
            a=a_[:n]-a_[n:]
            sv_X,sv_y=X.values,np.ones(n)
            if keep_nonsv==False:
                sv_idx=(a!=0)
                self.a,self.sv_X,self.sv_y=[a[sv_idx]],[sv_X[sv_idx]],[sv_y[sv_idx]]
            else:
                self.a,self.sv_X,self.sv_y=[a],[sv_X],[sv_y]
            self.w,self.b=[w],[b]
            self.cost_h,self.optimize_h=[cost_h],[optimize_h]
        else:
            raise ValueError('unsupported mode')
        time_cost=time.clock()-start
        if show_time==True:
            print('\ntime used for training: %f'%time_cost)
        
    #预测
    def predict(self,X,return_u=False,show_time=False,check_input=True):
        '''\n
        Function: 对输入数据进行预测
        
        Parameters
        ----------
        X: 特征矩阵,DataFrame类型
        return_u: 是否返回函数间隔，bool类型，默认False
        show_time: 是否显示时间开销，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测值向量，Series类型
        -------
        '''
        start=time.clock()
        #输入校验
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            X=self.check_input_X_(X)
            check_feats_match(X.columns,self.sv_X[0][0,:],'features in X','support vector',mode='len')
        #计算函数间隔
        if self.mode=='c':
            classes_n=len(self.classes)
            classifiers_n=len(self.a)
            #二分类
            if classes_n==2:
                u=self.decision_(X.values,self.a[0],self.sv_y[0],self.sv_X[0],
                                 self.b[0],self.k_type,self.k_args)
            #多分类
            else:
                u=np.zeros((len(X),classes_n))
                #ovr
                if classifiers_n==classes_n:
                    for i in range(classes_n):
                        u_=self.decision_(X.values,self.a[i],self.sv_y[i],self.sv_X[i],
                                          self.b[i],self.k_type,self.k_args)
                        u[:,i]+=u_
                #tree
                elif classifiers_n<classes_n:
                    tree=self.tree
                    flow=np.zeros(len(X)).astype('int')
                    for i in range(len(self.a)):
                        ft=(flow==i)
                        X_=X[ft].values
                        u_=self.decision_(X_,self.a[i],self.sv_y[i],self.sv_X[i],
                                          self.b[i],self.k_type,self.k_args)
                        left_child=tree.loc[(tree['svm']==i)&(tree['y']==1),'next'].values[0]
                        right_child=tree.loc[(tree['svm']==i)&(tree['y']==-1),'next'].values[0]
                        left_classes=tree.loc[(tree['svm']==i)&(tree['y']==1),'classes'].values[0]
                        right_classes=tree.loc[(tree['svm']==i)&(tree['y']==-1),'classes'].values[0]
                        flow_=flow[ft]
                        if left_child!=-1:
                            flow_[u_>=0]=left_child
                        else:
                            class_idx=self.classes.index(left_classes[0])
                            u__=np.zeros(len(X_))
                            u__[u_>=0]=1.
                            u[ft,class_idx]+=u__
                        if right_child!=-1:
                            flow_[u_<0]=right_child
                        else:
                            class_idx=self.classes.index(right_classes[0])
                            u__=np.zeros(len(X_))
                            u__[u_<0]=1.
                            u[ft,class_idx]+=u__
                        flow[flow==i]=flow_
                    self.flow=flow
                else:
                    raise ValueError('too many classifiers')
            if return_u==False:
                #划分类别
                if classes_n==2:
                    p_y=self.devide_(u)
                    p_y[p_y==1]=self.classes[1]
                    p_y[p_y==-1]=self.classes[0]
                else:
                    u_max=u.max(axis=1)
                    max_idx=(u.T==u_max).T.astype('int')
                    classes_idx=np.array(range(u.shape[1]))
                    p_y_=np.dot(max_idx,classes_idx).astype('int')
                    p_y=np.full(len(p_y_),'')
                    for i in range(classes_n):
                        p_y[p_y_==i]=self.classes[i]
                p_y=pd.Series(p_y,index=X.index)
                time_cost=time.clock()-start
                if show_time==True:
                    print('\ntime used for predict: %f'%time_cost)
                return p_y
            else:
                if classes_n==2:
                    u=pd.DataFrame(np.c_[u,-u],columns=self.classes,index=X.index)
                else:
                    u=pd.DataFrame(u,columns=self.classes,index=X.index)
                time_cost=time.clock()-start
                if show_time==True:
                    print('\ntime used for predict: %f'%time_cost)
                return u
        elif self.mode=='r':
            u=self.decision_(X.values,self.a[0],self.sv_y[0],self.sv_X[0],
                             self.b[0],self.k_type,self.k_args)
            u=pd.Series(u,index=X.index)
            time_cost=time.clock()-start
            if show_time==True:
                print('\ntime used for predict: %f'%time_cost)
            return u
        else:
            raise ValueError('unsupported mode')
        
    #评估
    def assess(self,y,p_y,return_dist=False,check_input=True):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:观测值，Series类型
        p_y:预测值，Series类型
        return_dist: 是否返回预测分布，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        mode=self.mode
        #校验输入
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            y=self.check_input_y_(y,name='y',mode=mode)
            p_y=self.check_input_y_(p_y,name='p_y',mode=mode)
            check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='c':
            return stats.accuracy(y,p_y,return_dist,self.classes)
        elif mode=='r':
            return stats.r_sqr(y,p_y)