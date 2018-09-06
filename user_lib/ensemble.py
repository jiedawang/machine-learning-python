# -*- coding: utf-8 -*-
import time
import user_lib.deci_tree as dt
import pandas as pd
import numpy as np
import user_lib.statistics as stats
import user_lib.data_prep as dp
from user_lib.check import check_type,check_limit,check_index_match,check_items_match

#随机森林
#基于Bagging集成学习原理，基本单元一般为决策树（可以替换为其他模型）
class RandomForest:
    '''\n  
    Note: 随机森林(决策树)，简称RF，支持分类和回归
     
    Parameters
    ----------
    mode: 模式，分类->'c'，回归->'r'，默认'c'
    units_n: 集成单元的数量，int类型(>=1)，默认10
    units_type: 集成单元的类型，str类型(id3,c4.5,cart)，
                'id3'->分类，离散特征+离散输出，
                 'c4.5'->分类，离散或连续特征+离散输出，
                 'cart'->分类或回归，离散或连续特征+离散或连续输出
                 默认值'cart'
    depth_max: 最大深度，int类型(>=1)，None表示无限制，默认值None
    split_sample_n: 分裂所需最少样本数，int类型(>=2)，默认值2
    leaf_sample_n: 叶节点所需最少样本数，int类型(>=1)，默认值1
    features_use: 每次使用的特征数量，str/float/int类型
                 'all'->全量，
                 'sqrt'->总数量的平方根，
                 'log2'->总数量的2的对数，
                 float->总数量的相应比例，区间(0.0,1.0)，
                 int->固定数量，区间[1,feature_num]，
                 默认值'sqrt'
    features_reuse: 是否允许一个特征重复使用，bool类型，默认值True
    ----------
    
    Attributes
    ----------
    units: 集成单元列表，list(object)类型
    features: 特征列表，list(str)类型
    classes: 分类标签列表，list(str)类型，分类模式下有效
    oob_score: 袋外误差，float类型
    units_oob_score: 集成单元袋外误差，list(float)类型
    ----------
    '''
    
    def __init__(self,mode='c',units_n=10,units_type='cart',
                 depth_max=None,split_sample_n=2,leaf_sample_n=1,
                 features_use='sqrt',features_reuse=True):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        mode=mode.lower()
        check_limit('mode',mode in mode_list,str(mode_list))
        
        check_type('units_n',type(units_n),type(0))
        check_limit('units_n',units_n>=1,'value>=1')
        
        check_type('units_type',type(units_type),type(''))
        type_list=['id3','c4.5','cart']
        units_type=units_type.lower()
        check_limit('units_type',units_type in type_list,str(type_list))
        #保存参数
        self.unit_test=dt.DecisionTree(mode=mode,model_type=units_type,depth_max=depth_max,
                                       split_sample_n=split_sample_n,leaf_sample_n=leaf_sample_n,
                                       features_use=features_use,features_reuse=features_reuse)
        self.mode=mode
        self.units_n=units_n
        self.units_type=units_type
        self.depth_max=depth_max
        self.split_sample_n=split_sample_n
        self.leaf_sample_n=leaf_sample_n
        self.features_use=features_use
        self.features_reuse=features_reuse

    #拟合
    def fit(self,X,y,show_time=False):
        '''\n
        Function: 使用输入数据拟合随机森林
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签
              
        Description: 
            (a)从大小为N的训练集中随机且有放回地抽取N个样本(bootstrap sample)作为每棵树的训练集
            (b)每个节点分裂时，从总共M个特征中随机地选取m个特征子集(m<<M)，从这m个特征中选择最优分裂
            (c)每棵树都尽最大程度的生长，并且没有剪枝过程
            (d)使用未被当前树训练集选中的数据作为测试集计算泛化误差(out of bag error)

        Parameters
        ----------
        X: 特征列，DataFrame类型
        y: 目标列，Series类型
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        '''
        start = time.clock()
        check_type('show_time',type(show_time),type(True))
        #校验X,y输入
        X,self.continuity_X,self.mapping_X,X0=\
            self.unit_test.check_input_X_(X,to_index=True,return_source=True)
        y,self.continuity_y,self.mapping_y,y0=\
            self.unit_test.check_input_y_(y,to_index=True,return_source=True)
        #校验X,y输入是否匹配
        check_index_match(X,y,'X','y')
        #计算每次分裂使用的特征数量上限
        self.features_use_n=self.unit_test.compute_features_use_n_(len(X.columns),self.features_use)
        #集成单元序列和集成单元oob评分列表
        self.units,self.units_oob_score=[],[]
        self.features=X.columns.tolist()
        #oob袋外数据预测矩阵初始化
        if self.mode=='c':
            self.classes=y0.drop_duplicates().sort_values().astype('str').tolist()
            oob_predict=pd.DataFrame(np.zeros((len(X.index),len(self.classes))),
                                     index=X.index,columns=self.classes)
        elif self.mode=='r':
            self.classes=[]
            oob_predict=pd.Series(np.zeros(len(X.index)),index=X.index)
        oob_trees_n=pd.Series(np.zeros(len(X.index)),index=X.index)
        #逐个拟合（有尝试过使用原生python的多进程和多线程，但效果不佳）
        for i in range(self.units_n):
            if show_time==True:
                print('\nfitting with unit %d ---'%i)
            #随机有放回抽样生成训练集,大小不变,同时提取oob样本
            #注：注意重新生成一下索引，有放回抽样会产生重复索引
            X_=X.sample(frac=1.0,replace=True)
            y_=y[X_.index]
            iob_index=X_.index.drop_duplicates()
            oob_X0_=X0[~X0.index.isin(iob_index)]
            oob_y0_=y0[oob_X0_.index]
            X_.index=range(len(X_))
            y_.index=range(len(y_))
            #构建并拟合模型
            unit=dt.DecisionTree(mode=self.mode,model_type=self.units_type,depth_max=self.depth_max,
                                 split_sample_n=self.split_sample_n,leaf_sample_n=self.leaf_sample_n,
                                 features_use=self.features_use,features_reuse=self.features_reuse)
            unit.continuity_X,unit.mapping_X=self.continuity_X,self.mapping_X
            unit.continuity_y,unit.mapping_y=self.continuity_y,self.mapping_y
            unit.features_use_n=self.features_use_n
            unit.fit(X_,y_,show_time=show_time,check_input=False)
            #obb预测
            if self.mode=='c':
                p_y_=unit.predict(oob_X0_,return_proba=True,check_input=False)
                p_y_0=unit.choose_class_(p_y_,self.classes)
                score_=unit.assess(oob_y0_,p_y_0,check_input=False)
                oob_predict.loc[p_y_.index,:]+=p_y_
            elif self.mode=='r':
                p_y_=unit.predict(oob_X0_,check_input=False)
                score_=unit.assess(oob_y0_,p_y_,check_input=False)
                oob_predict.loc[p_y_.index]+=p_y_
            oob_trees_n.loc[p_y_.index]+=1
            self.units_oob_score.append(score_)
            #添加进随机森林
            unit.mapping_X,unit.mapping_y=None,None
            self.units.append(unit)
        #oob整体预测
        #注：由于存在少量数据不满足oob条件所以没有预测结果，需要筛去
        boolIdx=(oob_trees_n!=0.0)
        if self.mode=='c':
            oob_predict=self.unit_test.choose_class_(oob_predict[boolIdx],self.classes)
        elif self.mode=='r':
            oob_predict=oob_predict[boolIdx]/oob_trees_n[boolIdx]
        score=self.unit_test.assess(y0[boolIdx],oob_predict,mode=self.mode,check_input=False)
        self.oob_score=score
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for trainning: %f'%(end-start))
    
    #预测        
    def predict(self,X,units=None,mode=None,units_result=False,
                return_proba=False,return_paths=False,show_time=False):
        '''\n
        Function: 使用输入数据和所有集成单元进行预测，没有输入集成单元时使用内部缓存
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        units: 集成单元，list(DecitionTree)类型，默认调用内部缓存
        mode:模式，str类型，默认使用内部集成单元的属性，
             'c'->分类，'r'->回归
        units_result: 是否返回每个单元的分类结果，bool类型，默认False
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False，
                      分类概率不能直接用于评估
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率，Series/DataFrame类型
        1: 各个单元的预测的分类/分类概率，list(Series)/list(DataFrame)类型
        2: 所有数据最终抵达的节点和决策路径，list(DataFrame)类型
        -------
        '''
        start = time.clock()
        #校验参数
        if type(units)==type(None):
            units=self.units
        if type(mode)==type(None):
            mode=units[0].tree.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        check_type('units',type(units),type([]))
        check_type('element in units',type(units[0]),type(dt.DecisionTree()))
        check_type('return_proba',type(return_proba),type(True))
        check_type('return_paths',type(return_paths),type(True))
        check_type('show_time',type(show_time),type(True))
        X,continuity_X=self.unit_test.check_input_X_(X)
        features=[]
        for unit in units:
            features+=unit.tree.features
        features=list(set(features))
        check_items_match(X.columns,features,'X','unit','features',mode='right')
        #分类模式先求分类概率，回归模式直接求回归值
        n=len(X)
        if mode=='c':
            classes=units[0].tree.classes
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    np.zeros((n,len(classes))),
                    index=X.index,columns=classes)
        elif mode=='r':
            #定义存放回归值的Series
            p_y=pd.Series(np.zeros(n),index=X.index)
        #逐个调用每个单元进行预测
        units_p_y,units_paths=[],[]
        for i in range(len(units)):
            if show_time==True:
                print('\npredicting with unit %d ---'%i)
            if return_paths==True:
                p_y_,paths=units[i].predict(X,return_proba=True,return_paths=True,
                                            show_time=show_time,check_input=False)
                units_paths.append(paths)
            else:
                p_y_=units[i].predict(X,return_proba=True,return_paths=False,
                                      show_time=show_time,check_input=False)
            #整体结果累加
            p_y+=p_y_
            #记录单元预测结果
            if units_result==True:
                if (mode=='c')&(return_proba==False):
                    p_y_=units[i].choose_class_(p_y_,classes)
                units_p_y.append(p_y_)
        #分类模式下归一化，回归模式下求平均值，公式一致
        p_y=p_y/len(units)
        #分类模式下返回分类概率或最终分类
        if (mode=='c')&(return_proba==False):
            p_y=self.unit_test.choose_class_(p_y,classes)
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for predict: %f'%(end-start))
        if units_result==True:
            if return_paths==True:
                return p_y,units_p_y,paths
            else:
                return p_y,units_p_y
        else:
            if return_paths==True:
                return p_y,paths
            else:
                return p_y

    #评估
    def assess(self,y,p_y,mode=None):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:观测值，Series类型
        p_y:预测值，Series类型
        mode:模式，str类型，默认使用内部集成单元的属性，
             'c'->分类，'r'->回归
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        #校验输入
        if type(mode)==type(None):
            mode=self.units[0].tree.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        y,continuity_y=self.unit_test.check_input_y_(y,name='y')
        p_y,continuity_p_y=self.unit_test.check_input_y_(p_y,name='p_y')
        check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='c':
            return stats.accuracy(y,p_y)
        elif mode=='r':
            return stats.r_sqr(y,p_y)
    
    #以下几个重要概念可以了解下
    #边缘函数，泛化误差，强度，平均相关度，泛化误差上界，c/s2比率
    
    #随机选择
    #从1~k分别随机抽取指定数量的集成单元，选出在测试集上表现最好的子集
    def random_selection_(self,test_X,test_y,units):
        best_subset,best_score=[],0.0
        units_idx=pd.Series(range(len(units)))
        for n in range(1,len(units)+1):
            units_idx_=units_idx.samples(n)
            units_=[units[i] for i in units_idx_]
            p_y_=self.predict(test_X,units=units_)
            score_=self.assess(test_y,p_y_,mode=units[0].tree.mode)
            if score_>best_score:
                best_score=score_
                best_subset=units_
        return best_subset
    
    #袋外样本准确率选择
    #先根据obb评分对集成单元排序，取前1~k个集成单元作为子集，选出在测试集上表现最好的子集
    def oob_selection_(self,test_X,test_y,units,units_oob_score):
        best_subset,best_score=[],0.0
        units_idx=pd.Series(range(len(units)))
        units_oob_score=pd.Series(units_oob_score)
        units_idx=units_idx[units_oob_score.sort_values().index]
        for n in range(1,len(units)+1):
            units_idx_=units_idx[:n]
            units_=[units[i] for i in units_idx_]
            p_y_=self.predict(test_X,units=units_)
            score_=self.assess(test_y,p_y_,mode=units[0].tree.mode)
            if score_>best_score:
                best_score=score_
                best_subset=units_
        return best_subset
    
    #模型选择
    def selection(self,test_X,test_y,units=None,units_oob_score=None,
                  use='oob',return_units=False,show_time=False):
        '''\n
        Function: 在生成好的模型上进行选择，筛选出集成单元的一个子集
        
        Notes: 作用类似于决策树的剪枝，通过一些规则生成可选子集，
               再通过在测试集上的表现选择最优的一个，能够得到更简单且泛化能力更强的模型
        
        Parameters
        ----------
        test_X: 测试集特征列，DataFrame类型
        test_y: 测试集目标列，Series类型
        units: 集成单元，list(DecitionTree)类型
        units_oob_score: 集成单元obb评分，list(float)类型
        use: 使用的选择方法，str类型，默认'oob'
             'rd'->随机选择，'oob'->oob选择
        return_units: 是否以返回值形式给到选择后的集成单元，bool类型，默认False
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        start = time.clock()
        if units==None:
            units=self.units
        if units_oob_score==None:
            units_oob_score=self.units_oob_score
        #输入校验
        check_type('units',type(units),type([]))
        check_type('element in units',type(units[0]),type(dt.DecisionTree()))
        check_type('units_oob_score',type(units_oob_score),type([]))
        check_type('element in units_oob_score',type(units_oob_score[0]),[type(0.0),np.float64])
        check_type('use',type(use),type(''))
        check_type('return_units',type(return_units),type(True))
        use_list=['rd','oob']
        check_limit('use',use in use_list,str(use_list))
        test_X,continuity_X=self.unit_test.check_input_X_(test_X,'test_X')
        test_y,continuity_y=self.unit_test.check_input_y_(test_y,'test_y')
        check_index_match(test_X,test_y,'test_X','test_y')
        features=[]
        for unit in units:
            features+=unit.tree.features
        features=list(set(features))
        check_items_match(test_X.columns,features,'test_X','tree','features',mode='right')
        #选择
        if use=='rd':
            subset=self.random_selection_(test_X,test_y,units)
        elif use=='oob':
            subset=self.oob_selection_(test_X,test_y,units,units_oob_score)
        end = time.clock()
        if show_time==True:
            print('\ntime used for selection:%f'%(end-start))
        if return_units==False:
            self.units=subset
        else:
            return subset

#自适应提升
class AdaBoost:
    '''\n  
    Note: 自适应提升(决策树)，支持分类和回归
     
    Parameters
    ----------
    mode: 模式，分类->'c'，回归->'r'，默认'c'
    iter_max: 迭代优化次数，int类型(>=1)，默认10
    units_type: 集成单元类型，str类型，目前仅一种'cart'
    depth_max: 最大深度，int类型，0表示采用默认值(分类->1,回归->3)，默认0
    learning_rate: 学习率，float类型(>0.0,<=1.0),默认1.0
    ----------
    
    Attributes
    ----------
    units: 集成单元列表，list(object)类型
    features: 特征列表，list(str)类型
    classes: 分类标签列表，list(str)类型，分类模式下有效
    units_weight: 集成单元权重，list(float)类型
    fit_h: 拟合过程中的观测值/预测值/样本权重，list(DataFrame)类型
    units_error: 集成单元误差和加权误差，DataFrame类型
    ----------
    '''
    
    def __init__(self,mode='c',iter_max=10,units_type='cart',depth_max=0,
                 learning_rate=1.0):
        
        #校验参数类型和取值
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        mode=mode.lower()
        check_limit('mode',mode in mode_list,str(mode_list))
        
        check_type('iter_max',type(iter_max),type(0))
        check_limit('iter_max',iter_max>=1,'value>=1')
        
        check_type('units_type',type(units_type),type(''))
        type_list=['cart']
        units_type=units_type.lower()
        check_limit('units_type',units_type in type_list,str(type_list))
        
        if type(depth_max)==type(0):
            if depth_max==0:
                if mode=='r':
                    depth_max=3
                elif mode=='c':
                    depth_max=1
        
        check_type('learning_rate',type(learning_rate),type(0.0))
        check_limit('learning_rate',learning_rate>0.0,'value>0.0')
        check_limit('learning_rate',learning_rate<=1.0,'value<=1.0')
        
        #保存参数
        #注：此处depth_max参考了sklearn,尝试过回归也用depth_max=1，效果很糟糕
        self.unit_test=dt.DecisionTree(mode=mode,model_type=units_type,
                                       depth_max=depth_max)
        self.mode=mode
        self.iter_max=iter_max
        self.units_type=units_type
        self.learning_rate=learning_rate
        self.depth_max=depth_max

    #个体误差
    def errors_(self,y,p_y,mode):
        #分类模式下1表示正确分类，0(或-1)表示错误分类
        if mode=='c':
            return y!=p_y
        #回归模式下为归一化的平方误差(也可以是绝对值或指数)
        elif mode=='r':
            re2=(y-p_y)**2
            return re2/re2.max()
    
    #指数代价函数(公式推导的源头，算法中并未使用)
    def cost_(self,errors):
        return np.sum(np.e**errors)
    
    #加权整体误差
    def wgt_err_(self,errors,sample_weight,mode):
        #加权错分率
        if mode=='c':
            return (sample_weight*errors).sum()/sample_weight.sum()
        #加权平方误差
        elif mode=='r':
            return (sample_weight*errors).sum()/sample_weight.sum()

    #新预测器权重
    def unit_weight_(self,learning_rate,wgt_err,k,mode):
        #多分类采用SUMME,第二项将对wgt_err的要求降到1-1/k(即错误率与随机猜测持平时a为0)
        if mode=='c':
            return learning_rate*(np.log((1-wgt_err)/wgt_err)+np.log(k-1))
        elif mode=='r':
            return learning_rate*np.log((1-wgt_err)/wgt_err)
    
    #更新样本权重
    def sample_weight_(self,errors,unit_weight,sample_weight,mode):
        if mode=='c':
            new_sample_weight=sample_weight*(np.e**(unit_weight*errors))
        elif mode=='r':
            new_sample_weight=sample_weight*(np.e**(unit_weight*(errors-1)))
        #归一化
        new_sample_weight=new_sample_weight/new_sample_weight.sum()
        return new_sample_weight
    
    #拟合
    def fit(self,X,y,show_time=False):
        '''\n
        Function: 使用输入数据拟合自适应提升(决策树)
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签
              
        Description: 对于m=1,2,…,M
            (a)使用具有权值分布Dm的训练数据集进行学习，得到弱学习器Gm(x)
            (b)计算Gm(x)在训练数据集上的误差率
            (c)计算Gm(x)在强学习器中所占的权重：
            (d)更新训练数据集的权值分布（需要归一化，使样本的概率分布和为1）

        Parameters
        ----------
        X: 特征列，DataFrame类型
        y: 目标列，Series类型
        show_time: 是否显示耗时，bool类型，默认值False
        ----------  
        '''
        start = time.clock()
        check_type('show_time',type(show_time),type(True))
        #校验X,y输入
        X,self.continuity_X,self.mapping_X,X0=\
            self.unit_test.check_input_X_(X,to_index=True,return_source=True)
        y,self.continuity_y,self.mapping_y,y0=\
            self.unit_test.check_input_y_(y,to_index=True,return_source=True)
        #校验X,y输入是否匹配
        check_index_match(X,y,'X','y')
        feature_use_n=len(X.columns)
        #特征/分类标签
        self.features=X.columns.tolist()
        if self.mode=='c':
            self.classes=y0.drop_duplicates().sort_values().astype('str').tolist()
            k=len(self.classes)
        elif self.mode=='r':
            k=0
        #迭代训练弱学习器
        sample_weight=np.ones(len(X))
        sample_weight=pd.Series(sample_weight/len(sample_weight),index=X.index)
        self.units,self.units_weight,self.units_error,self.fit_h=[],[],[],[]
        for i in range(self.iter_max):
            if show_time==True:
                print('\nfitting with unit %d ---'%i)
            #构建并拟合模型
            unit=dt.DecisionTree(mode=self.mode,model_type=self.units_type,
                                 depth_max=self.depth_max)
            unit.continuity_X,unit.mapping_X=self.continuity_X,self.mapping_X
            unit.continuity_y,unit.mapping_y=self.continuity_y,self.mapping_y
            unit.features_use_n=feature_use_n
            unit.fit(X,y,sample_weight,show_time=show_time,check_input=False)
            #计算当前弱学习器加权误差和预测器权重
            mode=unit.tree.mode
            p_y=unit.predict(X0,check_input=False)
            fit_h_=pd.DataFrame()
            fit_h_['y'],fit_h_['p_y'],fit_h_['sp_wgt']=y0,p_y,sample_weight
            self.fit_h.append(fit_h_)
            errors=self.errors_(y0,p_y,mode)
            wgt_err=self.wgt_err_(errors,sample_weight,mode)
            error=self.wgt_err_(errors,self.fit_h[0]['sp_wgt'],mode)
            #误差达到0，不需要继续训练
            if wgt_err==0.0:
                if show_time==True:
                    print('\nwarning: early stopping')
                break
            unit_weight=self.unit_weight_(self.learning_rate,wgt_err,k,mode)
            #权重大于0表示弱学习器优于随即猜测
            if unit_weight>0:
                self.units_weight.append(unit_weight)
                self.units_error.append([error,wgt_err])
                #添加进强学习器
                unit.continuity_X,unit.mapping_X=None,None
                unit.continuity_y,unit.mapping_y=None,None
                self.units.append(unit)
                #更新样本权重
                if i<self.iter_max-1:
                    sample_weight=self.sample_weight_(errors,unit_weight,sample_weight,mode)
                    sample_weight=pd.Series(sample_weight,index=X.index)
            else:
                if show_time==True:
                    print('\nwarning: unit is worse than random, discard')
        self.units_error=pd.DataFrame(self.units_error,columns=['err','wgt_err'])
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for trainning: %f'%(end-start))
    
    #预测        
    def predict(self,X,units=None,mode=None,units_weight=None,units_result=False,
                return_proba=False,return_paths=False,show_time=False):
        '''\n
        Function: 使用输入数据和所有集成单元进行预测，没有输入集成单元时使用内部缓存
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        units: 集成单元，list(DecitionTree)类型，默认调用内部缓存
        mode: 模式，分类->'c'，回归->'r'，默认'c'
        units_weight: 集成单元权重，list(float)类型，默认调用内部缓存
        units_result: 是否返回每个单元的分类结果，bool类型，默认False
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False，
                      分类概率不能直接用于评估
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率，Series/DataFrame类型
        1: 各个单元的预测的分类/分类概率，list(Series)/list(DataFrame)类型
        2: 所有数据最终抵达的节点和决策路径，list(DataFrame)类型
        -------
        '''
        start = time.clock()
        #校验参数
        if type(units)==type(None):
            units=self.units
            
        if type(units_weight)==type(None):
            units_weight=self.units_weight
            
        if type(mode)==type(None):
            mode=units[0].tree.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        
        check_type('units',type(units),type([]))
        if len(units)==0:
            raise ValueError('lack of units')
        check_type('element in units',type(units[0]),type(dt.DecisionTree()))
        
        check_type('units_weight',type(units_weight),type([]))
        if len(units_weight)==0:
            raise ValueError('lack of units_weight')
        check_type('element in units_weight',type(units_weight[0]),[type(0.0),np.float64])
        
        check_type('return_proba',type(return_proba),type(True))
        check_type('return_paths',type(return_paths),type(True))
        check_type('show_time',type(show_time),type(True))
        
        X,continuity_X=self.unit_test.check_input_X_(X)
        features=[]
        for unit in units:
            features+=unit.tree.features
        features=list(set(features))
        check_items_match(X.columns,features,'X','unit','features',mode='right')
        check_items_match(units,units_weight,'units','units_weight','numbers',mode='len')

        #分类模式先求分类概率，回归模式直接求回归值
        n=len(X)
        if mode=='c':
            classes=units[0].tree.classes
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    np.zeros((n,len(classes))),
                    index=X.index,columns=classes)
        elif mode=='r':
            #定义存放回归值的Series
            p_y=pd.Series(np.zeros(n),index=X.index)
            
        #逐个调用每个单元进行预测,并将结果累加
        units_p_y,units_paths=[],[]
        for i in range(len(units)):
            if show_time==True:
                print('\npredicting with unit %d ---'%i)
            if return_paths==True:
                p_y_,paths=units[i].predict(X,return_proba=True,return_paths=True,
                                           show_time=show_time,check_input=False)
                units_paths.append(paths)
            else:
                p_y_=units[i].predict(X,return_proba=True,return_paths=False,
                                     show_time=show_time,check_input=False)
            p_y+=units_weight[i]*p_y_
            if units_result==True:
                if (mode=='c')&(return_proba==False):
                    p_y_=units[i].choose_class_(p_y_,classes)
                units_p_y.append(p_y_)
                
        #分类概率归一化或回归值取平均
        if mode=='c':
            p_y=(p_y.T/p_y.sum(axis=1)).T
        elif mode=='r':
            #注：sklearn中此处取了中值而不是平均
            p_y=p_y/sum(units_weight)
            
        #返回分类概率或唯一分类
        if (mode=='c')&(return_proba==False):
            p_y=self.unit_test.choose_class_(p_y,classes)
            
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for predict: %f'%(end-start))
            
        if units_result==True:
            if return_paths==True:
                return p_y,units_p_y,paths
            else:
                return p_y,units_p_y
        else:
            if return_paths==True:
                return p_y,paths
            else:
                return p_y
    
    #评估
    def assess(self,y,p_y,mode=None):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:观测值，Series类型
        p_y:预测值，Series类型
        mode:模式，str类型，默认使用内部集成单元的属性，
             'c'->分类，'r'->回归
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        #校验参数
        if type(mode)==type(None):
            mode=self.units[0].tree.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        y,continuity_y=self.unit_test.check_input_y_(y,name='y')
        p_y,continuity_p_y=self.unit_test.check_input_y_(p_y,name='p_y')
        check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='c':
            return stats.accuracy(y,p_y)
        elif mode=='r':
            return stats.r_sqr(y,p_y)

#梯度提升
class GradientBoosting:
    '''\n  
    Note: 梯度提升(决策树)，简称GBDT，支持分类和回归
     
    Parameters
    ----------
    mode: 模式，分类->'c'，回归->'r'
    units_type: 集成单元模型类型，目前只有'cart_r'
    iter_max: 迭代优化次数，int类型(>=1)，默认10
    depth_max: 最大深度，int类型，0表示采用默认值(分类->1,回归->3)，默认0
    learning_rate: 学习率，float类型(>0.0,<=1.0),默认1.0
    ----------
    
    Attributes
    ----------
    units: 集成单元列表，分类->list(list(object))类型，回归->list(object)类型
    features: 特征列表，list(str)类型
    classes: 分类标签列表，list(str)类型，分类模式下有效
    ----------
    '''
    def __init__(self,mode='c',units_type='cart',iter_max=10,depth_max=0,
                 learning_rate=1.0):
        
        #校验参数类型和取值
        check_type('mode',type(mode),type(''))
        type_list=['r','c']
        mode=mode.lower()
        check_limit('mode',mode in type_list,str(type_list))
        
        check_type('units_type',type(units_type),type(''))
        type_list=['cart']
        units_type=units_type.lower()
        check_limit('units_type',units_type in type_list,str(type_list))
        
        check_type('iter_max',type(iter_max),type(0))
        check_limit('iter_max',iter_max>=1,'value>=1')
        
        check_type('learning_rate',type(learning_rate),type(0.0))
        check_limit('learning_rate',learning_rate>0.0,'value>0.0')
        check_limit('learning_rate',learning_rate<=1.0,'value<=1.0')

        if type(depth_max)==type(0):
            if depth_max==0:
                if mode=='r':
                    depth_max=3
                elif mode=='c':
                    depth_max=3
        
        #保存参数
        #注：此处depth_max参考了sklearn,尝试过回归也用depth_max=1，效果很糟糕
        self.unit_test=dt.DecisionTree(mode='r',model_type='cart',
                                       depth_max=depth_max)
        self.mode=mode
        self.units_type='cart'
        self.units_mode='r'
        self.iter_max=iter_max
        self.depth_max=depth_max
        self.learning_rate=learning_rate
    
    #代价函数
    def cost_(self,y,p_y,mode):
        #多分类使用对数似然损失
        #注：此处的y需要是one-hot编码，p_y需要是分类概率预测
        if mode=='c':
            return (-y*np.log(p_y)).sum()
        #回归使用平方损失
        elif mode=='r':
            re=y-p_y
            return np.dot(re.T,re)/2/len(y)
    
    #求梯度 
    #注：逻辑回归中的梯度下降针对的是参数，梯度提升中则是针对预测值
    def gradient_(self,y,p_y,mode):
        #分类模式下梯度等同于观测分类概率与预测分类概率的偏差
        #注：此处的y需要是one-hot编码，p_y需要是分类概率预测
        if mode=='c':
            return y-p_y
        #回归模式下梯度等同于观测值与预测值的偏差
        elif mode=='r':
            return y-p_y
        
    #拟合
    def fit(self,X,y,show_time=False):
        '''\n
        Function: 使用输入数据拟合梯度提升(决策树)
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签
              
        Description: 对迭代轮数t=1,2,...T有：
            (a)对样本i=1,2，...m，计算代价函数的负梯度
            (b)利用负梯度作为目标值, 拟合一颗弱学习器
            (c)为弱学习器拟合一个权重使当前代价最小，更新强学习器

        Parameters
        ----------
        X: 特征列，DataFrame类型
        y: 目标列，Series类型
        show_time: 是否显示耗时，bool类型，默认值False
        ----------  
        '''
        start = time.clock()
        check_type('show_time',type(show_time),type(True))
        #校验X,y输入
        X,self.continuity_X,self.mapping_X,X0=\
            self.unit_test.check_input_X_(X,to_index=True,return_source=True)
        y,self.continuity_y,self.mapping_y,y0=\
            self.unit_test.check_input_y_(y,to_index=True,return_source=True)
        #校验X,y输入是否匹配
        check_index_match(X,y,'X','y')
        feature_use_n=len(X.columns)
        #特征标签
        self.features=X.columns.tolist()
        #初始化强学习器的预测值向量
        n=len(y)
        self.units_p_y=[]
        self.r_h=[]
        if self.mode=='c':
            self.classes=y0.drop_duplicates().sort_values().astype('str').tolist()
            y=dp.dummy_var(y)
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    stats.softmax(np.zeros((n,len(self.classes)))),
                    index=y.index,columns=y.columns)
        elif self.mode=='r':
            #定义存放回归值的Series
            p_y=pd.Series(np.zeros(n),index=X.index)
        #迭代训练弱学习器
        self.units=[]
        for i in range(self.iter_max):
            if show_time==True:
                print('\nfitting with unit %d ---'%i)
            #针对预测值向量计算负梯度作为下一轮的拟合目标
            r=self.learning_rate*self.gradient_(y,p_y,self.mode)
            self.r_h.append(r)
            #提前结束拟合(暂未设置阈值，所以是0)
            if (r**2).values.sum()<=0:
                print('\nwarning: early stopping')
                break
            if self.mode=='r':
                #构建并拟合模型
                unit=dt.DecisionTree(mode=self.units_mode,model_type=self.units_type,
                                     depth_max=self.depth_max)
                unit.continuity_X,unit.mapping_X=self.continuity_X,self.mapping_X
                unit.continuity_y,unit.mapping_y=self.continuity_y,self.mapping_y
                unit.features_use_n=feature_use_n
                unit.fit(X,r,show_time=show_time,check_input=False)
                #计算弱学习器的预测值
                p_y_=unit.predict(X,return_proba=True)
                #为弱学习器计算一个乘数，使代价最小（一维优化问题）
                #注：也可以尝试对每个叶节点区域拟合乘数，精度更高
                #    即使不拟合该乘数，gbdt也能正常运作
                try:
                    gamma=(r*p_y_).sum()/(p_y_**2).sum()
                    for node in unit.tree.nodes:
                        if node.is_leaf==True:
                            node.output*=gamma
                except ZeroDivisionError:
                    gamma=1
                p_y+=gamma*p_y_
                #添加进强学习器
                unit.continuity_X,unit.mapping_X=None,None
                unit.continuity_y,unit.mapping_y=None,None
                self.units.append(unit)
                self.units_p_y.append(p_y_)
            elif self.mode=='c':
                sub_units,sub_units_p_y=[],[]
                #对每个类别的预测概率按负梯度方向的目标概率值变化量拟合弱学习器
                for j in range(len(self.classes)):
                    if show_time==True:
                        print('\n|| sub-unit for class %s'%str(self.classes[j]))
                    #构建并拟合模型
                    unit=dt.DecisionTree(mode=self.units_mode,model_type=self.units_type,
                                         depth_max=self.depth_max)
                    unit.continuity_X,unit.mapping_X=self.continuity_X,self.mapping_X
                    unit.continuity_y,unit.mapping_y=self.continuity_y,self.mapping_y
                    unit.features_use_n=feature_use_n
                    r_=r.iloc[:,j]
                    unit.fit(X,r_,show_time=show_time,check_input=False)
                    #计算弱学习器的预测值
                    p_y_=unit.predict(X,return_proba=True)
                    sub_units_p_y.append(p_y_)
                    #为弱学习器计算一个权重，使代价最小（一维优化问题）
                    try:
                        gamma=(r_*p_y_).sum()/(p_y_**2).sum()
                        for node in unit.tree.nodes:
                            if node.is_leaf==True:
                                node.output*=gamma
                    except ZeroDivisionError:
                        gamma=1
                    p_y.iloc[:,j]+=gamma*p_y_
                    #添加进强学习器当前层集合
                    unit.continuity_X,unit.mapping_X=None,None
                    unit.continuity_y,unit.mapping_y=None,None
                    sub_units.append(unit)
                #添加进强学习器
                self.units.append(sub_units)
                self.units_p_y.append(sub_units_p_y)
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for trainning: %f'%(end-start))  
            
    #预测        
    def predict(self,X,units=None,mode=None,classes=None,units_result=False,
                return_proba=False,return_paths=False,show_time=False):
        '''\n
        Function: 使用输入数据和所有集成单元进行预测，没有输入集成单元时使用内部缓存
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        units: 集成单元，list(DecitionTree)类型，默认调用内部缓存
        mode: 模式，分类->'c'，回归->'r'，默认'c'
        classes: 分类标签列表，list(str)类型
        units_result: 是否返回每个单元的分类结果，bool类型，默认False
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False，
                      分类概率不能直接用于评估
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率，Series/DataFrame类型
        1: 各个单元的预测的分类/分类概率，list(Series)/list(DataFrame)类型
        2: 所有数据最终抵达的节点和决策路径，list(DataFrame)类型
        -------
        '''
        start = time.clock()        
        #校验参数
        if type(units)==type(None):
            units=self.units
            
        if type(mode)==type(None):
            mode=self.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        
        if (type(classes)==type(None))&(mode=='c'):
            classes=self.classes
            
        check_type('units',type(units),type([]))
        if len(units)==0:
            raise ValueError('lack of units')
        if mode=='r':
            check_type('element in units',type(units[0]),type(dt.DecisionTree()))
        elif mode=='c':
            check_type('element in units',type(units[0][0]),type(dt.DecisionTree()))
        
        check_type('return_proba',type(return_proba),type(True))
        check_type('return_paths',type(return_paths),type(True))
        check_type('show_time',type(show_time),type(True))
        
        X,continuity_X=self.unit_test.check_input_X_(X)
        features=[]
        if mode=='c':
            for units_ in units:
                for unit in units_:
                    features+=unit.tree.features
        elif mode=='r':
            for unit in units:
                features+=unit.tree.features
        features=list(set(features))
        check_items_match(X.columns,features,'X','unit','features',mode='right')
        #分类模式先求分类概率，回归模式直接求回归值
        n=len(X)
        if mode=='c':
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    np.zeros((n,len(classes))),
                    index=X.index,columns=classes)
        elif mode=='r':
            #定义存放回归值的Series
            p_y=pd.Series(np.zeros(n),index=X.index)
        #逐个调用每个单元进行预测,并将结果累加
        units_p_y,units_paths=[],[]
        for i in range(len(units)):
            if show_time==True:
                print('\npredicting with unit %d ---'%i)
            if mode=='r':
                if return_paths==True:
                    p_y_,paths=units[i].predict(X,return_proba=True,return_paths=True,
                                               show_time=show_time,check_input=False)
                    units_paths.append(paths)
                else:
                    p_y_=units[i].predict(X,return_proba=True,return_paths=False,
                                         show_time=show_time,check_input=False)
                p_y+=p_y_
                if units_result==True:
                    if (mode=='c')&(return_proba==False):
                        p_y_=units[i].choose_class_(p_y_,classes)
                    units_p_y.append(p_y_)
            #分类模式需要调用子单元对每个类的概率进行预测
            elif mode=='c':
                classes_p_y,classes_paths=[],[]
                for j in range(len(classes)):
                    if return_paths==True:
                        p_y_,paths=units[i][j].predict(X,return_proba=True,return_paths=True,
                                                       show_time=show_time,check_input=False)
                        classes_paths.append(paths)
                    else:
                        p_y_=units[i][j].predict(X,return_proba=True,return_paths=False,
                                                 show_time=show_time,check_input=False)
                    p_y.iloc[:,j]+=p_y_
                    if units_result==True:
                        if (mode=='c')&(return_proba==False):
                            p_y_=units[i].choose_class_(p_y_,classes)
                        classes_p_y.append(p_y_)
                if return_paths==True:
                    units_paths.append(classes_paths)
                if units_result==True:
                    units_p_y.append(classes_p_y)
        #返回分类概率或唯一分类
        if (mode=='c')&(return_proba==False):
            p_y=self.unit_test.choose_class_(p_y,classes)
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for predict: %f'%(end-start))
        if units_result==True:
            if return_paths==True:
                return p_y,units_p_y,paths
            else:
                return p_y,units_p_y
        else:
            if return_paths==True:
                return p_y,paths
            else:
                return p_y
    
    #评估
    def assess(self,y,p_y,mode=None):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:观测值，Series类型
        p_y:预测值，Series类型
        mode:模式，str类型，默认使用内部集成单元的属性，
             'c'->分类，'r'->回归
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        #校验参数
        if type(mode)==type(None):
            mode=self.mode
        check_type('mode',type(mode),type(''))
        mode_list=['c','r']
        check_limit('mode',mode in mode_list,str(mode_list))
        check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='c':
            return stats.accuracy(y.astype('str'),p_y.astype('str'))
        elif mode=='r':
            r_sqr=stats.r_sqr(y,p_y)
            if r_sqr<0:
                print('warning: R2 is less than 0, which means bad fitting,'+
                      '\ntry to reduce the learning rate')
            return r_sqr
        