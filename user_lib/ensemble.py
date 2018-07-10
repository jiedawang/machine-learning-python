# -*- coding: utf-8 -*-
import time
import user_lib.deci_tree as dt
import pandas as pd
import numpy as np
import user_lib.statistics as stats
from multiprocessing import Pool,cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from user_lib.check import check_type,check_limit,check_index_match,check_feats_match

#随机森林
#基于Bagging集成学习原理，基本单元一般为决策树（可以替换为其他模型）
class RandomForest:
    '''\n  
    Note: 以展示各个算法的实现为目的
     
    Parameters
    ----------
    units_n: 集成单元的数量，int类型(>=1)，默认10
    units_type: 集成单元的类型，str类型(id3,c4.5,cart_c,cart_r)，
                'id3'->分类，离散特征+离散输出，
                'c4.5','cart_c'->分类，离散或连续特征+离散输出，
                'cart_r'->回归，离散或连续特征+连续输出，
                默认值'cart_c'
    depth_max: 最大深度，int类型(>=1)，None表示无限制，默认值10
    split_sample_n: 分裂所需最少样本数，int类型(>=2)，默认值2
    leaf_sample_n: 叶节点所需最少样本数，int类型(>=1)，默认值1
    features_use: 每次使用的特征数量，str/float/int类型
                 'all'->全量，
                 'sqrt'->总数量的平方根，
                 'log2'->总数量的2的对数，
                 float->总数量的相应比例，区间(0.0,1.0)，
                 int->固定数量，区间[1,feature_num]，
                 默认值'sqrt'
    features_reuse: 是否允许一个特征重复使用，bool类型，默认值False
    ----------
    '''
    
    def __init__(self,units_n=10,units_type='cart_c',
                 depth_max=None,split_sample_n=2,leaf_sample_n=1,
                 features_use='sqrt',features_reuse=False):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('units_n',type(units_n),type(0))
        check_type('units_type',type(units_type),type(''))
        check_limit('units_n',units_n>=1,'value>=1')
        type_list=['id3','c4.5','cart_c','cart_r']
        units_type=units_type.lower()
        check_limit('units_type',units_type in type_list,str(type_list))
        #保存参数
        self.units_n=units_n
        self.units_type=units_type
        self.unit_test=dt.DecisionTree(model_type=units_type,depth_max=depth_max,
                                       split_sample_n=split_sample_n,leaf_sample_n=leaf_sample_n,
                                       features_use=features_use,features_reuse=features_reuse)
        self.depth_max=depth_max
        self.split_sample_n=split_sample_n
        self.leaf_sample_n=leaf_sample_n
        self.features_reuse=features_reuse
        self.features_use=features_use
        if units_type=='cart_r':
            self.mode='Regressor'
        else:
            self.mode='Classifier'

    #拟合
    def fit(self,X,y,show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合随机森林
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签

        Parameters
        ----------
        X: 所有的特征列，DataFrame类型
        y: 分类列，Series类型
        show_time: 是否显示耗时，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        '''
        start = time.clock()
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
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
        self.units,self.units_oob_score=[],[]
        self.features=X.columns.tolist()
        #out of bag袋外数据预测矩阵初始化
        if self.mode=='Classifier':
            self.classes=y0.drop_duplicates().sort_values().astype('str').tolist()
            oob_predict=pd.DataFrame(np.zeros((len(X.index),len(self.classes))),
                                     index=X.index,columns=self.classes)
        elif self.mode=='Regressor':
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
            unit=dt.DecisionTree(model_type=self.units_type,depth_max=self.depth_max,
                                 split_sample_n=self.split_sample_n,leaf_sample_n=self.leaf_sample_n,
                                 features_use=self.features_use,features_reuse=self.features_reuse)
            unit.continuity_X,unit.mapping_X=self.continuity_X,self.mapping_X
            unit.continuity_y,unit.mapping_y=self.continuity_y,self.mapping_y
            unit.features_use_n=self.features_use_n
            unit.fit(X_,y_,show_time=show_time,check_input=False)
            #obb预测(整体汇总+单元结果)
            #注：目前回归模式下obb error算的是1-r2，不确定这种算法是否正确
            if self.mode=='Classifier':
                p_y_=unit.predict(oob_X0_,return_proba=True,check_input=False)
                p_y_0=unit.choose_class_(p_y_,self.classes)
                score_=unit.assess(oob_y0_,p_y_0,check_input=False)
                oob_predict.loc[p_y_.index,:]+=p_y_
            elif self.mode=='Regressor':
                p_y_=unit.predict(oob_X0_,check_input=False)
                score_=unit.assess(oob_y0_,p_y_,check_input=False)
                oob_predict.loc[p_y_.index]+=p_y_
            oob_trees_n.loc[p_y_.index]+=1
            self.units_oob_score.append(score_)
            #添加进随机森林
            unit.continuity_X,unit.mapping_X=None,None
            unit.continuity_y,unit.mapping_y=None,None
            self.units.append(unit)
        #oob整体预测结果计算
        #注：由于存在少量数据不满足oob条件所以没有预测结果，需要筛去
        boolIdx=(oob_trees_n!=0.0)
        if self.mode=='Classifier':
            oob_predict=self.unit_test.choose_class_(oob_predict[boolIdx],self.classes)
        elif self.mode=='Regressor':
            oob_predict=oob_predict[boolIdx]/oob_trees_n[boolIdx]
        score=self.unit_test.assess(y0[boolIdx],oob_predict,mode=self.mode,check_input=False)
        self.oob_score=score
        end = time.clock()
        if show_time==True:
            print('\ntotal time used for trainning: %f'%(end-start))
    
    #预测        
    def predict(self,X,units=None,units_result=False,return_proba=False,
                return_paths=False,show_time=False,check_input=True):
        '''\n
        Function: 使用输入数据和所有集成单元进行预测，没有输入集成单元时使用内部缓存的树
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        units: 集成单元，list(DecitionTree)类型，默认调用内部缓存
        units_result: 是否返回每个单元的分类结果，bool类型，默认False
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False，
                      分类概率不能直接用于评估
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率，Series/DataFrame类型
        1: 各个单元的预测的分类/分类概率，list(Series)/list(DataFrame)类型
        2: 所有数据最终抵达的节点和决策路径，list(DataFrame)类型
        -------
        '''
        start = time.clock()
        if type(units)==type(None):
            units=self.units
        #校验参数
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
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
            check_feats_match(X.columns,features,'X','unit',mode='right')
        n=len(X)
        mode=units[0].tree.mode
        classes=units[0].tree.classes
        #分类模式先求分类概率，回归模式直接求回归值
        if mode=='Classifier':
            #定义存放分类结果的DataFrame
            p_y=pd.DataFrame(
                    np.zeros((n,len(classes))),
                    index=X.index,columns=classes)
        elif mode=='Regressor':
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
            if units_result==True:
                units_p_y.append(p_y_)
            p_y+=p_y_
        #分类模式下归一化，回归模式下求平均值，公式一致
        p_y=p_y/len(units)
        #分类模式下可以返回分类概率或唯一分类
        if (mode=='Classifier')&(return_proba==False):
            p_y=self.unit_test.choose_class_(p_y,classes)
            if units_result==True:
                for i in range(len(units_p_y)):
                    units_p_y[i]=self.unit_test.choose_class_(units_p_y[i],classes)
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
    def assess(self,y,p_y,mode=None,check_input=True):
        '''\n
        Function: 使用输入的观测值和预测值进行模型评估
        
        Notes: 注意数据集的数据类型，分类首选类型str，回归首选类型float64，
               拟合时数据集采用非首选类型可能会导致此处类型不匹配，建议提前转换
        
        Parameters
        ----------
        y:实际的分类，Series类型
        p_y:预测的分类，Series类型
        mode:模式，str类型，默认使用内部缓存树的属性，
             'Classifier'->分类，'Regressor'->回归
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        if type(mode)==type(None):
            mode=self.units[0].tree.mode
        #校验输入
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            mode_list=['Classifier','Regressor']
            check_limit('mode',mode in mode_list,str(mode_list))
            y,continuity_y=self.unit_test.check_input_y_(y,name='y')
            p_y,continuity_p_y=self.unit_test.check_input_y_(p_y,name='p_y')
            check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='Classifier':
            return stats.accuracy(y,p_y)
        elif mode=='Regressor':
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
            p_y_=self.predict(test_X,units=units_,check_input=False)
            score_=self.assess(test_y,p_y_,check_input=False)
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
            p_y_=self.predict(test_X,units=units_,check_input=False)
            score_=self.assess(test_y,p_y_,check_input=False)
            if score_>best_score:
                best_score=score_
                best_subset=units_
        return best_subset
    
    #模型选择
    def selection(self,test_X,test_y,units=None,units_oob_score=None,
                  mode='oob',return_units=False,show_time=False,check_input=True):
        '''\n
        Function: 在生成好的模型上进行选择，筛选出集成单元的一个子集
        
        Notes: 作用类似于决策树的剪枝，通过一些规则生成可选子集，
               再通过在测试集上的表现选择最优的一个，能够得到更简单且泛化能力更强的模型
        
        Parameters
        ----------
        test_X: 测试集特征矩阵，DataFrame类型
        test_y: 测试集观测向量，Series类型
        units: 集成单元，list(DecitionTree)类型
        units_oob_score: 集成单元obb评分，list(float)类型
        mode:模式，str类型，默认'oob'
             'rd'->随机选择，'oob'->oob选择
        return_units: 是否以返回值形式给到选择后的集成单元，bool类型，默认False
        check_input: 是否进行输入校验，bool类型，默认值True
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
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            check_type('units',type(units),type([]))
            check_type('element in units',type(units[0]),type(dt.DecisionTree()))
            check_type('units_oob_score',type(units_oob_score),type([]))
            check_type('element in units_oob_score',type(units_oob_score[0]),[type(0.0),np.float64])
            check_type('mode',type(mode),type(''))
            check_type('return_units',type(return_units),type(True))
            mode_list=['rd','oob']
            check_limit('mode',mode in mode_list,str(mode_list))
            test_X,continuity_X=self.unit_test.check_input_X_(test_X,'test_X')
            test_y,continuity_y=self.unit_test.check_input_y_(test_y,'test_y')
            check_index_match(test_X,test_y,'test_X','test_y')
            features=[]
            for unit in units:
                features+=unit.tree.features
            features=list(set(features))
            check_feats_match(test_X.columns,features,'test_X','tree',mode='right')
        #选择
        if mode=='rd':
            subset=self.random_selection_(test_X,test_y,units)
        elif mode=='oob':
            subset=self.oob_selection_(test_X,test_y,units,units_oob_score)
        end = time.clock()
        if show_time==True:
            print('\ntime used for selection:%f'%(end-start))
        if return_units==False:
            self.units=subset
        else:
            return subset
