# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

#使用numba加速运算，第一次运行时需要一些时间编译，
#且只能接收Numpy数组，对于pandas的数据对象可通过values属性获取

#信息熵etp=sum(p*log2(p))
#p=每个取值value的占比
'''
info:需要求熵的数据列,Series类型
continuous:连续性
value:分裂点,只有在处理连续数据时有意义
return> 0:熵
'''
@nb.jit(nopython=True)
def entropy(info,continuous=False,value=0):
    #数据集大小,初始化熵
    n,etp=len(info),0.0
    #是否是连续数据,连续数据按指定阈值分裂成两部分,离散数据按每个取值分裂
    if continuous==True:
        #统计左半部分大小，由于只有两个子集，另一半不需要另外统计
        count=0       
        for i in range(n):
            if info[i]<=value:
                count+=1
        p=count/n
        etp=-p*np.log2(p)-(1-p)*np.log2(1-p)
    else:
        #先排序和初始化变量
        info_=np.sort(info)
        value,count,etp=info_[0],0,0.0
        for i in range(n):
            #是正在统计的值，计数+1；否则完成剩余计算并开始统计新的取值
            if info_[i]==value:
                count+=1
            else:
                p=count/n
                etp-=p*np.log2(p)
                value,count=info_[i],1
        #最后一个取值的计算收尾
        p=count/n
        etp-=p*np.log2(p)
    return etp

#条件熵con_etp=sum(p*etp)
#p,etp=x的各个取值的数量占比以及按x的值分裂后每个子集y的熵
'''
x:用于分裂的特征列,Series类型
y:分类列,Series类型
continuous:连续性
value:分裂点,只有在处理连续数据时有意义
return> 0:条件熵
'''
@nb.jit(nopython=True)
def con_entropy(x,y,continuous=False,value=0):
    n=len(x)
    #连续特征和离散特征采用不同计算方式
    if continuous==True:
        boolIdx=(x<=value)
        p=len(x[boolIdx])/n
        con_ent=p*entropy(y[boolIdx])\
            +(1-p)*entropy(y[~boolIdx])
    else:
        #x取值列表
        values=np.zeros(len(x))
        #x取值列表长度，条件熵
        length,con_ent=0,0.0
        #遍历x每个值
        for i in range(len(x)):
            #判断是否是没处理过的取值
            find_flag=False
            for j in range(length):
                if x[i]==values[j]:
                    find_flag=True
                    break
            #新取值计算
            if find_flag==False:
                values[length]=x[i]
                length+=1
                y_=y[x==x[i]]
                p=len(y_)/n
                con_ent+=p*entropy(y_)
    return con_ent

#最优分裂点选择
'''
x:用于分裂的特征列,Series类型
y:分类列,Series类型
return> 0:最优分裂点 1:所有可能分裂点的数量
'''
@nb.jit(nopython=True)
def choose_split_value(x,y):
    #计算分裂前的信息熵
    baseEntropy=entropy(y)
    #需要尝试的分裂点
    values,n=filter_split_values(x,y)
    #初始化变量
    bestInfGain=0.0
    bestSplitValue=values[0]
    #逐个计算所有可能分裂点的条件熵
    for j in range(len(values)-1):
        split_value=values[j]
        infGain=baseEntropy-con_entropy(x,y,True,split_value)
        if infGain>bestInfGain:
            bestInfGain=infGain
            bestSplitValue=split_value
    return bestSplitValue,n

#筛选分裂点
'''
x:用于分裂的特征列,Series类型
y:分类列,Series类型
return> 0:经过筛选的分裂点集 1:所有可能分裂点的数量
'''
@nb.jit(nopython=True)
def filter_split_values(x,y):
    n=len(x)
    #将x,y按x升序排序
    sortIdx=np.argsort(x)
    x,y=x[sortIdx],y[sortIdx]
    #需要选取的点的布尔索引(因为不能直接初始化bool类型所以用int代替)
    filterIdx=np.zeros(n)
    filterIdx[0],filterIdx[n-1]=1,1
    #将分类结果y有变化的位置选取出来
    for i in range(n-1):
        if y[i]!=y[i+1]:
            filterIdx[i],filterIdx[i+1]=1,1
    return x[filterIdx==1].copy(),n

#方法1：树以dict格式存储，通过递归生成树
class DecisionTree:
    
    #信息熵,可以用于求类别的熵，也可以用于求特征的熵,只能计算单列
    #表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
    #0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
    '''
    def entropy(self,info,continuous=False,value=0):
        n=len(info)
        if continuous==True:
            #计算值的概率分布
            p=len(info[info<=value])/n
            #计算信息熵
            etp=-p*np.log2(p)-(1-p)*np.log2(1-p)
        else:
            #计算值的概率分布
            values_count=info.groupby(info).count()
            p=values_count/n
            #计算信息熵
            etp=-np.sum(p*np.log2(p))
        return etp
    '''
    
    #条件熵
    #在x中第i个随机变量确定的情况下，随机变量y的不确定性
    #即按第i个特征划分数据后的信息熵
    '''
    def con_entropy(self,x,y,continuous=False,value=0):
        n=len(x)
        #计算条件熵
        con_ent=0.0
        if continuous==True:
            boolIdx=(x<=value)
            p=len(x[boolIdx])/n
            con_ent+=p*self.entropy(y[boolIdx])
            con_ent+=(1-p)*self.entropy(y[~boolIdx])
        else:
            values=x.drop_duplicates().tolist()
            for i in range(len(values)):
                boolIdx=(x==values[i])
                p=len(x[boolIdx])/n
                con_ent+=p*self.entropy(y[boolIdx])
        return con_ent
        
    (另一种写法,使用pandas进行多级聚合统计)
    def con_entropy(self,x,y,continuous=False,value=0):
        #如果x是连续值，将x转化为关于分裂点的布尔索引
        if continuous==True:
            x=(x<=value)
        #根据划分特征和分类统计数量
        values_count=y.groupby([x,y]).size()
        #单独提出划分特征的取值计数
        split_values_count=values_count.sum(level=x.name)
        #每个特征取值中各个分类出现的概率
        p_y=values_count/split_values_count
        #统计每个特征取值下子集的熵
        etp=(-p_y*np.log2(p_y)).sum(level=x.name)
        #统计不同特征取值出现的概率
        p_x=split_values_count/split_values_count.sum()
        #计算条件熵
        con_ent=np.dot(p_x.T,etp)
        return con_ent
    '''
    
    #筛选分裂点，取分类结果有变化的点
    '''
    def filterSplitValues(self,x,y):
        #重整源数据
        data=pd.DataFrame()
        data['x'],data['y']=x,y
        data=data.sort_values('x')
        data.index=np.linspace(0,len(data)-1,len(data)).astype(np.int64)
        #提取首末行
        head,foot=data.iloc[0:1,:],data.iloc[len(data)-1:,:]
        #复制一份作为后续数
        subsequent=data.copy()
        subsequent.index=subsequent.index-1
        subsequent.columns=subsequent.columns+'_next'  
        #每个数与自己的后续数匹配并找出变化位置
        data=data.join(subsequent,how='inner')   
        change_points=data[data['y']!=data['y_next']]  
        #提取变化位置前后的点 
        change_f=change_points.iloc[:,0:2]
        change_b=change_points.iloc[:,2:]
        change_b.columns=change_f.columns
        #将所有的检查点整合并去重排序
        check_points=pd.concat([head,change_f,change_b,foot])
        check_points=check_points.drop_duplicates().sort_index().iloc[:,0].tolist()
        return check_points
    '''
       
    #最优特征选择(ID3)
    #选择依据：信息增益
    #=划分前类别的信息熵-划分后类别的条件熵
    #用于衡量经过某特征的划分后分类的不确定性降低了多少
    '''
    X:所有参与选择的特征列,DataFrame类型
    y:分类列,Series类型
    return> 0:最优分裂特征的索引
    '''
    def choose_feature_by_id3(self,X,y):
        #计算分割前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGain=0.0
        bestFeatureIdx=-1
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(X.columns)):
            infGain=baseEntropy-con_entropy(X.iloc[:,i].values,y.values)
            if infGain>bestInfGain:
                bestInfGain=infGain
                bestFeatureIdx=i
        return bestFeatureIdx
    
    #最优特征选择(C4.5)
    #选择依据：信息增益比
    #=信息增益/划分特征的信息熵
    #避免因为特征取值多而导致信息增益偏大
    #C4.5增加了对连续数据的处理，连续特征根据信息增益选择最佳分裂点转换为离散值
    '''
    X:所有参与选择的特征列,DataFrame类型
    y:分类列,Series类型
    continuous:连续性,list类型
    return> 0:最优分裂特征的索引
    '''
    def choose_feature_by_c45(self,X,y,continuous):
        #计算分割前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGainRatio=0.0
        bestFeatureIdx=-1
        bestSplitValue=0.0
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in range(len(X.columns)):
            x=X.iloc[:,i]
            #是否为连续特征
            if continuous[i]==True:
                splitValue,n=choose_split_value(x.values,y.values)
                splitFeatEntropy=entropy(x.values,True,splitValue)
                infGain=baseEntropy\
                    -con_entropy(x.values,y.values,True,splitValue)
                    #-np.log2(n-1)/len(x)
            else:
                splitValue=0.0
                splitFeatEntropy=entropy(x.values)
                infGain=baseEntropy-con_entropy(x.values,y.values)
            if splitFeatEntropy==0:
                continue
            infGainRatio=infGain/splitFeatEntropy
            if infGainRatio>bestInfGainRatio:
                bestInfGainRatio=infGainRatio
                bestFeatureIdx=i
                bestSplitValue=splitValue
        return bestFeatureIdx,bestSplitValue
    
    #计算每个类的概率
    '''
    y:分类列,Series类型
    return> 0:分类概率,dict类型
    '''
    def compute_proba(self,y):
        proba={'proba_':{}}
        values_count=y.groupby(y).count()
        total_count=values_count.sum()
        for i in range(len(values_count)):
            p=values_count.iloc[i]/total_count
            value=self.get_ylabel(self.mapping_y,values_count.index[i])
            proba['proba_'][value]=p
        return proba   
          
    #选择概率最高的类作为叶节点判定的类,用于预测
    '''
    p_y_:预测的分类概率,DataFrame类型
    fill_empty:是否填充未被成功分类的数据
    return> 0:选择的类,Series类型
    '''
    def choose_class(self,p_y_,fill_empty):
        classify=self.classify
        p_max=p_y_.max(axis=1)
        p_y=pd.Series(np.full(len(p_y_),''),index=p_y_.index)
        for i in range(len(classify)):
            p_y[p_y_.iloc[:,i]==p_max]=classify[i]
                #按类别分布情况加权随机填充未能分类的记录
        if fill_empty==True:
            nullIdx=(p_y=='')
            n=p_y[nullIdx].count()
            p_y.loc[nullIdx]=p_y[~nullIdx].sample(n=n,replace=True).tolist()
        return p_y
    
    #选择概率最高的类作为叶节点判定的类，用于观察树结构
    '''
    proba_dict:分类概率,dict类型
    return> 0:选择的类
    '''
    def choose_class_(self,proba_dict):
        label = list(proba_dict.keys())[0]
        proba_=proba_dict[label]
        class_,proba_max='',0.0
        for key in proba_.keys():
            if proba_[key]>proba_max:
                proba_max=proba_[key]
                class_=key
        return class_
    
    #校验输入数据类型并返回X的离散性默认判定
    '''
    X:所有的特征列,DataFrame类型
    y:分类列,Series类型
    model_type:模型算法类型,str类型,id3/c4.5/cart三种
    return> 0:连续性判定,list类型
    '''
    def check_input(self,X,y,model_type):
        type_list=('id3','c4.5','cart')
        if model_type.lower() not in type_list:
            print('model_type should in:')
            print(type_list)
            raise TypeError('Unknown type')
        if type(X)!=type(pd.DataFrame()):
            raise TypeError('X should be dataframe')
        if type(y)!=type(pd.Series()):
            raise TypeError('y should be series')
        #离散性默认设置
        continuous=[]
        for dtype in X.dtypes:
            if str(dtype) in ['object','category','bool']:
                continuous.append(False)
            else:
                continuous.append(True)
        #ID3不支持连续特征
        if model_type=='ID3':
            if True in continuous:
                raise TypeError('ID3 does not support continuous features')
        return continuous
    
    #将离散变量转化为数值型以支持numba运行
    '''
    X:所有的特征列,DataFrame类型
    continuous:连续性,list类型
    return> 0:转化后的X 1:映射关系,DataFrame类型
    '''
    def format_X(self,X,continuous):
        mapping_list=[]
        X_=X.copy()
        for i in range(len(continuous)):
            if continuous[i]==False:
                feature_label=X.columns[i]
                values=X.iloc[:,i].sort_values().drop_duplicates()
                mapping={label:idx for idx,label in enumerate(values.astype('str'))}
                X_.iloc[:,i]=X.iloc[:,i].astype('str').map(mapping)
                mapping_list+=[[feature_label,idx,label] for idx,label in enumerate(values)]
        return X_,pd.DataFrame(mapping_list,columns=['feature','valueId','label'])
    '''
    y:分类列,Series类型
    continuous:连续性,list类型
    return> 0:转化后的y 1:映射关系,DataFrame类型
    '''
    def format_y(self,y):
        y_=y.copy()
        values=y.sort_values().drop_duplicates()
        mapping={label:idx for idx,label in enumerate(values.astype('str'))}
        y_=y.astype('str').map(mapping)
        mapping_list=[[idx,label] for idx,label in enumerate(values)]
        return y_,pd.DataFrame(mapping_list,columns=['valueId','label'])
    
    #转换回原来的标签
    '''
    mapping_X: X中数值型标签和原标签的映射关系,DataFrame类型
    feature: 特征名,即列名
    valueId: 数值型标签,int类型
    return> 0:原标签
    '''
    def get_xlabel(self,mapping_X,feature,valueId):
        boolIdx=(mapping_X['feature']==feature)&(mapping_X['valueId']==valueId)
        return mapping_X['label'][boolIdx].values[0]
    '''
    mapping_y: y中数值型标签和原标签的映射关系,DataFrame类型
    valueId: 数值型标签,int类型
    return> 0:原标签
    '''
    def get_ylabel(self,mapping_y,valueId):
        boolIdx=(mapping_y['valueId']==valueId)
        return mapping_y['label'][boolIdx].values[0]
    
    #根据第i列特征分裂数据集
    '''
    X:当前所有的特征列,DataFrame类型
    y:分类列,Series类型
    continuous:连续性
    value:分裂点,只有在处理连续数据时有意义
    return> 0:X分裂后的集合,list(DataFrame) 
            1:y分裂后的集合,list(Series) 
            2:分裂值列表,list
    '''
    def split(self,X,y,i,continuous=False,value=0):
        #抽取第i列特征
        x=X.iloc[:,i]
        feature_label=X.columns[i]
        #连续特征和离散特征采用不同的处理方式
        if continuous==True:
            #根据分裂点将数据集拆分
            values=['<=%s'%str(value),'>%s'%str(value)]
            boolIdx=(x<=value)
            result_X=[X[boolIdx],X[~boolIdx]]
            result_y=[y[boolIdx],y[~boolIdx]]
        else:
            #去重得到特征值列表
            values=x.sort_values().drop_duplicates().tolist()
            #根据不同的特征值进行分割
            result_X,result_y=[],[]
            for j in range(len(values)):
                result_X.append(X[x==values[j]])
                result_y.append(y[x==values[j]])
                values[j]=self.get_xlabel(self.mapping_X,feature_label,values[j])
        return result_X,result_y,values
    
    #创建树，结果以字典形式返回
    '''
    X:所有的特征列,DataFrame类型
    y:分类列,Series类型
    continuous:连续性,list类型
    model_type:模型算法类型,str类型,id3/c4.5/cart三种,
    depth_max:最大深度
    return> 0:决策树,dict类型
    '''
    def fit(self,X,y,continuous=[],model_type='c4.5',depth_max=10,output=False):
        start = time.clock()
        if continuous==[]:
            continuous=self.check_input(X,y,model_type)
        self.features=X.columns.tolist()
        X,self.mapping_X=self.format_X(X,continuous)
        self.classify=y.drop_duplicates().tolist()
        y,self.mapping_y=self.format_y(y)
        self.deciTree=self.build(X,y,continuous,model_type,depth_max)
        end = time.clock()
        print('\ntime used for trainning:%f'%(end-start))
        if output==True:
            return self.deciTree
    
    #构建树
    '''
    X:当前所有的特征列,DataFrame类型
    y:分类列,Series类型
    continuous:连续性,list类型
    model_type:模型算法类型,str类型,id3/c4.5/cart三种,
    depth_max:最大深度
    depth:当前深度,不需要自主赋值
    return> 0:决策树分支,dict类型 / 类名
    '''
    def build(self,X,y,continuous,model_type,depth_max,depth=0):
        print('Current dataset size: %d'%len(y))
        #数据集只有一个类时返回这个类名
        if len(y.groupby(y).count())==1:
            print('<LeafNode> only one class')
            return self.compute_proba(y)
        #可用特征不足，返回出现频数最高的类名
        if len(X.columns)==0:
            print('<LeafNode> lack of feature')
            return self.compute_proba(y)
        #超出高度上限，返回出现频数最高的类名
        if depth>depth_max:
            print('<LeafNode> reach maximum depth')
            return self.compute_proba(y)
        #选择最优特征进行分割，并以字典形式记录结果
        #格式：{特征名：{特征值（中间结点）：{...},特征值（叶结点）：类名}}
        if model_type.lower()=='id3':
            bestFeatureIdx,bestSplitValue=self.choose_feature_by_id3(X,y),0.0
        elif model_type.lower()=='c4.5':
            bestFeatureIdx,bestSplitValue=self.choose_feature_by_c45(X,y,continuous)
        else:
            raise TypeError('Unknown type')
        #特征值统一，无法继续分割
        if bestFeatureIdx==-1:
            print('<LeafNode> only one feature value')
            return self.compute_proba(y)
        #获取最优划分特征的相关信息
        bestFeatureLabel=X.columns[bestFeatureIdx]
        #定义树
        deciTree={bestFeatureLabel:{}}
        #分割数据集
        splited_X,splited_y,split_values=self.split(X,y,bestFeatureIdx,
                                          continuous[bestFeatureIdx],
                                          bestSplitValue)
        continuous_=continuous.copy()
        continuous_.pop(bestFeatureIdx)
        #对各个结点进行递归生成剩余部分
        for i in range(len(split_values)):
            print('<Split> feature:%s value:%s'%(bestFeatureLabel,str(split_values[i])))
            deciTree[bestFeatureLabel][split_values[i]]=self.build(
                    splited_X[i].drop(bestFeatureLabel,axis=1),
                    splited_y[i],continuous_,
                    model_type,depth_max,depth+1)
        return deciTree
    
    #决策路径，只能一次处理一行数据，主要用于校对
    '''
    dr:数据行,Series类型
    tree:决策树,dict类型
    print> 分支选择和到达的叶节点
    '''
    def decition_path(self,dr,tree=None):
        if type(tree)==type(None):
            tree=self.deciTree
        #获取首个结点在dict中对应的key，即最优划分特征
        bestFeature=list(tree.keys())[0]
        #截取该节点下方的分支树
        childDict=tree[bestFeature]
        #遍历每个分支
        for key in childDict.keys():
            match_flag=False
            #连续特征和离散特征采用不同方式处理
            if type(key)==type(''):
                if key.find('<=')>=0:
                    #注：暂不支持时间类型的分割，所以直接转为float
                    splitValue=float(key.replace('<=',''))
                    if dr[bestFeature]<=splitValue:
                        match_flag=True
                elif key.find('>')>=0:
                    splitValue=float(key.replace('>',''))
                    if dr[bestFeature]>splitValue:
                        match_flag=True
                else:
                    if dr[bestFeature]==key:
                        match_flag=True
            else:
                if dr[bestFeature]==key:
                    match_flag=True
            #如果是中间结点就拆分数据并继续递归，如果是叶结点则返回类名
            if match_flag==True:
                if type(childDict[key])==type({}):
                    print('<Branch> feature:%s value:%s'%(bestFeature,str(key)))
                    self.decition_path(dr,childDict[key]) 
                else:
                    print('<LeafNode> class:%s'%str(childDict[key]))
        return
    
    #预测
    '''
    tree:决策树,dict类型
    X:所有特征列,DataFrame类型
    fill_empty:是否填充未能成功分类的数据
    return_proba:是否返回分类概率
    first:递归函数中判断是否是第一次调用的标志位,不需要自主赋值
    return> 0:预测的分类,Series类型
    '''
    def predict(self,X,tree=None,fill_empty=True,return_proba=False,first=True):
        if type(tree)==type(None):
            tree=self.deciTree
        start = time.clock()
        classify=self.classify
        #定义存放分类结果的series
        p_y_=pd.DataFrame(
                np.zeros(len(X)*len(classify)).reshape(len(X),len(classify)),
                index=X.index,columns=classify)
        #获取首个结点在dict中对应的key，即最优划分特征
        bestFeature=list(tree.keys())[0]
        #截取该节点下方的分支树
        childDict=tree[bestFeature]
        #遍历每个分支
        for key in childDict.keys():
            #如果是中间结点就拆分数据并继续递归，如果是叶结点则返回类名
            if bestFeature!='proba_':
                #连续特征和离散特征采用不同方式处理
                if type(key)==type(''):
                    if key.find('<=')>=0:
                        #注：暂不支持时间类型的分割，所以直接转为float
                        splitValue=float(key.replace('<=',''))
                        boolIdx=(X[bestFeature]<=splitValue)
                    elif key.find('>')>=0:
                        splitValue=float(key.replace('>',''))
                        boolIdx=(X[bestFeature]>splitValue)
                    else:
                        boolIdx=(X[bestFeature]==key)
                else:
                    boolIdx=(X[bestFeature]==key)
                p_y_.update(self.predict(
                    X[boolIdx],childDict[key],first=False))
            else:
                p_y_.loc[:,key]=childDict[key]
        if first==True:
            if return_proba==False:
                p_y_=self.choose_class(p_y_,fill_empty)
            end = time.clock()
            print('\ntime used for predict:%f'%(end-start))
        return p_y_
    
    #评估
    '''
    y:实际的分类
    p_y:预测的分类
    return> 0:准确率
    '''
    def assess(self,y,p_y):
        p_y.index=y.index
        cp=pd.DataFrame()
        cp['y'],cp['p']=y,p_y
        accuracy=len(cp[cp['y']==cp['p']])*1.0/len(y)
        return accuracy
    
    #打印结点信息
    '''
    tree:决策树,dict类型
    nodeId:当前正在打印的节点Id,不需要自主赋值
    depth:当前深度,不需要自主赋值
    print> 内节点和叶节点信息
    '''
    def print_nodes(self,tree=None,nodeId='0',depth=0):
        if type(tree)==type(None):
            tree=self.deciTree
        bestFeature=list(tree.keys())[0]
        childDict=tree[bestFeature]
        childNodeId=0
        if depth==0:
            print('\n[Nodes Info]')
        print('<inNode Id=%s pId=%s depth=%d> bestFeature:%s'
              %(nodeId,nodeId[:-1],depth,bestFeature))
        for key in childDict.keys():
            bestFeature_ = list(childDict[key].keys())[0]
            if bestFeature_!='proba_':
                print('|--%s=%s'%(bestFeature,str(key)))
                self.print_nodes(childDict[key],nodeId+str(childNodeId),depth+1)
            else:
                print('|--%s=%s'%(bestFeature,str(key)))
                print('<leafNode Id=%s pId=%s depth=%d> class:%s'
                      %(nodeId+str(childNodeId),nodeId,depth+1,str(childDict[key]['proba_'])))
            childNodeId+=1
    
    #保存树结构
    '''
    tree:决策树,dict类型
    file_path:保存文件的路径
    '''
    def save_tree(self,file_path,tree=None):
        if type(tree)==type(None):
            tree=self.deciTree
        treeStr=str(tree)
        treeStr=treeStr.replace('Interval','pd.Interval')
        file=open(file_path,'w')
        file.write(treeStr)
        file.close()
    
    #读取树结构    
    '''
    file_path:文件的路径
    return> 0:决策树,dict类型
    '''
    def read_tree(self,file_path,output=False):
        file=open(file_path,'r')
        treeStr=file.read()
        file.close()
        self.deciTree=eval(treeStr)
        if output==True:
            return self.deciTree
    
    #计算树的叶节点数
    '''
    tree:决策树,dict类型
    return> 0:叶节点数量
    '''
    def get_leaf_num(self,tree=None):
        if type(tree)==type(None):
            tree=self.deciTree
        leafNum=0
        bestFeature=list(tree.keys())[0]
        childDict=tree[bestFeature]
        #如果是中间结点就加上分支树的叶结点数，如果是叶结点则数量加1
        if bestFeature!='proba_':
            for key in childDict.keys():
                leafNum+=self.get_leaf_num(childDict[key])
        else:
            leafNum+=1
        return leafNum
    
    #计算树的深度
    '''
    tree:决策树,dict类型
    return> 0:树的深度
    '''
    def get_tree_depth(self,tree=None):
        if type(tree)==type(None):
            tree=self.deciTree
        depth_max=0
        bestFeature=list(tree.keys())[0]
        childDict=tree[bestFeature]
        for key in childDict.keys():
            #如果是中间结点就在分支树的高度上加1，如果是叶结点算作1高度
            if bestFeature!='proba_':
                depth=1+self.get_tree_depth(childDict[key])
            else:
                depth=1
            #保留分支能抵达的最大的一个高度
            if depth>depth_max:
                depth_max=depth
        return depth_max
    
    #注：可视化用于展示复杂的树会看不清
    #定义可视化格式
    style_inNode = dict(boxstyle="round4", color='#3366FF')  # 定义中间判断结点形态
    style_leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
    style_arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头
    
    #绘制带箭头的注释
    '''
    node_text:节点上的文字
    location:中心点坐标
    p_location:父节点坐标
    node_type:节点类型
    first:起始节点标志位，不用管它
    '''
    def plot_node(self,node_text, location, p_location, node_type,first):
        if first==True:
            self.ax1.annotate(node_text, xy=p_location,  xycoords='axes fraction',
                     xytext=location, textcoords='axes fraction',
                     va="center", ha="center", bbox=node_type)
        else:
            self.ax1.annotate(node_text, xy=p_location,  xycoords='axes fraction',
                     xytext=location, textcoords='axes fraction',
                     va="center", ha="center", bbox=node_type, 
                     arrowprops=self.style_arrow_args)
    
    #在父子结点间填充文本信息
    '''
    location:中心点坐标
    p_location:父节点坐标
    text:文本
    '''
    def plot_mid_text(self,location, p_location, text):
        xMid = (p_location[0]-location[0])/2.0 + location[0]
        yMid = (p_location[1]-location[1])/2.0 + location[1]
        self.ax1.text(xMid, yMid, text, va="center", ha="center", rotation=30)
    
    #绘制当前结点
    '''
    tree:决策树分支，dict类型
    location:中心点坐标
    p_location:父节点坐标
    mid_text:父子结点中间的文本
    feature_label:是否显示特征的标签
    value_label:是否显示分裂值的标签
    class_label:是否显示分类的标签
    first:起始标志位，不用管它
    '''
    def plot_tree(self,tree,p_location,mid_text,feature_label,
                  value_label,class_label,first=True):
        leafNum = self.get_leaf_num(tree)
        #depth = self.getTreeHeight(tree)
        bestFeature = list(tree.keys())[0]
        childDict = tree[bestFeature]
        xOff=self.xOff + (1.0 + float(leafNum))/2.0/self.totalW
        location = (xOff, self.yOff)
        #绘制中间结点并标记对应的划分属性值
        #print('<inNode> x:%f y:%f'%(xOff,self.yOff))
        if feature_label==True:
            inNodeText=bestFeature
        else:
            inNodeText=str(self.features.index(bestFeature))
        self.plot_mid_text(location, p_location, mid_text)
        self.plot_node(inNodeText, location, p_location, self.style_inNode,first)
        #减少y偏移
        self.yOff = self.yOff - 1.0/self.totalD
        #中间结点继续调用该方法绘制
        for key in childDict.keys(): 
            if value_label==True:
                midText=str(key)
            else:
                boolIdx=(self.mapping_X['feature']==bestFeature)&\
                        (self.mapping_X['label']==key)
                midText=self.mapping_X['valueId'][boolIdx].values[0]
            bestFeature_ = list(childDict[key].keys())[0]
            if bestFeature_!='proba_':  
                self.plot_tree(childDict[key],location,midText,
                               feature_label,value_label,class_label,False)
            #绘制叶结点
            else:
                self.xOff = self.xOff + 1.0/self.totalW
                #print('<leafNode> x:%f y:%f'%(self.xOff,self.yOff))
                class_=self.choose_class_(childDict[key])
                if class_label==True:
                    leafNodeText=class_
                else:
                    leafNodeText=self.classify.index(class_)
                self.plot_node(leafNodeText,(self.xOff, self.yOff), 
                               location, self.style_leafNode,False)
                self.plot_mid_text((self.xOff, self.yOff), location, midText)
        self.yOff = self.yOff + 1.0/self.totalD
    
    #绘制树
    '''
    tree:决策树分支，dict类型
    feature_label:是否显示特征的标签
    value_label:是否显示分裂值的标签
    class_label:是否显示分类的标签
    '''
    def plot(self,tree=None,feature_label=True,value_label=True,class_label=True):
        if type(tree)==type(None):
            tree=self.deciTree
        print('\n[Tree Plot]')
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)
        self.totalW = float(self.get_leaf_num(tree))
        self.totalD = float(self.get_tree_depth(tree))-0.9
        self.xOff = -0.5*self.totalW
        self.yOff = 1.0
        self.plot_tree(tree,(0.5,1.0),'',feature_label,value_label,class_label)
        plt.show()
    
'''
#备用代码:以类的形式存储树
#节点
class Node:
    #可以通过第一个参数传入series初始化（优先），也可以单独传入各个属性
    #--parent：父节点索引
    #--sample_n：流至该节点的训练样本数
    #--is_leaf：是否时叶节点
    #--feature：内节点参数,用于分裂的特征
    #--limit：内节点参数,限制类型（=，<=,>）
    #--value：内节点参数,用于分裂的值
    #--classify：叶节点参数，分类结果
    def __init__(self,data=None,parent=-1,sample_n=0,is_leaf=False,feature=None,
                 limit='=',value=np.NaN,classify=None):
        if type(data)!=type(None):
            self.load_series(data)
        else:
            self.load(parent,sample_n,is_leaf,feature,limit,value,classify)
     
    #用于设置属性
    #（以下几个属性不在构造节点时，而在构造树时赋值）
    #--child：子节点索引列表
    #--depth：深度
    #--idx：该节点索引，在构造树时自动赋值
    def load(self,parent,sample_n,is_leaf,feature,limit,value,classify):
        self.parent=parent
        self.childs=[]
        self.sample_n=sample_n
        self.is_leaf=is_leaf
        self.feature=feature
        self.limit=limit
        self.value=value
        self.classify=classify
        self.depth=0
        self.idx=0
    
    #通过series加载
    def load_series(self,data):
        #判断输入格式是否正确
        if type(data)!=type(pd.Series()):
            raise TypeError('The input should be series')
        #标签是否对应
        label=Node.info_label()
        if data.index.tolist()!=label:
            raise TypeError('The index do not meet the requirements')
        #加载
        self.load(data['parent'],data['sample_n'],data['is_leaf'],
                  data['feature'],data['limit'],data['value'],data['classify'])
     
    #添加子节点信息
    def add_child(self,child):
        self.childs.append(child)
      
    #将节点属性转换为字符串
    def info_to_str(self):
        if self.is_leaf==False:
            return '<inNode Id=%d pId=%d d=%d> feature:%s value:%s%f'\
                %(self.idx,self.parent,self.depth,self.feature,self.limit,self.value)
        else:
            return '<leafNode Id=%d pId=%d d=%d> classify:%s'\
                %(self.idx,self.parent,self.depth,str(self.classify))
    
    #将节点属性转换为列表
    def info_to_list(self):
        return [self.idx,self.depth,self.parent,self.childs,self.sample_n,
                self.is_leaf,self.feature,self.limit,self.value,self.classify]
    
    #节点属性标签
    def info_label():
        return ['idx','depth','parent','childs','sample_n',
                'is_leaf','feature','limit','value','classify']
        
#树
class Tree:
    #可传入dataframe初始化，也可不传入生成一个空的树
    def __init__(self,data=None):
        if type(data)!=type(None):
            self.load_dataframe(data)
        else:
            self.reset()
     
    #重置树
    def reset(self):
        self.node_count=0
        self.nodes=[]
        self.depth=0
    
    #添加节点，返回新添加节点的索引
    def add_node(self,node):
        idx=self.node_count
        node.idx=idx
        if node.parent!=-1:
            node.depth=self.nodes[node.parent].depth+1
            self.nodes[node.parent].add_child(idx)
        if node.depth>self.depth:
            self.depth=node.depth
        self.nodes.append(node)
        self.node_count+=1
        return idx
    
    #查找某节点的父节点
    def get_parent(self,node):
        return self.nodes[node.parent]
    
    #查找某节点的所有子节点
    def get_childs(self,node):
        childs=[]
        for idx in node.childs:
            childs.append(self.nodes[idx])
        return childs
    
    #将树结构转化为dataframe
    def to_dataframe(self):
        label=Node.info_label()
        nodes=[]
        for n in self.nodes:
            nodes.append(n.info_to_list())
        tree=pd.DataFrame(nodes,columns=label)
        return tree
    
    #将dataframe还原为树结构
    def load_dataframe(self,data):
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('The input should be dataframe')
        label=Node.info_label()
        if data.columns.tolist()!=label:
            raise TypeError('The columns do not meet the requirements')
        self.reset()
        for i in range(len(data)):
            dr=data.iloc[i,:]
            self.add_node(Node(dr))
    
    #获取流至该节点的路径        
    def get_path(self,node):
        path=[node.idx]
        while node.idx>0:
            node=self.get_parent(node)
            path.append(node.idx)
        path.reverse()
        return path
    
    #打印全部节点信息
    def print_nodes(self):
        for i in range(self.node_count):
            print(self.nodes[i].info_to_str())
'''             
    
    
    
    
    
            
            
            