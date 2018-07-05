# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
import user_lib.statistics as stats
import user_lib.data_prep as dp
from user_lib.check import check_type,check_limit,check_index_match,check_feats_match

#[数据结构]

#节点
class Node:
    #可以通过第一个参数传入series初始化（优先），也可以单独传入各个属性
    '''
    parent:父节点索引，int类型
    sample_n:流至该节点的训练样本数，int类型
    error:该节点上的误差，分类模式下为分类错误的个数，回归模式下为方差，float类型
    is_leaf:是否是叶节点，bool类型
    feature:内节点参数，用于分裂的特征，str类型
    limit:内节点参数，限制类型（=，<=,>,in），str类型
    value:内节点参数，用于分裂的值，float/str/list(str)类型
    output:叶节点参数，分类结果/回归预测值，dict(str->float)类型/float类型
    '''
    def __init__(self,data=None,parent=-1,sample_n=0,error=0,is_leaf=False,
                 feature=None,limit=None,value=None,output=None):
        if limit not in ['<=','>','=','in',None]:
            raise ValueError('unknown limit')
        if type(data)!=type(None):
            self.load_series(data)
        else:
            self.load(parent,sample_n,error,is_leaf,feature,limit,value,output)
     
    #用于设置属性
    '''
    （以下属性不在构造节点时，而在构造树时赋值）
    childs:子节点索引列表，list(int)类型
    childs_feature:分裂特征，str类型
    depth:深度，int类型
    nid:该节点索引，在构造树时自动赋值，int类型
    '''
    def load(self,parent,sample_n,error,is_leaf,feature,limit,value,output):
        self.parent=parent
        self.childs=[]
        self.childs_feature=None
        self.sample_n=sample_n
        self.error=error
        self.is_leaf=is_leaf
        self.feature=feature
        self.limit=limit
        self.value=value
        #output类型不符合要求时尝试转换
        try:
            if type(output)==type(None):
                output={}
            if type(output)==type(''):
                output=eval(output)
            if type(output)!=type({}):
                output=float(output)
        except:
            raise TypeError('The dtype of output should be dict or float')
        self.output=output
        self.depth=0
        self.nid=0
    
    #通过series加载
    '''
    data:节点信息，Series类型
    '''
    def load_series(self,data):
        #判断输入格式是否正确
        if type(data)!=type(pd.Series()):
            raise TypeError('The input should be series')
        #标签是否对应
        label=Node.info_label()
        if data.index.tolist()!=label:
            raise ValueError('The index do not meet the requirements')
        #加载
        self.load(data['parent'],data['sample_n'],data['error'],data['is_leaf'],
                  data['feature'],data['limit'],data['value'],data['output'])
     
    #添加子节点信息
    '''
    child:子节点，Node类型
    '''
    def add_child(self,child):
        self.childs.append(child)
      
    #将节点属性转换为字符串
    '''
    return> 0:节点信息，str类型
    '''
    def info_to_str(self):
        if self.is_leaf==False:
            if self.parent==-1:
                return '<inNode Id=%d pId=%d depth=%d> sample_n:%d error:%f'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4))
            else:
                return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                    '\n<inNode Id=%d pId=%d depth=%d> sample_n:%d error:%f'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4))
        else:
            if self.parent==-1:
                return '<leafNode Id=%d pId=%d depth=%d> sample_n:%d error:%f\n output >> %s'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4),str(self.output))
            else:
                return '|-- %s %s %s'%(self.feature,self.limit,self.value)+\
                    '\n<leafNode Id=%d pId=%d depth=%d> sample_n:%d error:%f\n output >> %s'\
                    %(self.nid,self.parent,self.depth,self.sample_n,round(self.error,4),str(self.output))
    
    #将节点属性转换为列表
    '''
    return> 0:节点属性列表，list类型
    '''
    def info_to_list(self):
        return [self.nid,self.depth,self.parent,self.childs,self.sample_n,self.error,
                self.is_leaf,self.feature,self.limit,self.value,self.output]
    
    #节点属性标签
    '''
    return> 0:节点属性标签列表，list(str)类型
    '''
    def info_label():
        return ['nid','depth','parent','childs','sample_n','error',
                'is_leaf','feature','limit','value','output']
    
    #复制节点
    def copy(self):
        new_node=Node()
        new_node.parent=self.parent
        new_node.childs=self.childs
        new_node.childs_feature=self.childs_feature
        new_node.sample_n=self.sample_n
        new_node.error=self.error
        new_node.is_leaf=self.is_leaf
        new_node.feature=self.feature
        new_node.limit=self.limit
        new_node.value=self.value
        new_node.output=self.output
        new_node.depth=self.depth
        new_node.nid=self.nid
        return new_node
        
#树
class Tree:
    #可传入dataframe初始化，也可不传入生成一个空的树
    '''
    data:完整树节点信息，DataFrame类型
    '''
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
        self.mode=None
        self.classes=[]
        self.features=[]
    
    #添加节点，返回新添加节点的索引
    '''
    node:新节点，Node类型
    return> 0:新增节点的索引，int类型
    '''
    def add_node(self,node):
        if type(node)!=type(Node()):
            raise TypeError('Input should be a Node')
        #分配节点id
        if len(self.nodes)==0:
            nid=0
        else:
            nid=self.nodes[-1].nid+1
        node.nid=nid
        #根节点和非根节点有不一样的额外处理
        if node.parent!=-1:
            #更新节点的深度，和其父节点的子节点信息
            p_node,iloc=self.find_node(node.parent,return_iloc=True)
            node.depth=p_node.depth+1
            self.nodes[iloc].add_child(nid)
            self.nodes[iloc].childs_feature=node.feature
            #更新特征列表
            self.features=list(set(self.features+[node.feature]))
        else:
            if type(node.output)==type({}):
                self.mode='Classifier'
            else:
                self.mode='Regressor'
        #更新深度和节点数量
        if node.depth>self.depth:
            self.depth=node.depth
        self.nodes.append(node)
        self.node_count+=1
        #分类模式下更新类别列表
        if self.mode=='Classifier':
            self.classes=list(set(self.classes+list(node.output)))
            self.classes.sort()
        return nid
    
    #查找节点
    '''
    node_id:节点标识，int/float类型
    start:节点列表查找的起始位置，int类型
    start:节点列表查找的结束位置(不包含)，int类型
    return_iloc:是否同时返回节点在列表中的位置索引，bool类型
    return> 0:查找到的节点，Node类型
            1:节点在列表中的位置索引，int类型
    '''
    def find_node(self,node_id,start=0,end=0,return_iloc=False):
        if end==0:
            end=self.node_count
        if (start<0)|(end>self.node_count)|(end<=start):
            raise IndexError('Invalid range')
        try:
            nodeId=int(node_id)
        except:
            raise TypeError('Unrecognized dtype of node_id')
        if nodeId<0:
            raise IndexError('Fail to find node(nid=%d)'%node_id)
        for i in range(start,end):
            if self.nodes[i].nid==node_id:
                if return_iloc==True:
                    return self.nodes[i],i
                else:
                    return self.nodes[i]
        raise IndexError('Fail to find node(nid=%d)'%node_id)
    
    #查找某节点的父节点
    '''
    node_id:节点标识，int/float类型
    return> 0:父节点，Node类型
    '''
    def get_parent(self,node_id):
        node,iloc=self.find_node(node_id,return_iloc=True)
        p_node=self.find_node(node.parent,end=iloc)
        return p_node
    
    #查找某节点的所有子节点
    '''
    node_id:节点标识，int/float类型
    return> 0:子节点列表，list(Node)类型
    '''
    def get_childs(self,node_id):
        node,iloc=self.find_node(node_id,return_iloc=True)
        childs=[]
        for nid in node.childs:
            node,iloc=self.find_node(nid,start=iloc+1)
            childs.append(node)
        return childs
    
    #将树结构转化为dataframe
    '''
    return> 0:全部树节点信息，DataFrame类型
    '''
    def to_dataframe(self):
        label=Node.info_label()
        nodes=[]
        for n in self.nodes:
            nodes.append(n.info_to_list())
        df_tree=pd.DataFrame(nodes,columns=label)
        return df_tree
    
    #将dataframe还原为树结构
    '''
    data:全部树节点信息，DataFrame类型
    '''
    def load_dataframe(self,data):
        if type(data)!=type(pd.DataFrame()):
            raise TypeError('The input should be dataframe')
        label=Node.info_label()
        if data.columns.tolist()!=label:
            raise ValueError('The columns do not meet the requirements')
        self.reset()
        for i in range(len(data)):
            dr=data.iloc[i,:]
            self.add_node(Node(dr))
    
    #获取流至该节点的路径 
    '''
    node_id:节点标识，int/float类型
    return_nodes:是否返回节点列表，False时返回id列表，bool类型
    return> 0:流经节点索引列表，list(int)类型
    '''       
    def get_path(self,node_id,return_nodes=False):
        node=self.find_node(node_id)
        if return_nodes==False: 
            path=[node.nid]
            while node.nid>0:
                node=self.get_parent(node.nid)
                path.append(node.nid)
        else:
            path=[node]
            while node.nid>0:
                node=self.get_parent(node.nid)
                path.append(node)
        path.reverse()
        return path
    
    #打印全部节点信息
    def print_nodes(self):
        for i in range(self.node_count):
            print(self.nodes[i].info_to_str())     
            
    #计算树的叶节点数
    '''
    start_id:起始的节点id，int类型
    return> 0:叶节点数量，int类型
    '''
    def get_leaf_num(self,start_id=0):
        leafNum=0
        node,iloc=self.find_node(start_id,return_iloc=True)
        queue=[node]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                leafNum+=1
            else:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append(child)
        return leafNum
    
    #计算树的深度
    '''
    start_id:起始的节点id，int类型
    return> 0:树的深度，int类型
    '''
    def get_depth(self,start_id=0):
        depth_max=0
        node,iloc=self.find_node(start_id,return_iloc=True)
        depth_start=node.depth
        queue=[node]
        while len(queue)>0:
            node=queue.pop(0)
            if node.is_leaf==True:
                depth=node.depth-depth_start
                if depth>depth_max:
                    depth_max=depth
            else:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append(child)
        return depth_max        
    
    #单次剪枝操作，将指定id的节点下属所有节点移除并将该节点改为叶节点
    '''
    node_id:指定节点id，int类型
    '''
    def cut(self,node_id,return_trash=False):
        cut_node,first_iloc=self.find_node(node_id,return_iloc=True)
        #叶节点不能剪枝
        if cut_node.is_leaf==True:
            raise IndexError('LeafNode can not be cut')
        #筛选所有需要删除的下属节点
        iloc=first_iloc
        remove=[(cut_node,first_iloc)]
        for node,iloc in remove:
            if node.is_leaf==False:
                for childId in node.childs:
                    child,iloc=self.find_node(childId,start=iloc+1,return_iloc=True)
                    remove.append((child,iloc))
        #倒序逐个按位置索引删除节点
        remove.reverse()
        trash=[]
        for node,iloc in remove:
            if iloc==first_iloc:
                trans_node=self.nodes[iloc].copy()
                trans_node.is_leaf=True
                trans_node.childs=[]
                trans_node.childs_feature=None
                self.nodes[iloc]=trans_node
            else:
                trash.append(self.nodes.pop(iloc))
        #更新树信息
        self.node_count=len(self.nodes)
        self.depth=self.get_depth()
        #返回删除的节点
        if return_trash==True:
            return trash
    
    #复制树
    def copy(self,deep=False):
        new_tree=Tree()
        new_tree.node_count=self.node_count
        if deep==True:
            new_tree.nodes=[node.copy() for node in self.nodes]
        else:
            new_tree.nodes=self.nodes.copy()
        new_tree.depth=self.depth
        new_tree.mode=self.mode
        new_tree.classes=self.classes.copy()
        return new_tree

#[运算函数]

#信息熵etp=sum(p*log2(p))
#p=每个取值value的占比
#表示随机变量不确定性的度量，范围0~log2(n)，数值越大不确定性越大,n为离散值种类数
#0log0=0 ；当对数的底为2时，熵的单位为bit；为e时，单位为nat。
'''
array:需要求熵的数据列，narray类型
continuity:连续性，bool类型
value:分裂点，float类型，只有在处理连续数据时有意义
return> 0:熵，float类型
'''
@nb.jit(nopython=True,cache=True)
def entropy(array,continuity=False,value=0):
    #数据集大小,初始化熵
    n,etp=len(array),0.0
    #是否是连续数据,连续数据按指定阈值分裂成两部分,离散数据按每个取值分裂
    if continuity==True:
        #统计左半部分大小，由于只有两个子集，另一半不需要另外统计
        count=0       
        for i in range(n):
            if array[i]<=value:
                count+=1
        p=count/n
        etp=-p*np.log2(p)-(1-p)*np.log2(1-p)
    else:
        #去重并统计
        values,counts=dp.unique_count(array)
        #遍历每一个值计算
        for i in range(len(values)):
            p=counts[i]/n
            etp-=p*np.log2(p)
    return etp

#条件熵con_etp=sum(p*etp)
#p,etp=x的各个取值的数量占比以及按x的值分裂后每个子集y的熵
'''
x:用于分裂的特征列，narray类型
y:分类列，narray类型
continuity:连续性，bool类型
value:分裂点，float类型,只有在处理连续数据时有意义
return> 0:条件熵，float类型
'''
@nb.jit(nopython=True,cache=True)
def con_entropy(x,y,continuity=False,value=0):
    n=len(x)
    #连续特征和离散特征采用不同计算方式
    if continuity==True:
        boolIdx=(x<=value)
        p=len(x[boolIdx])/n
        con_ent=p*entropy(y[boolIdx])\
            +(1-p)*entropy(y[~boolIdx])
    else:
        #去重并统计
        values,counts=dp.unique_count(x)
        con_ent=0.0
        #遍历每个取值
        for i in range(len(values)):
            y_=y[x==values[i]]
            p=counts[i]/n
            con_ent+=p*entropy(y_)
    return con_ent

#基尼指数gini=1-sum(p*p)
#p=每个取值value的占比
#表示随机变量不确定性的度量，范围0~1，数值越大不确定性越大
#基尼指数的计算一般只针对离散分类，不用于特征
'''
array:需要求基尼指数的数据列,narray类型
return> 0:基尼指数，float类型
'''
@nb.jit(nopython=True,cache=True)
def gini(array):
    #数据集大小,初始化基尼指数
    n,g=len(array),1.0
    #先排序和初始化变量
    values,counts=dp.unique_count(array)
    #遍历每一个元素
    for i in range(len(values)):
        p=counts[i]/n
        g-=p*p
    return g

#条件基尼
'''
x:用于分裂的特征列,narray类型
y:需要求基尼指数的分类列,narray类型
return> 0:基尼指数，float类型
'''
@nb.jit(nopython=True,cache=True)
def con_gini(x,y,continuity=False,value=[]):
    n,con_gini=len(x),0.0
    #连续特征和离散特征采用不同计算方式
    if continuity==True:
        boolIdx=(x<=value[0])
    else:
        boolIdx=dp.isin(x,value)
    p=len(x[boolIdx])/n
    con_gini=p*gini(y[boolIdx])\
        +(1-p)*gini(y[~boolIdx])
    return con_gini

#平方误差
'''
array:需要求平方误差的数据列,narray类型
return> 0:平方误差，float类型
'''
@nb.jit(nopython=True,cache=True)
def sqr_err(array):
    re=array-np.mean(array)
    return np.dot(re.T,re)

#条件平方误差
'''
x:用于分裂的特征列,narray类型
y:需要求平方误差的分类列,narray类型
return> 0:平方误差，float类型
'''
@nb.jit(nopython=True,cache=True)
def con_sqr_err(x,y,continuity=False,value=[]):
    con_sqr_err_=0.0
    #连续特征和离散特征采用不同计算方式
    if continuity==True:
        boolIdx=(x<=value[0])
    else:
        boolIdx=dp.isin(x,value)
    con_sqr_err_=sqr_err(y[boolIdx])+sqr_err(y[~boolIdx])
    return con_sqr_err_

#连续特征最优分裂阈值选择
#注：适用于c4.5,cart
'''
x:用于分裂的特征列，连续变量，narray类型
y:分类列，narray类型
criterion:衡量标准，1/2/3->entropy/gini/square_error，int类型
return> 0:最优分裂阈值，float类型 
        1:所有可能的分裂阈值的数量，int类型
'''
@nb.jit(nopython=True,cache=True)
def choose_threshold(x,y,criterion):
    #需要尝试的分裂点
    values,n=filter_thresholds(x,y,criterion)
    #初始化变量
    if criterion==1:
        bestCriterion=entropy(y)
    elif criterion==2:
        bestCriterion=gini(y)
    elif criterion==3:
        bestCriterion=sqr_err(y)
    else:
        return 0.0,0
    bestThreshold=values[0]
    #逐个计算所有可能分裂点的衡量指数
    for j in range(len(values)-1):
        threshold=values[j]
        if criterion==1:
            c=con_entropy(x,y,True,threshold)
        elif criterion==2:
            c=con_gini(x,y,True,[threshold])
        elif criterion==3:
            c=con_sqr_err(x,y,True,[threshold])
        if c<bestCriterion:
            bestCriterion=c
            bestThreshold=threshold
    return bestThreshold,n

#筛选分裂阈值
'''
x:用于分裂的特征列，narray类型
y:分类列，narray类型
criterion:衡量标准，1/2/3->entropy/gini/square_error，int类型
return> 0:经过筛选的分裂阈值列表，list(float)类型
        1:所有可能的分裂阈值的数量，int类型
'''
@nb.jit(nopython=True,cache=True)
def filter_thresholds(x,y,criterion):
    n=len(x)
    #将x,y按x升序排序
    sortIdx=np.argsort(x)
    x,y=x[sortIdx],y[sortIdx]
    if criterion==3:
        #需要选取的点的布尔索引(因为不能直接初始化bool类型所以用int代替)
        filterIdx=np.zeros(n)
        filterIdx[0],filterIdx[n-1]=1,1
        #将分类结果y有变化的位置选取出来
        for i in range(n-1):
            if y[i]!=y[i+1]:
                filterIdx[i],filterIdx[i+1]=1,1
        values,counts=dp.unique_count(x[filterIdx==1])
    else:
        values,counts=dp.unique_count(x)
    return values,n

#离散特征最优分裂组合选择
#注：适用于cart
'''
x:用于分裂的特征列,Series类型
y:分类列,Series类型
criterion:衡量标准，2/3->gini/square_error，int类型
simplify:简化模式，True时使用ovr(one vs rest)代表每个取值与其他值构成一种二分组合
    False时mvm(many vs many)代表无限制的所有二分组合，在取值个数多时该模式会非常耗时
return> 0:最优分裂组合左子集，list类型 
        1:右子集，list类型
'''
@nb.jit(nopython=True,cache=True)
def choose_combine(x,y,criterion,simplify=False):
    #需要尝试的分裂组合
    values,counts=dp.unique_count(x)
    if len(values)>1:
        if simplify==False:
            combines=dp.combine_enum(values,split_mode=True)
        else:
            combines=np.eye(len(values),len(values))
        #初始化变量
        if criterion==2:
            bestCriterion=gini(y)
        elif criterion==3:
            bestCriterion=sqr_err(y)
        else:
            return [0],[0]
        bestCombineIdx=-1
        #逐个计算所有可能分裂方式的基尼系数
        for i in range(len(combines)):
            combine=dp.combine_take(values,combines[i])
            if criterion==2:
                c=con_gini(x,y,False,combine)
            elif criterion==3:
                c=con_sqr_err(x,y,False,combine)
            if c<bestCriterion:
                bestCriterion=c
                bestCombineIdx=i
        #选择失败
        if bestCombineIdx==-1:
            return [0],[0]
        #生成左右子集取值列表
        bestCombine=combines[bestCombineIdx]
        left,right=[],[]
        for j in range(len(values)):
            if bestCombine[j]==1:
                left.append(values[j])
            else:
                right.append(values[j])
        return left,right
    else:
        return [0],[0]

#[构造类]

#决策树
class DecisionTree:
    '''\n  
    Note: 以展示各个算法的实现为目的，id3其实不太实用
    
    Parameters
    ----------
    model_type : 模型算法类型，str类型(id3,c4.5,cart_c,cart_r)，
                 'id3'->分类，离散特征+离散输出，
                 'c4.5','cart_c'->分类，离散或连续特征+离散输出，
                 'cart_r'->回归，离散或连续特征+连续输出，
                 默认值'cart_c'
    depth_max : 最大深度，int类型(>=1)，None表示无限制，默认值10
    split_min_n : 分裂所需最少样本数，int类型(>=2)，默认值2
    leaf_min_n : 叶节点所需最少样本数，int类型(>=1)，默认值1
    feature_use : 每次使用的特征数量，str/float/int类型
                  'all'->全量，
                  'sqrt'->总数量的平方根，
                  'log2'->总数量的2的对数，
                  float->总数量的相应比例，区间(0.0,1.0)，
                  int->固定数量，区间[1,feature_num]，
                  默认值'sqrt'
    feature_reuse : 是否允许一个特征重复使用，bool类型，默认值False
    ----------
    '''
    
    #构造函数，主要作用是校验和保存配置变量
    def __init__(self,model_type='cart_c',depth_max=10,split_min_n=2,
                 leaf_min_n=1,feature_use='sqrt',feature_reuse=False):
        #校验参数类型和取值
        #check_type(变量名，变量类型，要求类型)
        #check_limit(变量名，限制条件，正确取值提示)
        check_type('model_type',type(model_type),type(''))
        check_type('split_min_n',type(split_min_n),type(0))
        check_type('leaf_min_n',type(leaf_min_n),type(0))
        check_type('feature_reuse',type(feature_reuse),type(True))
        if type(depth_max)!=type(None):
            check_type('depth_max',type(depth_max),type(0))
        check_type('feature_use',type(feature_use),[type(0),type(1.0),type('')])
        type_list=['id3','c4.5','cart_c','cart_r']
        model_type=model_type.lower()
        check_limit('model_type',model_type in type_list,str(type_list))
        check_limit('split_min_n',split_min_n>=2,'value>=2')
        check_limit('leaf_min_n',leaf_min_n>=1,'value>=1')
        required="float(>0.0,<1.0),int(>=1,<=feature_n),str(['all','sqrt','log2'])"
        if type(feature_use)==type(''):
            check_limit('feature_use',feature_use in ['all','sqrt','log2'],required)
        elif type(feature_use)==type(0):
            check_limit('feature_use',feature_use>=1,required)
        elif type(feature_use)==type(0.0):
            check_limit('feature_use',(feature_use>0.0)&(feature_use<1.0),required)
        #保存参数
        self.model_type=model_type
        self.depth_max=depth_max
        self.split_min_n=split_min_n
        self.leaf_min_n=leaf_min_n
        self.feature_reuse=feature_reuse
        self.feature_use=feature_use

    #最优特征选择(ID3)
    #选择依据：信息增益
    #=划分前类别的信息熵-划分后类别的条件熵
    #用于衡量经过某特征的划分后分类的不确定性降低了多少
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    return> 0:最优分裂特征的索引，int类型
            1:最优信息增益，float类型
    '''
    def choose_feature_by_id3_(self,X,y):
        #随机抽取特征
        feature_use_n=self.feature_use_n
        feature_n=len(X.columns)
        if feature_use_n>feature_n:
            feature_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(feature_use_n)
        #计算分裂前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGain=0.0
        bestFeatureIdx=-1
        #逐个计算按不同特征分割后的信息增益并选出增益最大的一个特征
        for i in features_idx:
            x=X.iloc[:,i]
            #特征列值统一，无法用于分裂
            if len(x.drop_duplicates())<=1:
                continue
            infGain=baseEntropy-con_entropy(x.values,y.values)
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
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    continuity:连续性，list(bool)类型
    return> 0:最优分裂特征的索引 ，int类型
            1:最优信息增益比，float类型
            2：最优分裂值，float类型
    '''
    def choose_feature_by_c45_(self,X,y,continuity):
        #随机抽取特征
        feature_use_n=self.feature_use_n
        feature_n=len(X.columns)
        if feature_use_n>feature_n:
            feature_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(feature_use_n)
        #计算分裂前的信息熵
        baseEntropy=entropy(y.values)
        #初始化变量
        bestInfGainRatio=0.0
        bestFeatureIdx=-1
        bestSplitValue=0.0
        #逐个计算按不同特征分割后的信息增益率并选出增益率最大的一个特征
        for i in features_idx:
            x=X.iloc[:,i]
            #特征列值统一，无法用于分裂
            if len(x.drop_duplicates())<=1:
                continue
            #是否为连续特征
            if continuity[i]==True:
                splitValue,n=choose_threshold(x.values,y.values,1)
                splitFeatEntropy=entropy(x.values,continuity[i],splitValue)
                infGain=baseEntropy\
                    -con_entropy(x.values,y.values,continuity[i],splitValue)
                    #-np.log2(n-1)/len(x)
            else:
                splitValue=0.0
                splitFeatEntropy=entropy(x.values,continuity[i])
                infGain=baseEntropy-con_entropy(x.values,y.values,continuity[i])
            if splitFeatEntropy==0:
                continue
            infGainRatio=infGain/splitFeatEntropy
            if infGainRatio>bestInfGainRatio:
                bestInfGainRatio=infGainRatio
                bestFeatureIdx=i
                bestSplitValue=splitValue
        return bestFeatureIdx,bestSplitValue
       
    #最优特征选择(CART分类树)
    #选择依据：基尼指数
    #分类树输出离散数值
    #cart和c4.5采用同样的方式处理连续特征
    #同时cart对离散特征也进行二分处理，将离散值划分到两个子集里，选择最优的一种分裂
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    continuity:连续性，list(bool)类型
    return> 0:最优分裂特征的索引，int类型 
            1:最优基尼指数，float类型
            2:最优分裂方案，float或list(list(int))类型
    '''
    def choose_feature_by_cart_c_(self,X,y,continuity):
        #随机抽取特征
        feature_use_n=self.feature_use_n
        feature_n=len(X.columns)
        if feature_use_n>feature_n:
            feature_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(feature_use_n)
        #初始化变量
        bestGini=gini(y.values)
        bestFeatureIdx=-1
        bestSplit=None
        #逐个计算按不同特征分割后的基尼指数并选出指数最小的一个特征
        for i in features_idx:
            x=X.iloc[:,i]
            #特征列值统一，无法用于分裂
            if len(x.drop_duplicates())<=1:
                continue
            #是否为连续特征
            if continuity[i]==True:
                splitValue,n=choose_threshold(x.values,y.values,2)
                g=con_gini(x.values,y.values,continuity[i],[splitValue])
                temp=splitValue
            else:
                left,right=choose_combine(x.values,y.values,2)
                if left==right:
                    continue
                g=con_gini(x.values,y.values,continuity[i],left)
                temp=[left,right]
            if g<bestGini:
                bestGini=g
                bestFeatureIdx=i
                bestSplit=temp
        return bestFeatureIdx,bestSplit
    
    #最优特征选择(CART回归树)
    #选择依据：方差
    #回归树输出连续数值
    '''
    X:所有参与选择的特征列，DataFrame类型
    y:分类列，Series类型
    continuity:连续性，list(bool)类型
    return> 0:最优分裂特征的索引，int类型 
            1:最优方差，float类型
            2:最优分裂方案，float或list(list(int))类型
    '''
    def choose_feature_by_cart_r_(self,X,y,continuity):
        #随机抽取特征
        feature_use_n=self.feature_use_n
        feature_n=len(X.columns)
        if feature_use_n>feature_n:
            feature_use_n=feature_n
        features_idx=pd.Series(range(feature_n))
        features_idx=features_idx.sample(feature_use_n)
        #初始化变量
        bestErr=sqr_err(y.values)
        bestFeatureIdx=-1
        bestSplit=None
        #逐个计算按不同特征分割后的方差并选出方差最小的一个特征
        for i in features_idx:
            x=X.iloc[:,i]
            #特征列值统一，无法用于分裂
            if len(x.drop_duplicates())<=1:
                continue
            #是否为连续特征
            if continuity[i]==True:
                splitValue,n=choose_threshold(x.values,y.values,3)
                e=con_sqr_err(x.values,y.values,continuity[i],[splitValue])
                temp=splitValue
            else:
                left,right=choose_combine(x.values,y.values,3)
                if left==right:
                    continue
                e=con_sqr_err(x.values,y.values,continuity[i],left)
                temp=[left,right]
            if e<bestErr:
                bestErr=e
                bestFeatureIdx=i
                bestSplit=temp
        return bestFeatureIdx,bestSplit
    
    #计算每个类的概率
    '''
    y:分类列，Series类型
    return> 0:分类概率，dict类型
    '''
    def compute_proba_(self,y,return_counts=False):
        proba={}
        values,counts=dp.unique_count(y.values)
        total_count=counts.sum()
        for i in range(len(values)):
            p=counts[i]/total_count
            value=dp.index_to_label_(y.name,values[i],self.mapping_y)
            proba[value]=p
        if return_counts==True:
            return proba,values,counts
        else:
            return proba
          
    #选择概率最高的类作为叶节点判定的类,用于预测
    '''
    p_y_:预测的分类概率，DataFrame类型
    return> 0:选择的类，Series类型
    '''
    def choose_class_(self,p_y_,classes):
        p_max=p_y_.max(axis=1)
        p_y=pd.Series(np.full(len(p_y_),''),index=p_y_.index)
        for i in range(len(classes)):
            p_y[p_y_.iloc[:,i]==p_max]=classes[i]
        #按类别分布情况加权随机填充未能分类的记录
        nullIdx=(p_y=='')
        n=p_y[nullIdx].count()
        if n>0:
            p_y.loc[nullIdx]=p_y[~nullIdx].sample(n=n,replace=True).tolist()
        return p_y
 
    #根据第i列特征分裂数据集
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    continuity:连续性，bool类型
    value:分裂依据，二分裂时使用，连续数据->float类型/离散数据->list类型
    return> 0:X分裂后的集合，list(DataFrame) 
            1:y分裂后的集合，list(Series) 
            2:分裂条件集合，list(tuple)
            3:最小子集样本数，int类型
    '''
    def split_(self,X,y,i,continuity=False,value=None):
        #抽取第i列特征
        x=X.iloc[:,i]
        featLabel=X.columns[i]
        #连续特征和离散特征采用不同的处理方式
        if continuity==True:
            if type(value)==type(None):
                raise TypeError('must provide split value for continuity feature')
            #根据分裂点将数据集拆分
            values=[('<=',value),('>',value)]
            boolIdx=(x<=value)
            result_X=[X[boolIdx],X[~boolIdx]]
            result_y=[y[boolIdx],y[~boolIdx]]
            min_sample_n=min(len(result_y[0]),len(result_y[1]))
        else:
            if type(value)==type([]):
                boolIdx=(x.isin(value[0]))
                result_X=[X[boolIdx],X[~boolIdx]]
                result_y=[y[boolIdx],y[~boolIdx]]
                values=[('in',[dp.index_to_label_(featLabel,m,self.mapping_X) for m in value[0]]),
                        ('in',[dp.index_to_label_(featLabel,n,self.mapping_X) for n in value[1]])]
                min_sample_n=min(len(result_y[0]),len(result_y[1]))
            else:
                #去重得到特征值列表
                values=x.sort_values().drop_duplicates().tolist()
                #根据不同的特征值进行分割
                result_X,result_y=[],[]
                min_sample_n=len(y)
                for j in range(len(values)):
                    result_X.append(X[x==values[j]])
                    result_y.append(y[x==values[j]])
                    values[j]=('=',dp.index_to_label_(featLabel,values[j],self.mapping_X))
                    sample_n=len(result_y[-1])
                    if sample_n<min_sample_n:
                        min_sample_n=sample_n
        return result_X,result_y,values,min_sample_n
    
    #叶节点判断(仅先行判断的条件，非全部条件)
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    depth:当前深度，int类型
    return> 0:是否叶节点，bool类型
    '''
    def is_leaf_(self,X,y,depth):
        #超出高度上限，不继续分裂
        if self.depth_max!=None:
            if depth>self.depth_max:
                if self.build_proc==True:
                    print('<LeafNode> reach maximum depth')
                return True
        #可用特征不足，不继续分裂
        if len(X.columns)==0:
            if self.build_proc==True:
                print('<LeafNode> lack of feature')
            return True
        #数据集过小，不继续分裂
        if len(X)<self.split_min_n:
            if self.build_proc==True:
                print('<LeafNode> samples too small')
            return True
        #只有一个类，不继续分裂
        if len(y.drop_duplicates())==1:
            if self.build_proc==True:
                print('<LeafNode> only one class')
            return True
        #特征向量统一，不继续分裂
        if len(X.drop_duplicates())==1:
            if self.build_proc==True:
                print('<LeafNode> feature vector unification')
            return True
        return False
    
    #构建树
    '''
    X:当前所有的特征列，DataFrame类型
    y:分类列，Series类型
    return> 0:决策树，Tree类型
    '''
    def build_(self,X,y):
        #初始化决策树
        deciTree=Tree()
        #等待处理的数据队列：特征，分类，连续性，父节点id，深度，
        #                   分裂特征名,约束方式，分裂值
        queue=[(X,y,self.continuity_X,-1,0,None,None,None)]
        while len(queue)>0:
            start0=time.clock()
            X_,y_,continuity_,parent,depth,feature,limit,value=queue.pop(0)
            self.time_cost['queue operate']+=time.clock()-start0
            if self.build_proc==True:
                if parent!=-1:
                    print('<Split> %s %s %s'%
                          (feature,limit,str(value)))
                print('Current dataset size: %d'%len(y_))
            start0=time.clock()
            is_leaf=self.is_leaf_(X_,y_,depth)
            self.time_cost['check input']+=time.clock()-start0
            #选择最优特征进行分裂，并记录结果
            if is_leaf==False:
                start0=time.clock()
                try:
                    if self.model_type=='id3':
                        bestFeatureIdx,bestSplit=self.choose_feature_by_id3_(X_,y_),None
                    elif self.model_type=='c4.5':
                        bestFeatureIdx,bestSplit=self.choose_feature_by_c45_(X_,y_,continuity_)
                    elif self.model_type=='cart_c':
                        bestFeatureIdx,bestSplit=self.choose_feature_by_cart_c_(X_,y_,continuity_)
                    elif self.model_type=='cart_r':
                        bestFeatureIdx,bestSplit=self.choose_feature_by_cart_r_(X_,y_,continuity_)
                    else:
                        raise TypeError('Unknown type')
                except:
                    self.err_X,self.err_y,self.err_con=X_,y_,continuity_
                    print('subsets of data which cause error are saved as .err_X,.err_y,.err_con')
                    raise
                self.time_cost['compute best split']+=time.clock()-start0
                #未能成功选出可供分裂的特征
                if bestFeatureIdx==-1:
                    if self.build_proc==True:
                        print('<LeafNode> fail to choose a feature')
                    is_leaf=True
            #当前节点分类概率/回归值和误差个数计算
            start0=time.clock()
            if self.model_type=='cart_r':
                output=y_.mean()
                error=sqr_err(y_.values)
            else:
                output,values,counts=self.compute_proba_(y_,return_counts=True)
                error=counts.sum()-counts.max()
            self.time_cost['compute node attr']+=time.clock()-start0
            if is_leaf==True:
                #添加叶节点
                start0=time.clock()
                node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=True,
                          feature=feature,limit=limit,value=value,output=output)
                nodeId=deciTree.add_node(node)
                self.time_cost['add node']+=time.clock()-start0
            else:
                #获取最优分裂特征的相关信息
                bestFeatureLabel=X_.columns[bestFeatureIdx]
                #分裂数据集
                start0=time.clock()
                splited_X,splited_y,split_values,min_sample_n=\
                    self.split_(X_,y_,bestFeatureIdx,
                                continuity_[bestFeatureIdx],bestSplit)
                self.time_cost['split data']+=time.clock()-start0
                #下属叶节点样本数存在小于设定值的，将该节点设为叶节点，否则内节点
                if min_sample_n<self.leaf_min_n:
                    start0=time.clock()
                    node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=True,
                    feature=feature,limit=limit,value=value,output=output)
                    nodeId=deciTree.add_node(node)
                    self.time_cost['add node']+=time.clock()-start0
                else:
                    start0=time.clock()
                    node=Node(parent=parent,sample_n=len(y_),error=error,is_leaf=False,
                              feature=feature,limit=limit,value=value,output=output)
                    nodeId=deciTree.add_node(node)
                    self.time_cost['add node']+=time.clock()-start0
                    if self.feature_reuse==False:
                        continuity__=continuity_.copy()
                        continuity__.pop(bestFeatureIdx)
                        #将分裂后的数据集加入队列继续处理
                        #注：目前的设置为每个特征仅用一次，可尝试变更
                        start0=time.clock()
                        for i in range(len(split_values)):
                            queue.append((splited_X[i].drop(bestFeatureLabel,axis=1),
                                          splited_y[i],continuity__,nodeId,depth+1,
                                          bestFeatureLabel,split_values[i][0],split_values[i][1]))
                        self.time_cost['queue operate']+=time.clock()-start0
                    else:
                        start0=time.clock()
                        for i in range(len(split_values)):
                            queue.append((splited_X[i],splited_y[i],continuity_,nodeId,depth+1,
                                          bestFeatureLabel,split_values[i][0],split_values[i][1]))
                        self.time_cost['queue operate']+=time.clock()-start0
        return deciTree
    
    #fit方法的输入校验   
    def fit_check_input_(self,X,y):
        #类型校验
        check_type('X',type(X),type(pd.DataFrame()))
        check_type('y',type(y),type(pd.Series()))
        #校验X,y输入是否匹配
        check_index_match(X,y,'X','y')
        #连续性判断
        continuity_X=dp.get_continuity(X,'X')
        continuity_y=dp.get_continuity(y,'y')
        #ID3不支持连续特征
        if self.model_type=='id3':
            if True in continuity_X:
                raise TypeError('ID3 does not support continuous features')
        #cart回归的y必须是数值型,其他分类树的y转换为str类型
        if self.model_type=='cart_r':
            if continuity_y==False:
                raise TypeError('CART Regressor only support y for numeric')
        else:
            y=y.astype('str')
        #确定每次分裂考虑的特征数量上限
        feature_n=len(X.columns)
        if type(self.feature_use)==type(''):
            if self.feature_use=='all':
                feature_use_n=feature_n
            elif self.feature_use=='sqrt':
                feature_use_n=np.sqrt(feature_n)
            elif self.feature_use=='log2':
                feature_use_n=np.log2(feature_n)
        elif type(self.feature_use)==type(0):
            if self.feature_use>feature_n:
                feature_use_n=feature_n
            else:
                feature_use_n=self.feature_use
        elif type(self.feature_use)==type(0.0):
            feature_use_n=self.feature_use*feature_n
        #视情况将X,y转换为数值索引(numba的需要)
        if False not in continuity_X:
            mapping_X=None
        else:
            cols=[]
            for i in range(len(continuity_X)):
                if continuity_X[i]==False:
                    cols.append(X.columns[i])
            X,mapping_X=dp.label_to_index(X,cols)
        if self.model_type=='cart_r':
            mapping_y=None
        else:
            y,mapping_y=dp.label_to_index(y,[y.name])
        #保存生成参数
        self.continuity_X=continuity_X
        self.continuity_y=continuity_y
        self.mapping_X=mapping_X
        self.mapping_y=mapping_y
        self.feature_use_n=int(feature_use_n)
        return X,y
    
    #拟合
    def fit(self,X,y,output=False,show_time=False,
            build_proc=False,check_input=True):
        '''\n
        Function: 使用输入数据拟合决策树
        
        Note: 数据列的连续性会进行自动判断，不被支持的类型需要预处理
              (int64,float64)->连续
              (bool,category,object)->离散
              所有离散数据会强制转换为str标签

        Parameters
        ----------
        X: 所有的特征列，DataFrame类型
        y: 分类列，Series类型
        output: 是否输出拟合好的树，bool类型，默认值False
        show_time: 是否显示耗时，bool类型，默认值False
        build_proc: 反馈构建过程，bool类型，默认值False
        check_input: 是否进行输入校验，bool类型，默认值True
        ----------
        
        Returns
        -------
        0: 决策树，Tree类型
        -------
        '''
        start = time.clock()
        #输入校验
        check_type('output',type(output),type(True))
        check_type('show_time',type(show_time),type(True))
        check_type('build_proc',type(build_proc),type(True))
        check_type('check_input',type(check_input),type(True))
        if check_input==True:
            X,y=self.fit_check_input_(X,y)
        self.time_cost=pd.Series(np.zeros(7),name='time cost',
                index=['total cost','queue operate','check input',
                       'compute best split','compute node attr',
                       'split data','add node'])
        self.build_proc=build_proc
        #构建树
        self.tree=self.build_(X,y)
        end = time.clock()
        self.time_cost['total cost']=end-start
        if show_time==True:
            print('\ntime used for trainning: %f'%(end-start))
        if output==True:
            return self.tree
        
    #决策路径
    def decition_path(self,dr,tree=None,return_path=False):
        '''\n
        Function: 获取单行数据的决策路径
        
        Note: 只能一次处理一行数据，主要用于校对，
        需要批量获取数据的决策路径可以使用predict，设置参数return_paths=True
        
        Parameters
        ----------
        dr: 数据行，Series类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        return_path: 是否仅以返回值形式给到路径，bool类型，默认值False，
                     此模式只返回节点id列表，不包含完整信息
        ----------
        
        Print
        -----
        0: 流经节点的信息
        -----
        
        Returns
        -----
        0: 流经节点，list(int)类型
        -----
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        #校验输入
        check_type('return_path',type(return_path),type(True))
        check_type('dr',type(dr),type(pd.Series()))
        check_feats_match(dr.index,tree.features,'dr','tree',mode='right')
        #准备访问的节点
        iloc=0
        queue=[tree.nodes[0]]
        print(tree.nodes[0].info_to_str())
        #直到没有等待访问的节点时结束
        while len(queue)>0:
            node=queue.pop(0)
            bestFeature=node.childs_feature
            match_flag=False
            #访问每个子节点查找分裂条件匹配的对象
            for childId in node.childs:
                child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                if child.limit=='<=':
                    if dr[bestFeature]<=child.value:
                        match_flag=True
                elif child.limit=='>':
                    if dr[bestFeature]>child.value:
                        match_flag=True
                elif child.limit=='in':
                    if dr[bestFeature] in child.value:
                        match_flag=True
                else:
                    if dr[bestFeature]==child.value:
                        match_flag=True
                #打印匹配上的节点信息，如果是内节点再加入待访问队列
                if match_flag==True:
                    print(child.info_to_str())
                    if child.is_leaf==False:
                        queue.append(child)
                    break
        return
    
    #流至节点
    def flow_to_node(self,X,node_id,tree=None):
        '''\n
        Function: 获取数据集流至指定节点处的子集
        
        Parameters
        ----------
        X: 所有的特征列，DataFrame类型
        node_id: 节点Id，int类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Returns
        -------
        0:流至该节点的数据，DataFrame类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        #校验输入
        check_type('node_id',type(node_id),type(0))
        check_type('X',type(X),type(pd.DataFrame()))
        check_feats_match(X.columns,tree.features,'X','tree',mode='right')
        X_=X.copy()
        path_nodes=tree.get_path(node_id,return_nodes=True)
        for node in path_nodes:
            if node.nid==0:
                continue
            if node.limit=='<=':
                X_=X_[X_[node.feature]<=node.value]
            elif node.limit=='>':
                X_=X_[X_[node.feature]>node.value]
            elif node.limit=='in':
                X_=X_[X_[node.feature].isin(node.value)]
            else:
                X_=X_[X_[node.feature]==node.value]
        return X_
    
    #预测
    def predict(self,X,tree=None,return_proba=False,return_paths=False,show_time=False):
        '''\n
        Function: 使用输入数据和树进行预测，没有输入树时使用内部缓存的树
        
        Parameters
        ----------
        X: 所有特征列，DataFrame类型
        tree: 决策树，Tree类型，默认调用内部缓存的树
        return_proba: 是否返回分类概率，分类模式下有效，bool类型，默认值False
        return_paths: 是否返回决策路径，bool类型，默认值False
                     （路径信息以str类型返回，可转换为list使用）
        show_time: 是否显示耗时，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 预测的分类/分类概率，Series/DataFrame类型
        1: 所有数据最终抵达的节点和决策路径，DataFrame类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        start = time.clock()
        #校验参数
        check_type('tree',type(tree),type(Tree()))
        check_type('return_proba',type(return_proba),type(True))
        check_type('return_paths',type(return_paths),type(True))
        check_type('show_time',type(show_time),type(True))
        check_type('X',type(X),type(pd.DataFrame()))
        check_feats_match(X.columns,tree.features,'X','tree',mode='right')
        #数据集大小
        n=len(X)
        #数据流，记录已达到节点
        flow=pd.Series(np.zeros(n),index=X.index)
        #分类模式先求分类概率，回归模式直接求回归值
        if tree.mode=='Classifier':
            #定义存放分类结果的DataFrame
            p_y_=pd.DataFrame(
                    np.zeros(n*len(tree.classes)).reshape(n,len(tree.classes)),
                    index=X.index,columns=tree.classes)
        else:
            #定义存放回归值的Series
            p_y_=pd.Series(np.zeros(n),index=X.index)
        #准备访问的节点
        iloc=0
        queue=[tree.nodes[0]]
        #直到没有等待访问的节点时结束
        while len(queue)>0:
            node=queue.pop(0)
            #筛选出流至该节点的数据
            boolIdx=(flow==node.nid)
            X_=X[boolIdx]
            flow_=flow[boolIdx]
            #当前节点是叶节点，返回分类
            if node.is_leaf==True:
                if tree.mode=='Classifier':
                    p_y_.update(pd.DataFrame(node.output,columns=tree.classes,index=X_.index))
                else:
                    p_y_.update(pd.Series(node.output,index=X_.index))
            else:
                #当前节点是内节点，遍历每个子节点
                for childId in node.childs:
                    child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                    #不同限制条件采用不同方式处理，更新数据流所到达的节点
                    if child.limit=='<=':
                        flow_[X_[child.feature]<=child.value]=childId
                    elif child.limit=='>':
                        flow_[X_[child.feature]>child.value]=childId
                    elif child.limit=='in':
                        flow_[X_[child.feature].isin(child.value)]=childId
                    else:
                        flow_[X_[child.feature]==child.value]=childId
                    flow.update(flow_)
                    #将子节点放入准备访问的队列
                    queue.append(child)
        #分类模式下可以返回分类概率或唯一分类
        if (tree.mode=='Classifier')&(return_proba==False):
            p_y_=self.choose_class_(p_y_,tree.classes)
        #是否返回决策路径
        if return_paths==False:
            end = time.clock()
            if show_time==True:
                print('\ntime used for predict:%f'%(end-start))
            return p_y_
        else:
            paths=pd.DataFrame()
            paths['reach']=flow
            paths['path']=''
            reach_nodes=flow.sort_values().drop_duplicates().astype('int').tolist()
            for nodeId in reach_nodes:
                path=tree.get_path(nodeId)
                paths.loc[paths['reach']==nodeId,'path']=str(path)
            end = time.clock()
            if show_time==True:
                print('\ntime used for predict: %f'%(end-start))
            return p_y_,paths
    
    #评估
    def assess(self,y,p_y,mode=None):
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
        ----------
        
        Returns
        -------
        0: 分类->准确率，回归->R方，float类型
        -------
        '''
        if type(mode)==type(None):
            mode=self.tree.mode
        else:
            mode_list=['Classifier','Regressor']
            check_limit('mode',mode in mode_list,str(mode_list))
        #校验输入
        check_type('p_y',type(p_y),type(pd.Series()))
        check_type('y',type(y),type(pd.Series()))
        check_index_match(y,p_y,'y','p_y')
        #分类模式求准确率，回归模式求R2
        if mode=='Classifier':
            if type(y.iloc[0])!=type(''):
                #print('it is recommended to transform y as str type')
                y=y.astype('str')
            return stats.accuracy(y,p_y)
        elif mode=='Regressor':
            return stats.r_sqr(y,p_y)
            
    #误差代价err_cost_=sum(E_i)+a*leafNum
    #E_i为下属各个子节点上的误差个数或方差，leafNum为下属叶节点总数，
    #a为平衡参数，用于平衡误差和复杂度，a越大越倾向于选择简单的模型，a为0则只考虑误差
    '''
    a:平衡参数，float类型
    start_id:起始节点id，int类型
    tree:决策树，Tree类型
    after_prun:是否计算剪枝后的误差代价，bool类型
    return> 0:误差代价
    '''
    def err_cost_(self,a=0.0,start_id=0,tree=None,after_prun=False):
        #未从外部传入树时使用内部存储的树
        if type(tree)==type(None):
            tree=self.tree
        node,iloc=tree.find_node(start_id,return_iloc=True)
        if after_prun==False:
            leafNum=tree.get_leaf_num(start_id)
            e,queue=leafNum*a,[(node,iloc)]
            while len(queue)>0:
                node,iloc=queue.pop(0)
                if node.is_leaf==True:
                    e+=node.error
                else:
                    for childId in node.childs:
                        child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                        queue.append((child,iloc))
        else:
            e=a+node.error
        return e
    
    #降低错误率剪枝 Reduced-Error Pruning
    '''
    test_X:测试数据集全部特征列，DataFrame类型
    test_y:测试数据集全部分类列，Series类型
    tree:决策树，Tree类型
    return_subtrees:是否返回剪枝过程的子树序列，bool类型
    return> 0:剪枝后的决策树，Tree类型/剪枝过程的子树序列+准确率+节点id，list(Tree,int,int)类型
    '''
    def pruning_rep_(self,test_X,test_y,tree=None,return_subtrees=False):
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        #计算完全树在测试集上的准确率
        p_y=self.predict(test_X,tree,show_time=False)
        best_score=self.assess(test_y,p_y,tree.mode)
        subtrees,scores,cut_ids=[tree.copy()],[best_score],[-1]
        #从下至上遍历每个节点
        for i in range(len(tree.nodes)-1,0,-1):
            node=tree.nodes[i]
            #非叶节点才可剪枝
            if node.is_leaf==False:
                #复制树并试剪枝
                tree_=tree.copy()
                tree_.cut(node.nid)
                #计算剪枝后在测试集上的准确率
                p_y=self.predict(test_X,tree_,show_time=False)
                score=self.assess(test_y,p_y,tree.mode)
                #若准确率没有降低，保留剪枝后的树
                if score>=best_score:
                    tree,best_score=tree_,score
                    subtrees.append(tree.copy())
                    scores.append(score)
                    cut_ids.append(node.nid)
        if return_subtrees==True:
            return subtrees,scores,cut_ids
        else:
            return tree
    
    #悲观剪枝 Pessimistic Error Pruning
    #注：暂未找到pep应用于回归树的参考案例，所以暂不支持回归树剪枝
    '''
    tree:决策树，Tree类型
    return_subtrees:是否返回剪枝过程的子树序列，bool类型
    return> 0:剪枝后的决策树，Tree类型/剪枝过程的子树序列+节点id，list(Tree,int)类型
    '''    
    def pruning_pep_(self,tree=None,return_subtrees=False):
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        subtrees,cut_ids=[tree.copy()],[-1]
        #自上而下遍历每个节点
        for i in range(1,len(tree.nodes)):
            if i>=len(tree.nodes):
                break
            node=tree.nodes[i]
            if node.is_leaf==False:
                #计算剪枝前的误差均值和标准差
                E=self.err_cost_(a=0.5,start_id=node.nid,tree=tree)
                S_E=np.sqrt(E*(node.sample_n-E)/node.sample_n)
                #计算剪枝后的误差均值
                E_=self.err_cost_(a=0.5,start_id=node.nid,tree=tree,after_prun=True)
                #剪枝后的误差均值变化在一个标准差内
                if E_<=E+S_E:
                    tree.cut(node.nid)
                    subtrees.append(tree.copy())
                    cut_ids.append(node.nid)
        if return_subtrees==True:
            return subtrees,cut_ids
        else:
            return tree
        
    #代价复杂度剪枝 Cost-Complexity Pruning
    '''
    test_X:测试数据集全部特征列，DataFrame类型
    test_y:测试数据集全部分类列，Series类型
    tree:决策树，Tree类型
    return_subtrees:是否返回剪枝过程的子树序列，bool类型
    return> 0:剪枝后的决策树，Tree类型/剪枝过程的子树序列+准确率+节点id，list(Tree,int,int)类型
    '''        
    def pruning_ccp_(self,test_X,test_y,tree=None,return_subtrees=False):
        if type(tree)==type(None):
            tree=self.tree.copy()
        else:
            tree=tree.copy()
        #计算完全树在测试集上的准确率
        p_y=self.predict(test_X,tree,show_time=False)
        best_score=self.assess(test_y,p_y,tree.mode)
        best_tree=tree.copy()
        subtrees,scores,cut_ids=[tree.copy()],[best_score],[-1]
        #从完全树开始迭代剪枝至只剩根节点
        while len(tree.nodes)>1:
            a_min,best_cut,leaf_reduce=np.inf,0,0
            #每次迭代选择表面误差增益值最小的节点剪枝,将剪枝后的子树用于下一次迭代
            for i in range(0,len(tree.nodes)):
                node=tree.nodes[i]
                if node.is_leaf==False:
                    leafNum=tree.get_leaf_num(node.nid)
                    #此处采用最简单的误差代价，也可以使用sum(sample_n*gini or entropy)
                    cost=self.err_cost_(start_id=node.nid,tree=tree)
                    cost_=self.err_cost_(start_id=node.nid,tree=tree,after_prun=True)
                    #表面误差率增益值a=误差代价增加量/叶节点数减少量
                    #注：分子要小(误差增加少)，分母要大(复杂度减少多)，即a要最小
                    a=(cost_-cost)/(leafNum-1)
                    if a<a_min:
                        a_min,best_cut,leaf_reduce=a,node.nid,leafNum-1
                    elif a==a_min:
                        if leafNum-1>leaf_reduce:
                            best_cut,leaf_reduce=node.nid,leafNum-1
            #当前迭代最优子树
            tree.cut(best_cut)
            p_y=self.predict(test_X,tree,show_time=False)
            score=self.assess(test_y,p_y,tree.mode)
            subtrees.append(tree.copy())
            scores.append(score)
            cut_ids.append(best_cut)
            #选择在测试集上表现最好的子树
            if score>=best_score:
                best_score=score
                best_tree=tree.copy()
        if return_subtrees==True:
            return subtrees,scores,cut_ids
        else:
            return best_tree
                
    #剪枝
    def pruning(self,test_X=None,test_y=None,tree=None,mode='ccp',return_tree=False,show_time=True):
        '''\n
        Function: 对输入树进行剪枝
        
        Note: 部分算法需要输入测试数据
        
        Parameters
        ----------
        test_X: 测试数据集全部特征列，DataFrame类型，默认值None
        test_y: 测试数据集全部分类列，Series类型，默认值None
        tree: 决策树，Tree类型，默认调用内部缓存的树
        mode: 模式，str类型，默认值'ccp'，
              'rep'->降低错误率剪枝
              'pep'->悲观剪枝
              'ccp'->代价复杂度剪枝
        return_tree: 是否直接返回树而不替换内部缓存的树，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 剪枝后的决策树，Tree类型
        -------
        '''
        start = time.clock()
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        #参数校验
        mode_list=['rep','pep','ccp']
        check_type('mode',type(mode),type(''))
        mode=mode.lower()
        check_limit('mode',mode in mode_list,str(mode_list))
        if mode in ['rep','ccp']:
            #校验输入
            check_type('test_X',type(test_X),type(pd.DataFrame()))
            check_type('test_y',type(test_y),type(pd.Series()))
            check_index_match(test_X,test_y,'test_X','test_y')
            check_feats_match(test_X.columns,tree.features,'test_X','tree',mode='right')
        #剪枝
        if mode=='rep':
            tree_=self.pruning_rep_(test_X,test_y,tree)
        elif mode=='pep':
            tree_=self.pruning_pep_(tree)
        elif mode=='ccp':
            tree_=self.pruning_ccp_(test_X,test_y,tree)
        end = time.clock()
        if show_time==True:
            print('\ntime used for pruning:%f'%(end-start))
        #是否以返回值形式给到剪枝后的树    
        if return_tree==False:
            self.tree=tree_
        else:
            return tree_
    
    #打印结点信息
    def print_nodes(self,tree=None):
        '''\n
        Function: 打印结点信息
                
        Parameters
        ----------
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Print
        -------
        0: 结点信息
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        print('\n[Nodes Info]')
        tree.print_nodes()
    
    #保存树
    def save_tree(self,file_path,tree=None):
        '''\n
        Function: 保存树

        Parameters
        ----------
        tree: 决策树，Tree类型
        file_path: 保存文件的路径，str类型
        ----------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('file_path',type(file_path),type(''))
        tree.to_dataframe().to_csv(file_path,encoding='utf-8',index=False)
    
    #读取树
    def read_tree(self,file_path,output=False):
        '''\n
        Function: 读取树结构
                
        Parameters
        ----------
        file_path: 文件的路径，str类型
        output: 是否返回读取到的树，bool类型，默认值False
        ----------
        
        Returns
        -------
        0: 读取到的树，Tree类型
        -------
        '''
        check_type('file_path',type(file_path),type(''))
        df=pd.read_csv(file_path,encoding='utf-8')
        if output==True:
            return Tree(df)
        else:
            self.tree=Tree(df)
        
    #计算树的叶节点数
    def get_leaf_num(self,start_id=0,tree=None):
        '''\n
        Function: 计算树的叶节点数
                
        Parameters
        ----------
        start_id: 起始的节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Returns
        -------
        0: 叶节点数量，int类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        leafNum=tree.get_leaf_num(start_id)
        return leafNum
    
    #计算树的深度
    def get_depth(self,start_id=0,tree=None):
        '''\n
        Function: 计算树的深度
                
        Parameters
        ----------
        start_id: 起始的节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Returns
        -------
        0: 树的深度，int类型
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        depth_max=tree.get_depth(start_id)
        return depth_max
    
    #注：可视化用于展示复杂的树会看不清
    #定义可视化格式
    style_inNode = dict(boxstyle="round4", color='#3366FF')  # 定义中间判断结点形态
    style_leafNode = dict(boxstyle="circle", color='#FF6633')  # 定义叶结点形态
    style_arrow_args = dict(arrowstyle="<-", color='g')  # 定义箭头
        
    #选择概率最高的类作为叶节点判定的类
    '''
    proba_dict:分类概率，dict类型
    return> 0:选择的类，str类型
    '''
    def choose_class__(self,proba_dict):
        class_,proba_max='',0.0
        for key in proba_dict.keys():
            if proba_dict[key]>proba_max:
                proba_max=proba_dict[key]
                class_=key
        return class_
    
    #绘制带箭头的注释
    '''
    node_text:节点上的文字，str类型
    location:中心点坐标，tuple(float)类型
    p_location:父节点坐标，tuple(float)类型
    node_type:节点类型，预定义的dict类型
    first:根节点标志位，bool类型
    '''
    def plot_node_(self,node_text,location,p_location,node_type,first):
        if first==True:
            self.ax1.annotate(node_text,xy=p_location,xycoords='axes fraction',
                     xytext=location,textcoords='axes fraction',
                     va="center",ha="center",bbox=node_type)
        else:
            self.ax1.annotate(node_text,xy=p_location, xycoords='axes fraction',
                     xytext=location,textcoords='axes fraction',
                     va="center",ha="center",bbox=node_type, 
                     arrowprops=self.style_arrow_args)
    
    #在父子结点间填充文本信息
    '''
    location:中心点坐标，tuple(float)类型
    p_location:父节点坐标，tuple(float)类型
    text:文本，str类型
    '''
    def plot_mid_text_(self,location,p_location,text):
        xMid=(p_location[0]-location[0])/2.0+location[0]
        yMid=(p_location[1]-location[1])/2.0+location[1]
        self.ax1.text(xMid,yMid,text,va="center",ha="center",rotation=30)
    
    #绘制树_遍历节点
    '''
    start_id:开始节点，int类型
    print_loc：打印节点位置信息，bool类型
    tree:决策树，Tree类型
    '''
    def plot_(self,start_id,tree):
        node,iloc=tree.find_node(start_id,return_iloc=True)
        #总宽度，总高度，x偏移，y偏移，当前深度
        totalW=float(self.get_leaf_num(start_id,tree=tree))-1
        totalD=float(self.get_depth(start_id,tree=tree))+0.1
        if totalW==0:
            if tree.mode=='Classifier':
                output=self.choose_class__(node.output)
            else:
                output=str(node.output)
            self.plot_node_(output,(0.5,0.5),(0.5,0.5),self.style_leafNode,True)
            return
        xOff,yOff=-1.0/totalW,1.0
        depth=tree.nodes[start_id].depth
        pLocation_=(xOff,yOff)
        #队列：待处理的节点，父节点坐标，父节点下属叶节点数
        queue=[(node,(xOff,yOff),0)]
        while len(queue)>0:
            node,pLocation,pLeafNum=queue.pop(0)
            #获取当前节点下属叶节点数
            leafNum=self.get_leaf_num(node.nid,tree=tree)
            #绘制方式是逐层绘制，深度变化时调整y偏移，父节点变化时根据父节点调整x偏移
            if node.depth>depth:
                depth=node.depth
                yOff=yOff-1.0/totalD
            if pLocation!=pLocation_:
                pLocation_=pLocation
                xOff=pLocation[0]-(1.0+float(pLeafNum))/2.0/totalW
            #首个节点不需要绘制箭头
            if node.nid==start_id:
                first,mid_text=True,''
            else:
                first=False
                mid_text=node.limit+' '+str(node.value)
            #叶节点/内结点绘制
            if node.is_leaf==True:
                #调整x偏移（每个叶节点的偏移量统一）
                xOff=xOff+1.0/totalW
                #选择概率最大的分类/显示回归值
                if tree.mode=='Classifier':
                    output=self.choose_class__(node.output)
                else:
                    int_len=len(str(int(node.output)))
                    dec_len=3-int_len
                    if dec_len<0:
                        dec_len=0
                    output=str(round(node.output,dec_len))
                #绘制当前节点和指向其的箭头
                self.plot_node_(output,(xOff, yOff), 
                                pLocation,self.style_leafNode,False)
                #显示箭头上的文字
                self.plot_mid_text_((xOff,yOff),pLocation,mid_text)
                #打印节点坐标
                '''
                print('<leafNode id=%d depth=%d text=%s> x:%f y:%f'%
                      (node.nid,depth,class_,xOff,yOff))
                '''
            else:
                #内节点显示分裂特征名
                bestFeature=node.childs_feature
                #调整x偏移（每个内结点偏移量由下属叶节点数决定）
                xOff=xOff+(1.0+float(leafNum))/2.0/totalW
                location=(xOff,yOff)
                #绘制节点和箭头
                self.plot_node_(bestFeature,location,pLocation, 
                                self.style_inNode,first)
                self.plot_mid_text_(location,pLocation,mid_text)
                #打印节点坐标
                '''
                print('<inNode id=%d depth=%d text=%s> x:%f y:%f'%
                      (node.nid,depth,bestFeature,xOff,yOff))
                '''
                #再次调整x偏移（上一次偏移得到的是中心位置）
                xOff=xOff+((1.0+float(leafNum))/2.0-1.0)/totalW
                #子结点加入队列
                for childId in node.childs: 
                    child,iloc=tree.find_node(childId,start=iloc+1,return_iloc=True)
                    queue.append((child,location,leafNum))
    
    #绘制树
    def plot(self,start_id=0,tree=None):
        '''\n
        Function: 绘制树
   
        Parameters
        ----------
        start_id: 开始节点id，int类型，默认值0
        tree: 决策树，Tree类型，默认调用内部缓存的树
        ----------
        
        Print
        -------
        基于matplotlib的决策树绘图
        -------
        '''
        if type(tree)==type(None):
            tree=self.tree
        else:
            check_type('tree',type(tree),type(Tree()))
        check_type('start_id',type(start_id),type(0))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig=plt.figure(1,facecolor='white')
        fig.clf()
        plt.suptitle("[Decision Tree Plot]")
        axprops=dict(xticks=[], yticks=[])
        self.ax1=plt.subplot(111,frameon=False,**axprops)
        self.plot_(start_id,tree)
        plt.show()  