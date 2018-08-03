# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numba as nb

#拆分训练集和测试集
def split_train_test(data,frac=0.8,random_state=None):
    '''
    data: 数据集，DataFrame类型或Series类型
    frac: 训练集比例，float类型(0.0~1.0)
    random_state: 随机种子
    return
    0: 训练集X，DataFrame类型  1: 训练集y，Series类型
    2: 测试集X，DataFrame类型  3: 测试集y，Series类型
    '''
    train=data.sample(frac=frac,random_state=random_state)
    test=data[~data.index.isin(train.index)]
    train_X,train_y=train.iloc[:,:-1],train.iloc[:,-1]
    test_X,test_y=test.iloc[:,:-1],test.iloc[:,-1]
    return train_X,train_y,test_X,test_y

#虚拟变量生成（也称one-hot编码）
#用于处理离散变量
def dummy_var(s):
    '''
    s: 数据集，Series类型
    return
    0: 处理后的数据集，DataFrame类型
    '''
    values=s.drop_duplicates().sort_values().tolist()
    result=np.zeros((len(s),len(values))).astype('int')
    for i in range(len(values)):
        result[s==values[i],i]=1
    return pd.DataFrame(result,columns=values,index=s.index)

#常数列补齐
#X首列填充1,即模型常数位忽略X的影响
def fill_x0(X):
    '''
    X: 特征数据集，DataFrame类型
    return
    0: 处理后的特征数据集，DataFrame类型
    '''
    #首位填充1，对应常量theta0
    if 'x0' not in X.columns.values:
        X_=X.copy()
        X_.insert(0,'x0',np.ones(len(X)))
        return X_
    else:
        print('x0 column already exists')
        return X
    
#获取输入数据集的连续性默认判定
def get_continuity(data,name):
    '''
    data: 数据集，DataFrame类型或Series类型
    name: 数据集名称，str类型
    return
    0: 连续性判定，list(bool)或bool类型
    '''
    if type(data)==type(pd.Series()):
        data=data.to_frame()
    type_list=['int64','float64','bool','category','object']
    continuity=[]
    for dtype in data.dtypes:
        if str(dtype) in ['int64','float64']:
            continuity.append(True)
        elif str(dtype) in ['bool','category','object']:
            continuity.append(False)
        else:
            raise TypeError('wrong dtypes of %s\nunsupported -> %s\nrequired -> %s'
                            %(name,str(dtype),str(type_list)))
    if len(data.columns)==1:
        return continuity[0]
    else:
        return continuity
            
#将离散标签转化为数值型索引
# 注：离散标签会统一转换为str识别
def label_to_index(data,cols):
    '''
    data: 数据集，Series或DataFrame类型
    cols: 需要转换的列，list(str)类型
    return
    0: 转化后的X，Series或DataFrame类型 
    1: 映射关系，DataFrame类型
    '''
    if type(data)==type(pd.Series()):
        data=data.to_frame()
    data=data.copy()
    mapping_list=[]
    for col in cols:
        values=data.loc[:,col].sort_values().drop_duplicates().astype('str')
        mapping_dict={label:idx for idx,label in enumerate(values)}
        data.loc[:,col]=data.loc[:,col].astype('str').map(mapping_dict)
        mapping_list+=[[col,label,idx] for idx,label in enumerate(values)]
    mapping=pd.DataFrame(mapping_list,columns=['column','label','labelIdx'])
    if len(data.columns)==1:
        return data.iloc[:,0],mapping
    else:
        return data,mapping
    
#转换回原来的标签,单个值
def index_to_label_(column,label_idx,mapping):
    '''
    column: 列名，str类型
    label_idx: 索引，int类型
    mapping: 映射关系，DataFrame类型
    return
    0:原标签，str类型
    '''
    #查找值，成功返回原标签，失败返回输入值
    if type(mapping)==type(None):
        return label_idx
    boolIdx=(mapping['column']==column)&\
            (mapping['labelIdx']==label_idx)
    if boolIdx.any()==True:
        return mapping['label'][boolIdx].values[0]
    else:
        return label_idx

#转换回原来的标签,数据集  
def index_to_label(data,mapping):
    '''
    data: 数据集，Series或DataFrame类型
    mapping: 映射关系，DataFrame类型
    return
    0: 转换后的数据集，Series或DataFrame类型
    ''' 
    if type(mapping)==type(None):
        return data
    if type(data)==type(pd.Series()):
        data=data.to_frame()
    columns_=mapping.loc[:,'column'].drop_duplicates().tolist()
    for column in data.columns:
        if column not in columns_:
            continue
        s=data.loc[:,column]
        values=s.sort_values().drop_duplicates().tolist()
        for value in values:
            data.loc[s==value,column]=index_to_label_(column,value,mapping)
    if len(data.columns)==1:
        return data.iloc[:,0]
    else:
        return data         

#特征离散化(等距分)
        
#计算参考标准
def discret_reference(X,n,cols=None):
    '''
    X: 特征数据集，Series或DataFrame类型
    n: 区间数量，int类型
    cols: 需要操作的列，list或Series类型
    return
    0: 划分区间，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    if cols==None:
        cols=X.columns
    drange=[]
    for col in cols:
        x=X.loc[:,col]
        drange.append(np.linspace(x.min(),x.max(),n+1))
    drange=pd.DataFrame(drange,index=cols)
    return drange.T
    
#根据参考标准离散化 
#注：离散化标签为str类型
def discret(X,drange,return_label=True,open_bounds=True):
    '''
    X: 特征数据集，Series或DataFrame类型
    drange: 划分区间，DataFrame类型
    return_label: 是否返回区间标签，bool类型，False时返回区间索引，默认True
    open_bounds: 开放边界，bool类型，True时超出整体范围的数据将划分在边缘区间，默认True
    return
    0: 处理后的数据集，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    result=X.copy()
    for col in drange.columns:
        x=X.loc[:,col]
        rg=drange.loc[:,col].tolist()
        if return_label==True:
            temp=pd.cut(x,bins=rg).astype('str')
            rg_edge=pd.cut([rg[1],rg[-1]],bins=rg)
            rg_min=str(rg_edge.min())
            rg_max=str(rg_edge.max())
        else:
            temp=pd.cut(x,bins=rg,labels=False)
            rg_min,rg_max=0,len(rg)-2
        if open_bounds==True:
            temp[x<=rg[0]]=rg_min
            temp[x>rg[-1]]=rg_max
        else:
            temp[x==rg[0]]=rg_min
        if return_label==False:
            temp=temp.astype('int')
        result.loc[:,col]=temp
    return result
    
#特征缩放
#注：缩放后求得的theta会不一样，预测数据时也需要进行缩放 
    
#获取缩放的参照标准，一般输入训练集X
def scaler_reference(X):
    '''
    X: 特征数据集，Series或DataFrame类型
    return
    0: 参照标准，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    ref=pd.DataFrame()
    ref['min']=X.min()
    ref['max']=X.max()
    ref['mean']=X.mean()
    ref['std']=X.std()
    return ref
    
#两种不同的缩放方式
def minmax_scaler(X,ref):
    '''
    X: 特征数据集，Series或DataFrame类型
    ref: 参照标准，DataFrame类型
    return
    0: 处理后的数据集，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    return (X-ref['min'])/(ref['max']-ref['min'])  
    
def standard_scaler(X,ref):
    '''
    X: 特征数据集，Series或DataFrame类型
    ref: 参照标准，DataFrame类型
    return
    0: 处理后的数据集，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    return (X-ref['mean'])/ref['std']
    
#特征映射（多项式）
#注：配合正则化使用，不然容易出现过拟合
def feature_mapping(X,h,cross=False):
    '''
    X: 特征数据集，Series或DataFrame类型
    h: 多项式的最高次数,int类型
    cross: 是否添加组合项（目前只有两两组合），bool类型，默认False
    return
    0: 处理后的数据集，DataFrame类型
    '''
    if type(X)==type(pd.Series()):
        X=X.to_frame()
    X_h=X.copy()
    for i in range(h):
        if i==0:
            continue
        new_X=X**(i+1)
        new_X.columns=new_X.columns+'^%d'%(i+1)
        X_h=X_h.join(new_X,how='inner')
        if cross==True:
            cfg=[]
            for m in range(len(X.columns)-1):
                for n in range(len(X.columns)-1-m):
                    cfg.append((m,m+n+1))
            for c in cfg:
                for j in range(i):
                    x1=X.iloc[:,c[0]]
                    x2=X.iloc[:,c[1]]
                    new_x=(x1**(j+1))*(x2**(i-j))
                    new_x.name=x1.name+'^%d'%(j+1)+'_'+x2.name+'^%d'%(i-j)
                    X_h=X_h.join(new_x,how='inner')
    return X_h


#使用numba加速运算，第一次运行时需要一些时间编译，
#且只能接收Numpy数组，对于pandas的数据对象可通过values属性获取

#去重统计
@nb.jit(nopython=True,cache=True)
def unique_count(array):
    '''
    array:需要去重的数据列，narray类型
    return
    0:去重后的数据列，narray类型
    1:数量统计，narray类型
    '''
    #排序
    array_=np.sort(array)
    #初始化值变量：取值列表/当前取值/计数列表/当前计数
    values,value=[array_[0]],array_[0]
    counts,count=[],0
    #遍历数据列，对每个取值计数
    for i in range(len(array_)):
        if array_[i]==value:
            count+=1
        else:
            values.append(array_[i])
            counts.append(count)
            value,count=array_[i],1
    counts.append(count)
    return np.array(values),np.array(counts)

#权重分类求和
@nb.jit(nopython=True,cache=True)
def weight_sum(array,weight):
    '''
    array:需要统计的数据列，narray类型
    weight:权重的数据列，narray类型
    return
    0:去重后的数据列，narray类型
    1:数量统计，narray类型
    '''
    #排序
    sort_idx=np.argsort(array)
    array_=array[sort_idx]
    weight_=weight[sort_idx]
    #初始化值变量：取值列表/当前取值/计数列表/当前计数
    values,value=[array_[0]],array_[0]
    weights,weight=[],0
    #遍历数据列，对每个取值计数
    for i in range(len(array_)):
        if array_[i]==value:
            weight+=weight_[i]
        else:
            values.append(array_[i])
            weights.append(weight)
            value,weight=array_[i],weight_[i]
    weights.append(weight)
    return np.array(values),np.array(weights)

#列表元素查找
@nb.jit(nopython=True,cache=True)
def isin(array,values):
    '''
    array:需要查找的数据列，narray类型
    values:查找的元素，list类型
    return
    0:布尔索引，narray类型
    '''
    #数据列长度，取值列表长度
    n,m=len(array),len(values)
    #用于过滤结果的布尔索引
    filterIdx=np.array([False for i in range(n)])
    #遍历数据列，将匹配到的行的布尔索引更改
    for i in range(n):
        for j in range(m):
            if array[i]==values[j]:
                filterIdx[i]=True
    return filterIdx

#组合枚举(numba的nopython不支持递归和嵌套list）
#注：剔除了空集和全集
@nb.jit(nopython=True,cache=True)
def combine_enum(values,split_mode=False,e_min=0,e_max=0): 
    '''
    values:离散取值，list类型
    split_mode:分割模式，求离散取值分割到两个子集中的所有组合，
               只返回一个子集的结果，bool类型
    e_min:元素数下限，int类型
    e_min:元素数上限，int类型,0表示不限制
                以上两个参数不影响组合枚举的过程，不会带来效率的提升
    return
    0:组合的结果，二维narray类型，01表示是否取，
      行对应每种组合，列对应各个value，narray类型
    '''
    #取值列表长度
    vl_count=len(values)
    #取值少于2没有除空集和全集以外的组合
    if vl_count<=1:
        return np.zeros((0,0))
    #元素数范围错误
    if ((e_max>0)&(e_max<e_min))|(e_max<0):
        return np.zeros((0,0))
    #将每个取值是否取用以01标志位表示，每一种组合为一行
    #结果矩阵行数对应组合种数，等于2的vl_count次方-2，列数等于vl_count
    #二分裂模式只需要考虑前面一半的组合，后半部分与前面是对称的
    if split_mode==True:
        cb_count=2**(vl_count-1)-1
    else:
        cb_count=2**vl_count-2
    result=np.zeros((cb_count,vl_count))
    #用二进制数遍历的方式获得全部01组合
    for i in range(1,cb_count+1):
        byte_size,j=0,i
        while j>1:
            if j%2==1:
                result[i-1][byte_size]=1
            j=j//2
            byte_size+=1
        if j==1:
            result[i-1][byte_size]=1
    #返回元素数小于限制的组合
    if e_max==0:
        return result
    elif e_max==e_min:
        return result[result.sum(axis=1)==e_max]
    else:
        e_counts=result.sum(axis=1)
        return result[(e_counts>=e_min)&
                      (e_counts<=e_max)]

#组合抽取
@nb.jit(nopython=True,cache=True)
def combine_take(values,take_array):
    '''
    values:离散取值，list类型
    take_array:取值标识，narray类型，可对上面一个方法的返回值进行切片得到
    return
    0:组合值列表，list类型
    '''
    combine=[]
    for i in range(len(take_array)):
        if take_array[i]==1:
            combine.append(values[i])
    return combine

#获取组合
def combines(values,split_mode=False,e_min=0,e_max=0):
    '''
    values:离散取值，list类型
    split_mode:分割模式，求离散取值分割到两个子集中的所有组合，
               只返回一个子集的结果，bool类型
    e_min:元素数下限，int类型
    e_min:元素数上限，int类型,0表示不限制
    return
    0:所有符合要求的组合，list(list)类型
    '''
    combine_enum_=combine_enum(values,split_mode,e_min,e_max)
    combines_=[]
    for i in range(len(combine_enum_)):
        take_array=combine_enum_[i,:]
        combines_.append(combine_take(values,take_array))
    return combines_

#列举正负样本组合
def combine_enum_paired(values,symmetry=False):
    '''
    values: 离散取值，list类型
    symmetry: 是否获取对称组合，即正负样本选取颠倒的组合，bool类型，默认False
    return
    0: 正样本选取矩阵
    1: 负样本选取矩阵
       均为二维narray类型，01表示是否取，
       行对应每种组合，列对应各个value
    '''
    #取值数量
    vl_count=len(values)
    if vl_count<2:
        raise ValueError('the length of values must be >=2')
    if symmetry==False:
        cb_count=int(vl_count*(vl_count-1)/2)
    else:
        cb_count=int(vl_count*(vl_count-1))
    #正负样本选取矩阵
    result_p=np.zeros((cb_count,vl_count))
    result_n=np.zeros((cb_count,vl_count))
    #遍历每一种组合
    if symmetry==False:
        for i in range(vl_count-1):
            for j in range(i+1,vl_count):
                combine_idx=int(i*(2*vl_count-i-1)/2+j-i-1)
                result_p[combine_idx][i]=1
                result_n[combine_idx][j]=1
    else:
        for i in range(vl_count):
            for j in range(vl_count):
                if j<i:
                    combine_idx=int(i*(vl_count-1)+j)
                elif j>i:
                    combine_idx=int(i*(vl_count-1)+j-1)
                else:
                    continue
                result_p[combine_idx][i]=1
                result_n[combine_idx][j]=1
    return result_p,result_n

#获取正负样本组合
def combines_paired(values,symmetry=False,merge_output=True):
    '''
    values: 离散取值，list类型
    symmetry: 是否获取对称组合，即正负样本选取颠倒的组合，bool类型，默认False
    return
    0: 正样本选用
    1: 负样本选用
       均为list类型，01表示是否取，
       行对应每种组合，列对应各个value
    '''
    combine_enum_p_,combine_enum_n_=combine_enum_paired(values,symmetry)
    if merge_output==False:
        combines_p_,combines_n_=[],[]
        for i in range(len(combine_enum_p_)):
            take_array_p,take_array_n=combine_enum_p_[i,:],combine_enum_n_[i,:]
            for j in range(len(take_array_p)):
                if take_array_p[j]==1:
                    combines_p_.append(values[j])
                if take_array_n[j]==1:
                    combines_n_.append(values[j])
        return combines_p_,combines_n_
    else:
        combines_=[]
        for i in range(len(combine_enum_p_)):
            take_array_p,take_array_n=combine_enum_p_[i,:],combine_enum_n_[i,:]
            for j in range(len(take_array_p)):
                if take_array_p[j]==1:
                    combine_p_=values[j]
                if take_array_n[j]==1:
                    combine_n_=values[j]
            combines_.append([combine_p_,combine_n_])
        return combines_