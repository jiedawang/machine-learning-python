# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_lib.data_prep as dp
import user_lib.deci_tree as dt
from datetime import datetime
from sklearn import tree
import time

#简单数据集测试
dataValues = [['youth', 'no', 'no', '1', 'refuse'],
           ['youth', 'no', 'no', '2', 'refuse'],
           ['youth', 'yes', 'no', '2', 'agree'],
           ['youth', 'yes', 'yes', '1', 'agree'],
           ['youth', 'no', 'no', '1', 'refuse'],
           ['mid', 'no', 'no', '1', 'refuse'],
           ['mid', 'no', 'no', '2', 'refuse'],
           ['mid', 'yes', 'yes', '2', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['mid', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '3', 'agree'],
           ['elder', 'no', 'yes', '2', 'agree'],
           ['elder', 'yes', 'no', '2', 'agree'],
           ['elder', 'yes', 'no', '3', 'agree'],
           ['elder', 'no', 'no', '1', 'refuse']]

labels = ['age', 'working?', 'house?', 'credit_situation','decition']
data=pd.DataFrame(dataValues,columns=labels)

X=data.iloc[:,:(len(data.columns)-1)]
y=data.iloc[:,len(data.columns)-1]

dtModel0=dt.DecisionTree()
deciTree0=dtModel0.fit(X,y,model_type='id3',output=True)

dtModel0.plot()
dtModel0.print_nodes()

test=data.drop('decition',axis=1)
result=dtModel0.predict(test)

#小麦种子数据集(离散输出)
f = open('D:\\Training Data\\used\\wheat_seed.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['区域','周长','压实度','籽粒长度','籽粒宽度','不对称系数','籽粒腹沟长度','类']
describe=buf.describe()
dp_tool=dp.DataPreprocessing()
#分割训练集和测试集
X1,y1,test_X1,test_y1=dp_tool.split_train_test(buf)
X1=X1.round(4)
#生成离散化区间
rg=dp_tool.discret_reference(X1,3)
#进行特征离散化
X1_=dp_tool.discret(X1,rg,str_label=True)
test_X1_=dp_tool.discret(test_X1,rg,str_label=True)

#ID3(没有处理连续特征的能力，需要预处理)
#训练决策树
dtModel1=dt.DecisionTree()
deciTree1=dtModel1.fit(X1_,y1,model_type='id3',output=True)
dtModel1.plot()
#dtModel1.plot(start_id=4,print_loc=True)
#dtModel1.print_nodes()
#预测
result1=dtModel1.predict(test_X1_)
score1=dtModel1.assess(test_y1,result1)
print('\nID3 test score: %f'%score1)

#存储和读取测试
dtModel1.save_tree('D:\\Model\\deciTree.txt')
dtModel1.read_tree('D:\\Model\\deciTree.txt')

#C4.5
#连续特征
#训练决策树
dtModel2=dt.DecisionTree()
deciTree2=dtModel2.fit(X1,y1,model_type='c4.5',output=True)
dtModel2.plot()
#dtModel2.print_nodes()
#预测(顺便测试一下获取决策路径)
result2=dtModel2.predict(X1)
score2=dtModel2.assess(y1,result2)
print('\nC4.5 train score: %f'%score2)
result2,paths=dtModel2.predict(test_X1,return_paths=True)
score2=dtModel2.assess(test_y1,result2)
print('\nC4.5 test score: %f'%score2)

#离散特征
#训练决策树
dtModel2_2=dt.DecisionTree()
deciTree2_2=dtModel2_2.fit(X1_,y1,model_type='c4.5',output=True)
dtModel2_2.plot()
#dtModel2_2.print_nodes()
#预测
result2_2=dtModel2_2.predict(test_X1_)
score2_2=dtModel2_2.assess(test_y1,result2_2)
print('\nC4.5 test score: %f'%score2_2)

#剪枝测试(pep)
#注：pep剪枝似乎挺容易失败的
tree2_pruned=dtModel2.pruning(mode='pep',return_tree=True)
dtModel2.plot(tree=tree2_pruned)
result2_pruned=dtModel2.predict(X1,tree=tree2_pruned)
score2_pruned=dtModel2.assess(y1,result2_pruned)
print('\nC4.5 train score after pruning: %f'%score2_pruned)
result2_pruned=dtModel2.predict(test_X1,tree=tree2_pruned)
score2_pruned=dtModel2.assess(test_y1,result2_pruned)
print('\nC4.5 test score after pruning: %f'%score2_pruned)

#单行数据的决策路径
#dtModel2.decition_path(X1.iloc[0,:])

#CART分类
#连续特征
#训练决策树
dtModel3=dt.DecisionTree()
deciTree3=dtModel3.fit(X1,y1,model_type='cart_c',output=True)
dtModel3.plot()
#dtModel3.print_nodes()
#预测
result3=dtModel3.predict(X1,time_cost=False)
score3=dtModel3.assess(y1,result3)
print('\nCART(Classifier) train score: %f'%score3)
result3=dtModel3.predict(test_X1,time_cost=False)
score3=dtModel3.assess(test_y1,result3)
print('\nCART(Classifier) test score: %f'%score3)

#离散特征
#训练决策树
dtModel3_2=dt.DecisionTree()
deciTree3_2=dtModel3_2.fit(X1_,y1,model_type='cart_c',output=True)
dtModel3_2.plot()
#dtModel3_2.print_nodes()
#预测
result3_2=dtModel3_2.predict(test_X1_)
score3_2=dtModel3_2.assess(test_y1,result3_2)
print('\nCART test score: %f'%score3_2)

#剪枝测试(ccp)
tree3_pruned=dtModel3.pruning(test_X1,test_y1,mode='ccp',return_tree=True)
dtModel3.plot(tree=tree3_pruned)
result3_pruned=dtModel3.predict(X1,tree=tree3_pruned,time_cost=False)
score3_pruned=dtModel3.assess(y1,result3_pruned)
print('\nCART(Classifier) train score after pruning: %f'%score3_pruned)
result3_pruned=dtModel3.predict(test_X1,tree=tree3_pruned,time_cost=False)
score3_pruned=dtModel3.assess(test_y1,result3_pruned)
print('\nCART(Classifier) test score after pruning: %f'%score3_pruned)

#dtModel3.decition_path(X1.iloc[0,:])

#分类决策面可视化
#基本参数
classes=y1.sort_values().drop_duplicates().tolist()
n_classes=len(classes)
plot_colors="bry"
plot_step=0.02

#获取所有特征的两两组合
column_idx=np.linspace(0,len(X1.columns)-1,len(X1.columns)).astype('int')
pair_enum=dt.get_combines(column_idx,e_min=2,e_max=2)

for pairidx,pair in enumerate(pair_enum):
    #if pair!=[0,1]:
        #continue
    #每次只用两个特征训练
    X1_2=X1.iloc[:,pair]
    dtModel4=dt.DecisionTree()
    dtModel4.fit(X1_2,y1,model_type='cart_c',time_cost=False)
    #dtModel4.plot()
    #绘制决策边界
    x_min, x_max = X1_2.iloc[:, 0].min() - 1, X1_2.iloc[:, 0].max() + 1
    y_min, y_max = X1_2.iloc[:, 1].min() - 1, X1_2.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    xy_=pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns=X1_2.columns)
    Z=dtModel4.predict(xy_,time_cost=False)
    Z=Z.values.reshape(xx.shape)
    ctf=plt.contourf(xx,yy,Z,cmap=plt.cm.Paired) 
    ct=plt.contour(xx,yy,Z,colors='black',linewidths=0.2)
    #绘制训练集散点图
    for i,color in zip(range(n_classes),plot_colors):
        boolIdx=(y1==classes[i])
        plt.scatter(X1_2[boolIdx].iloc[:,0],X1_2[boolIdx].iloc[:,1],
                    edgecolors='black',c=color,
                    label=classes[i],cmap=plt.cm.Paired)
    #其他配置
    plt.xlabel(X1_2.columns[0])
    plt.ylabel(X1_2.columns[1])
    plt.axis("tight")
    plt.suptitle("Decision surface using paired features %s"%str(X1_2.columns.values))
    plt.legend()
    plt.show()

#与sklearn对照
#criterion:默认为"gini"
#支持的标准有:
# "gini"代表的是Gini impurity(不纯度)
# "entropy"代表的是information gain（信息增益）
start = time.clock()
sk_dtModel = tree.DecisionTreeClassifier()
sk_dtModel.fit(X1, y1)
end = time.clock()
print('\ntime used for trainning:%f'%(end-start))
print('\nsklearn train score:%f'%sk_dtModel.score(X1,y1))

#保存模型
from sklearn.externals import joblib
joblib.dump(sk_dtModel, 'D:\\Model\\deciTree2.pkl')
sk_dtModel = joblib.load('D:\\Model\\deciTree2.pkl') 

sk_result = sk_dtModel.predict(test_X1)
print('\nsklearn test score:%f'%sk_dtModel.score(test_X1,test_y1))

#波士顿房价数据(连续输出)
f = open('D:\\Training Data\\used\\boston_house_price.txt')
buf = pd.read_table(f,header=None,delim_whitespace=True)
buf.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD',
                     'TAX','PTRATIO','B','LSTAT','MEDV']
dp_tool=dp.DataPreprocessing()
#分割训练集和测试集
X2,y2,test_X2,test_y2=dp_tool.split_train_test(buf)
X2=X2.round(4)

#cart回归树
#注：回归树的拟合和剪枝在性能上表现得比较糟糕，以后改进
#训练决策树
dtModel5=dt.DecisionTree()
deciTree5=dtModel5.fit(X2,y2,model_type='cart_r',depth_max=None,output=True)
dtModel5.plot()
#dtModel5.print_nodes()
#预测
result5=dtModel5.predict(test_X2)
score5=dtModel5.assess(test_y2,result5)
print('\nCART(Regressor) test score: %f'%score5)

#存储和读取测试
dtModel5.save_tree('D:\\Model\\deciTree2.txt')
dtModel5.read_tree('D:\\Model\\deciTree2.txt')

#剪枝测试(ccp)
tree5_pruned=dtModel5.pruning(test_X2,test_y2,mode='ccp',return_tree=True)
dtModel5.plot(tree=tree5_pruned)
result5_pruned=dtModel5.predict(test_X2,tree=tree5_pruned)
score5_pruned=dtModel5.assess(test_y2,result5_pruned)
print('\nCART(Regressor) test score after pruning: %f'%score5_pruned)

#拟合结果(由于特征维度较多，只能对每个维度进行投影)
for i in range(len(test_X2.columns)):
    test_x2=test_X2.iloc[:,i].sort_values()
    plt.scatter(test_x2,test_y2[test_x2.index],c='b')
    plt.plot(test_x2,result5_pruned[test_x2.index],c='r')
    plt.xlabel(test_X2.columns[i])
    plt.ylabel(test_y2.name)
    plt.suptitle("Fitting result on feature: %s"%(test_X2.columns[i]))
    plt.show()

#与sklearn对照
start = time.clock()
sk_dtModel2 = tree.DecisionTreeRegressor()
sk_dtModel2.fit(X2, y2)
end = time.clock()
print('\ntime used for trainning:%f'%(end-start))
print('\nsklearn train score:%f'%sk_dtModel2.score(X2,y2))

sk_result2 = sk_dtModel2.predict(test_X2)
print('\nsklearn test score:%f'%sk_dtModel2.score(test_X2,test_y2))

'''
start=time.clock()
trees=[]
for i in range(1000):
    trees.append(dtModel5.tree.copy())
print('time cost: %f'%(time.clock()-start))
'''