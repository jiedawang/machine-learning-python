# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#优化目标函数
def f(x):
    if len(x.shape)==1:
        return x[0]**2+50*x[1]**2
    else:
        return x[:,0]**2+50*x[:,1]**2

#目标函数梯度
def g(x):
    if len(x.shape)==1:
        return np.array([2*x[0],100*x[1]])
    else:
        return np.c_[2*x[:,0],100*x[:,1]]

#随机梯度下降
def sgd(init_x,grad,learning_rate=1.0,iter_max=200,stop_value=1.):
    x=np.array(init_x,dtype='float64')
    x_h=[x.tolist()]
    fx_h=[f(x)]
    no_desc=0
    ready_stop=0
    for i in range(iter_max):
        g=grad(x)
        x-=learning_rate*g
        x_h.append(x.tolist())
        fx_h.append(f(x))
        if fx_h[-1]<stop_value:
            ready_stop+=1
        else:
            ready_stop=0
        if ready_stop>=3:
            print('reach threshold,early stopping!')
            print('stop iter: %d'%(i+1))
            break
        if (fx_h[-2]-fx_h[-1])/fx_h[-2]<1e-4:
            no_desc+=1
        else:
            no_desc=0
        if no_desc>=20:
            print('no desc,early stopping!')
            print('stop iter: %d'%(i+1))
            i=iter_max-1
            break
    print('latest change value: %f'%(fx_h[-2]-fx_h[-1]))
    print('latest value: %f'%fx_h[-1])
    x_h=np.array(x_h)
    fx_h=np.array(fx_h)
    return x_h,fx_h,i

#动量加速梯度下降
def magd(init_x,grad,learning_rate=1.0,iter_max=50,p=0.9,stop_value=1.):
    x=np.array(init_x,dtype='float64')
    v=np.zeros_like(x)
    x_h=[x.tolist()]
    fx_h=[f(x)]
    no_desc=0
    ready_stop=0
    for i in range(iter_max):
        g=grad(x)
        v=p*v+learning_rate*g
        x-=v
        x_h.append(x.tolist())
        fx_h.append(f(x))
        if fx_h[-1]<stop_value:
            ready_stop+=1
        else:
            ready_stop=0
        if ready_stop>=3:
            print('reach threshold,early stopping!')
            print('stop iter: %d'%(i+1))
            break
        if (fx_h[-2]-fx_h[-1])/fx_h[-2]<1e-4:
            no_desc+=1
        else:
            no_desc=0
        if no_desc>=20:
            print('no desc,early stopping!')
            print('stop iter: %d'%(i+1))
            i=iter_max-1
            break
    print('latest change value: %f'%(fx_h[-2]-fx_h[-1]))
    print('latest value: %f'%fx_h[-1])
    x_h=np.array(x_h)
    fx_h=np.array(fx_h)
    return x_h,fx_h,i

#Nesterov加速梯度下降
def nagd(init_x,grad,learning_rate=1.0,iter_max=50,p=0.9,stop_value=1.):
    x=np.array(init_x,dtype='float64')
    v=np.zeros_like(x)
    x_h=[x.tolist()]
    fx_h=[f(x)]
    no_desc=0
    ready_stop=0
    for i in range(iter_max):
        x-=p*v
        g=grad(x)
        v=p*v+learning_rate*g
        x-=learning_rate*g
        x_h.append(x.tolist())
        fx_h.append(f(x))
        if fx_h[-1]<stop_value:
            ready_stop+=1
        else:
            ready_stop=0
        if ready_stop>=3:
            print('reach threshold,early stopping!')
            print('stop iter: %d'%(i+1))
            break
        if (fx_h[-2]-fx_h[-1])/fx_h[-2]<1e-4:
            no_desc+=1
        else:
            no_desc=0
        if no_desc>=20:
            print('no desc,early stopping!')
            print('stop iter: %d'%(i+1))
            i=iter_max-1
            break
    print('latest change value: %f'%(fx_h[-2]-fx_h[-1]))
    print('latest value: %f'%fx_h[-1])
    x_h=np.array(x_h)
    fx_h=np.array(fx_h)
    return x_h,fx_h,i

#adam
def adam(init_x,grad,learning_rate=1.0,iter_max=50,
          beta1=0.9,beta2=0.999,eps=1e-8,stop_value=1.,atten=False):
    x=np.array(init_x,dtype='float64')
    m=np.zeros_like(x)
    v=np.zeros_like(x)
    t=1
    x_h=[x.tolist()]
    fx_h=[f(x)]
    no_desc=0
    ready_stop=0
    for i in range(iter_max):
        g=grad(x)
        #m=beta1*m+(1.-beta1)*g
        #v=beta2*v+(1.-beta2)*(g**2)
        #m_=m/(1.-beta1**t)
        #v_=v/(1.-beta2**t)
        #x-=learning_rate/(np.sqrt(v_)+eps)*m_
        m=beta1*m+learning_rate*g
        v=beta2*v+learning_rate*(g**2)
        x-=m/(np.sqrt(v)+eps)
        t+=1
        x_h.append(x.tolist())
        fx_h.append(f(x))
        if fx_h[-1]<stop_value:
            ready_stop+=1
        else:
            ready_stop=0
        if ready_stop>=3:
            print('reach threshold,early stopping!')
            print('stop iter: %d'%(i+1))
            break
        if (fx_h[-2]-fx_h[-1])/fx_h[-2]<1e-4:
            no_desc+=1
        else:
            no_desc=0
        if no_desc>=20:
            print('no desc,early stopping!')
            print('stop iter: %d'%(i+1))
            i=iter_max-1
            break
        if atten==True:
            learning_rate*=(iter_max-i-1)/(iter_max-i)
            print('learning rate: %f'%learning_rate)
    print('latest change value: %f'%(fx_h[-2]-fx_h[-1]))
    print('latest value: %f'%fx_h[-1])
    x_h=np.array(x_h)
    fx_h=np.array(fx_h)
    return x_h,fx_h,i

#学习率批量测试
def learning_rate_bulk_test(start,end,step,gd_func,init_x,iter_max=1000,stop_value=1.):
    stop_iters=[]
    latest_values=[]
    latest_changes=[]
    learning_rates=[]
    learning_rate=start
    while learning_rate<=end:
        learning_rates.append(learning_rate)
        x_h1,fx_h1,iter1=gd_func(init_x,g,learning_rate=learning_rate,
                                 iter_max=iter_max,stop_value=stop_value/10)
        x_h2,fx_h2,iter2=gd_func(init_x,g,learning_rate=learning_rate,
                                 iter_max=iter_max,stop_value=stop_value)
        x_h3,fx_h3,iter3=gd_func(init_x,g,learning_rate=learning_rate,
                                 iter_max=iter_max,stop_value=stop_value*10)
        stop_iters.append([iter1,iter2,iter3])
        latest_values.append([fx_h1[-1],fx_h2[-1],fx_h3[-1]])
        latest_changes.append([np.abs(fx_h1[-2]-fx_h1[-1]),np.abs(fx_h2[-2]-fx_h2[-1]),
                               np.abs(fx_h3[-2]-fx_h3[-1])])
        learning_rate+=step
    stop_iters=pd.DataFrame(stop_iters,index=learning_rates,
                            columns=['stop_value/10','stop_value','stop_value*10'])
    stop_iters_=stop_iters.values
    best_idx=np.where(stop_iters_[:,1]==stop_iters_[:,1].min())[0][0]
    print('\nbest learning rate: %f stop iter: %f'%
         (learning_rates[best_idx],stop_iters_[best_idx,1]))
    stop_iters.plot()
    plt.xlabel('learning_rate')
    plt.ylabel('stop_iter')
    plt.show()

#绘制等高线图
def plot_contour(f,center=(0,0),xy_range=(200,100)):
    #等高线图
    x0=np.linspace(center[0]-xy_range[0],center[0]+xy_range[0],1001)
    y0=np.linspace(center[1]-xy_range[1],center[1]+xy_range[1],1001)
    X,Y=np.meshgrid(x0,y0)
    Z=f(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)
    plt.contour(X,Y,Z,colors='black')
    #最低点
    min_where=np.where(Z==Z.min())
    idx1,idx2=min_where[0][0],min_where[1][0]
    plt.scatter(X[idx1][idx2],Y[idx1][idx2])

#绘制优化过程
def plot_process(x_h,fx_h,f=f,center=(0,0),xy_range=(200,100)):
    
    #等高线图
    plot_contour(f,center,xy_range)

    #绘制优化过程
    for i in range(len(x_h)-1):
        plt.plot(x_h[i:i+2,0],x_h[i:i+2,1])
    
    plt.xlim((-xy_range[0],xy_range[0]))
    plt.ylim((-xy_range[1],xy_range[1]))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

#f变化曲线    
def plot_f_change(fx):
    plt.plot(range(len(fx_h)),fx_h)
    plt.xlabel('iter')
    plt.ylabel('f')
    plt.show()  
    
#起始点
#init_x=[np.random.randint(-200,200),np.random.randint(-100,100)]
init_x=[160,80]
xy_range=(200,100)
stop_value=1.0

init_x=[1.2,0.8]
xy_range=(1.5,1)
stop_value=0.01

beta1=0.9
beta2=0.999
eps=1e-8
learning_rate=10
t=1
x=np.array(init_x,dtype='float64')
for i in range(500):
    g_=g(x)
    m=(1.-beta1)*g_
    v=(1.-beta2)*(g_**2)
    m_=m/(1.-beta1**t)
    v_=v/(1.-beta2**t)
    delta=learning_rate/(np.sqrt(v_)+eps)*m_
    x-=delta
    t+=1
    print(delta)

#sgd
x_h,fx_h,i=sgd(init_x,g,learning_rate=0.010,iter_max=254,stop_value=stop_value)
x_h,fx_h,i=sgd(init_x,g,learning_rate=0.016,iter_max=159,stop_value=stop_value)
x_h,fx_h,i=sgd(init_x,g,learning_rate=0.020,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=sgd(init_x,g,learning_rate=0.019,iter_max=134,stop_value=stop_value)

#magd
x_h,fx_h,i=magd(init_x,g,learning_rate=0.010,iter_max=125,stop_value=stop_value)
x_h,fx_h,i=magd(init_x,g,learning_rate=0.003,iter_max=120,stop_value=stop_value)
x_h,fx_h,i=magd(init_x,g,learning_rate=0.001,iter_max=207,stop_value=stop_value)
x_h,fx_h,i=magd(init_x,g,learning_rate=0.030,iter_max=136,stop_value=stop_value)
x_h,fx_h,i=magd(init_x,g,learning_rate=0.040,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=magd(init_x,g,learning_rate=0.006,iter_max=118,stop_value=stop_value)

#nagd
x_h,fx_h,i=nagd(init_x,g,learning_rate=0.010,iter_max=63,stop_value=stop_value)
x_h,fx_h,i=nagd(init_x,g,learning_rate=0.003,iter_max=88,stop_value=stop_value)
x_h,fx_h,i=nagd(init_x,g,learning_rate=0.001,iter_max=213,stop_value=stop_value)
x_h,fx_h,i=nagd(init_x,g,learning_rate=0.030,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=nagd(init_x,g,learning_rate=0.006,iter_max=55,stop_value=stop_value)

#adam
x_h,fx_h,i=adam(init_x,g,learning_rate=0.010,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=adam(init_x,g,learning_rate=1,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=adam(init_x,g,learning_rate=100,iter_max=108,stop_value=stop_value)
x_h,fx_h,i=adam(init_x,g,learning_rate=10000,iter_max=300,stop_value=stop_value)
x_h,fx_h,i=adam(init_x,g,learning_rate=23.5,iter_max=89,stop_value=stop_value)

#绘制优化过程
plot_process(x_h,fx_h,xy_range=xy_range)

#f变化曲线
plot_f_change(fx_h)

#批量测试学习率
learning_rate_bulk_test(0.010,0.022,0.001,sgd,init_x,iter_max=300,stop_value=stop_value)
learning_rate_bulk_test(0.001,0.040,0.001,magd,init_x,iter_max=300,stop_value=stop_value)
learning_rate_bulk_test(0.5,100,0.5,adam,init_x,iter_max=300,stop_value=stop_value)
