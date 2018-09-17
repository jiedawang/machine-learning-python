# -*- coding: utf-8 -*-
from numba import cuda,jit,vectorize,guvectorize,float32,prange,njit
import numpy as np
import time
import math
from concurrent.futures import ThreadPoolExecutor,as_completed
import numexpr as ne
import scipy as sp
import pyculib as pc
import cupy as cp

#init
N=1000
A=np.ones((N,N),dtype=np.float32)
B=np.ones((N,N),dtype=np.float32)
C=np.ones((N,N),dtype=np.float32)

del A2,B2
A2=cp.array(A)
B2=cp.array(B)
C_list=[]
C2=cp.add(A2,B2)
C_list.append(C2)

#sigmoid
#注：使用exp(n)比e**n快
#numpy
def sigmoid0(a):
    return 1./(1.+np.exp(-1.*a))

#numexpr
def sigmoid3(a):
    return ne.evaluate("1./(1.+exp(-1.*a))")

#jit
@jit(nopython=True,nogil=True)
def sigmoid1_1(a):
    return 1./(1.+np.exp(-1.*a))

#jit+threads
#必须要加nogil=True,不然多线程无效
def sigmoid1_2(a,threads_n=4,tasks_n=4):
    c=np.empty_like(a)
    chunks_size=math.ceil(a.shape[0]/tasks_n)
    tp=ThreadPoolExecutor(max_workers=threads_n)
    a_chunks=[a[i*chunks_size:(i+1)*chunks_size,:] for i in range(tasks_n)]
    for i,data in enumerate(tp.map(sigmoid1_1,a_chunks)):
        c[i*chunks_size:(i+1)*chunks_size,:]=data
    return c

#自定义ufunc,cpu模式
@vectorize(['float32(float32)','float64(float64)'],target='cpu')
def sigmoid2_1(a):
    return 1./(1.+np.exp(-1.*a))

#自定义ufunc,parallel模式
@vectorize(['float32(float32)','float64(float64)'],target='parallel')
def sigmoid2_2(a):
    return 1./(1.+np.exp(-1.*a))

#自定义ufunc,cuda模式
@vectorize(['float32(float32)','float64(float64)'],target='cuda')
def sigmoid2_3(a):
    return 1./(1.+math.exp(-1.*a))

start=time.clock()
C0=sigmoid0(A)
print('\nnumpy compute time used: %f'%(time.clock()-start))

start=time.clock()
C3=sigmoid3(A)
print('\nnumexpr compute time used: %f'%(time.clock()-start))

start=time.clock()
C1=sigmoid1_1(A)
print('\nnumba.jit compute time used: %f'%(time.clock()-start))

start=time.clock()
C1=sigmoid1_2(A)
print('\nnumba.jit(nogil+threads) compute time used: %f'%(time.clock()-start))

start=time.clock()
C2=sigmoid2_1(A)
print('\nnumba.vectorize(cpu) compute time used: %f'%(time.clock()-start))

start=time.clock()
C2=sigmoid2_2(A)
print('\nnumba.vectorize(parallel) compute time used: %f'%(time.clock()-start))

start=time.clock()
C2=sigmoid2_3(A)
print('\nnumba.vectorize(cuda) compute time used: %f'%(time.clock()-start))

#dot
@cuda.jit
def cuda_dot_kernel1(a,b,c):
    x,y=cuda.grid(2)
    tmp = 0.
    if (x>=c.shape[0])|(y>=c.shape[1]):
        return 
    for k in range(a.shape[1]):
        tmp += a[x, k] * b[k, y]
    #结果矩阵中[x,y]位置的值
    c[x, y] = tmp
      
def cuda_dot1(a,b):
    if a.shape[1]!=b.shape[0]:
        raise ValueError('shape %s does not match to shape %s'%
                         (str(a.shape),str(b.shape)))
    try:
        TPB=16
        threadsperblock=(TPB,TPB)
        blockspergrid_x=math.ceil(a.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(b.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        da=cuda.to_device(a)
        db=cuda.to_device(b)
        dc=cuda.device_array((a.shape[0],b.shape[1]),dtype=a.dtype)
        cuda_dot_kernel1[blockspergrid,threadsperblock](da,db,dc)
        c=dc.copy_to_host()
        return c
    except:
        cuda.close()
        raise

@cuda.jit
def cuda_dot_kernel2(a,b,c):
    TPB=16
    #在线程块的共享内存中创建缓存数组
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    #当前线程在整个线程块网格中的绝对索引
    x,y=cuda.grid(2)
    #当前线程在所属线程块中的相对索引
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    #线程块网格的形状x
    bpg = cuda.gridDim.x
    #越界直接返回
    if (x>=c.shape[0])|(y>=c.shape[1]):
        return  
    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = a[x, ty + i * TPB]
        sB[tx, ty] = b[tx + i * TPB, y]
        # Wait until all threads finish preloading
        cuda.syncthreads()
        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        # Wait until all threads finish computing
        cuda.syncthreads()
    #结果矩阵中[x,y]位置的值
    c[x, y] = tmp
      
def cuda_dot2(a,b):
    if a.shape[1]!=b.shape[0]:
        raise ValueError('shape %s does not match to shape %s'%
                         (str(a.shape),str(b.shape)))
    try:
        c=np.ones((a.shape[0],b.shape[1]),dtype=a.dtype)
        TPB=16
        threadsperblock=(TPB,TPB)
        blockspergrid_x=math.ceil(a.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(b.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        da=cuda.to_device(a)
        db=cuda.to_device(b)
        dc=cuda.device_array((a.shape[0],b.shape[1]),dtype=a.dtype)
        cuda_dot_kernel2[blockspergrid,threadsperblock](da,db,dc)
        c=dc.copy_to_host()
        return c
    except:
        cuda.close()
        raise
        
@jit(nopython=True,fastmath=True,nogil=True,parallel=True)
def jit_dot(a,b):
    return np.dot(a,b)

start=time.clock()
C1=np.dot(A,B)
print('\nnumpy compute time used: %f'%(time.clock()-start))

del C5
start=time.clock()
C5=cp.dot(A2,B2)
print('\ncupy compute time used: %f'%(time.clock()-start))

start=time.clock()
C4=jit_dot(A,B)
print('\njit compute time used: %f'%(time.clock()-start))

start=time.clock()
C2=cuda_dot1(A,B)
print('\ncuda jit 1 compute time used: %f'%(time.clock()-start))

start=time.clock()
C2=cuda_dot2(A,B)
print('\ncuda jit 2 compute time used: %f'%(time.clock()-start))

start=time.clock()
C3=pc.blas.gemm('N','N',1.,A,B)
print('\npyculib compute time used: %f'%(time.clock()-start))

@jit(nopython=True,nogil=True,cache=True)
def mini_batch_(X,Y,size=256):
    if size>len(Y):
        size=len(Y)
    random_idx=np.random.choice(len(Y),size,replace=False)
    return X[random_idx],Y[random_idx]

def mini_batchs(X,Y,size=256,threads_n=4,tasks_n=4):
    X_batchs,Y_batchs=[],[]
    tp=ThreadPoolExecutor(max_workers=threads_n)
    tasks=[tp.submit(mini_batch_,X,Y) for i in range(tasks_n)]
    for task in as_completed(tasks):
        X_batchs.append(task.result()[0])
        Y_batchs.append(task.result()[1])
    return X_batchs,Y_batchs

start=time.clock()
for i in range(4):
    mb_X1,mb_y1=mini_batch_(A,B)
print('\nmini-batch time used: %f'%(time.clock()-start))

start=time.clock()
X_batchs,Y_batchs=mini_batchs(A,B,tasks_n=4)
for i in range(4):
    X_,Y_=X_batchs.pop(0),Y_batchs.pop(0)
print('\nmini-batch time used: %f'%(time.clock()-start))