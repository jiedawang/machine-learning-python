# -*- coding: utf-8 -*-
from PIL import Image
import pandas as pd
import numpy as np
import os
import struct
import shutil
import pickle

#图片管理基类
class ImageDataManager(object):
    
    def __init__(self,datafile_dir,default_datafile_dir,
                 image_shape,label_length,labels,
                 train_samples,test_samples,filelist_check,
                 external_monitor):
        if type(datafile_dir)==type(None):
            message='The default path of image data is set to:\n'+default_datafile_dir
            print(message)
            if type(external_monitor)!=type(None):
                external_monitor(message)
            datafile_dir=default_datafile_dir
        self.datafile_dir=datafile_dir
        self.default_datafile_dir=default_datafile_dir
        self.image_shape=image_shape
        self.label_length=label_length
        self.labels=labels
        self.train_samples=train_samples
        self.test_samples=test_samples
        self.filelist_check=filelist_check
        self.external_monitor=external_monitor
        
    #设置外部监视方法
    def set_external_monitor(self,external_monitor):
        self.external_monitor=external_monitor
    
    #反馈信息   
    def feedback(self,message):
        print(message)
        if type(self.external_monitor)!=type(None):
            self.external_monitor(message)
    
    #校对文件
    def check_file(self):
        #校对文件      
        filelist=os.listdir(self.datafile_dir)
        for filename in self.filelist_check:
            if filename not in filelist:
                raise IOError('missing file: %s'%filename)
        return self.filelist_check
    
    #从数据文件加载数据
    def load_from_datafile(self,chinese_label=False):
        self.feedback('reading data---')

        train_images=np.zeros((self.train_samples,)+self.image_shape)
        train_labels=np.zeros(self.train_samples).astype('<U%d'%self.label_length)
        test_images=np.zeros((self.test_samples,)+self.image_shape)
        test_labels=np.zeros(self.test_samples).astype('<U%d'%self.label_length)
        
        #此处对数据文件读取并处理
        
        self.feedback('completed')
        return train_images,train_labels,test_images,test_labels
    
    #将二进制文件转换为图片
    def to_images(self,rewrite=False,chinese_label=False):
        #创建输出文件夹
        folders_name=[self.datafile_dir+"\\train",self.datafile_dir+"\\test"]
        for folder_name in folders_name:
            if os.path.exists(folder_name)==False:
                os.makedirs(folder_name)
            else:
                if rewrite==True:
                    self.feedback('folder already exists,deleting---')
                    shutil.rmtree(folder_name)
                    self.feedback('folder has been deleted')
                    os.makedirs(folder_name)
                else:
                    raise IOError('folder already exists')
        
        #加载数据到numpy数组
        train_images,train_labels,test_images,test_labels=self.load_from_datafile(chinese_label)
        
        self.feedback('start saving--')
        
        #转换为图片保存   
        for i in range(train_labels.shape[0]):
            image=Image.fromarray(np.uint8(train_images[i]))
            label=train_labels[i]
            image.save(self.datafile_dir+"\\train\\%d_%s.png"%(i+1,label),'png')
            self.feedback('train set: save %d image as %s'%(i+1,label))
            
        for i in range(test_labels.shape[0]):
            image=Image.fromarray(np.uint8(test_images[i]))
            label=test_labels[i]
            image.save(self.datafile_dir+"\\test\\%d_%s.png"%(i+1,label),'png')
            self.feedback('test set: save %d image as %s'%(i+1,label))
        
        self.feedback('completed')

    #获取图片名列表            
    def images_list(self):
        if os.path.exists(self.datafile_dir+"\\train")==False|\
            os.path.exists(self.datafile_dir+"\\test")==False:
            raise IOError('The image folder does not exist, please run .to_images() first.')
        train_list=os.listdir(self.datafile_dir+"\\train")
        test_list=os.listdir(self.datafile_dir+"\\test")
        return train_list,test_list
    
    #以numpy数组的形式读取图片到内存
    def read_as_array(self,sort_by_labels=True,from_images=False,chinese_label=False):
        if from_images==True:
            train_list,test_list=self.images_list()
            train_images=np.zeros((len(train_list),)+self.image_shape)
            train_labels=np.zeros(len(train_list)).astype('<U%d'%self.label_length)
            test_images=np.zeros((len(test_list),)+self.image_shape)
            test_labels=np.zeros(len(test_list)).astype('<U%d'%self.label_length)
            self.feedback('reading train data---')
            for i in range(len(train_list)):
                image=Image.open(self.datafile_dir+"\\train\\"+train_list[i])
                label=train_list[i].split('.',1)[0].split('_',1)[1]
                train_images[i]=np.asarray(image)
                train_labels[i]=label
            self.feedback('completed')
            self.feedback('reading test data---')
            for i in range(len(test_list)):
                image=Image.open(self.datafile_dir+"\\test\\"+test_list[i])
                label=test_list[i].split('.',1)[0].split('_',1)[1]
                test_images[i]=np.asarray(image)
                test_labels[i]=label 
            self.feedback('completed')
        else:
            train_images,train_labels,test_images,test_labels=self.load_from_datafile(chinese_label)
        return self.sort_array(train_images,train_labels,test_images,test_labels)
    
    #数据集排序
    def sort_array(self,train_images,train_labels,test_images,test_labels):
        train_sort=np.argsort(train_labels)
        test_sort=np.argsort(test_labels)
        train_images=train_images[train_sort]
        train_labels=train_labels[train_sort]
        test_images=test_images[test_sort]
        test_labels=test_labels[test_sort]
        return train_images,train_labels,test_images,test_labels
    
    #图片转数组
    def image_to_array(self,image):
        return np.asarray(image)
    
    #数组转图片
    def array_to_image(self,array):
        if array.shape!=self.image_shape:
            raise ValueError('Unsupport shape, need %s'%str(self.image_shape))
        return Image.fromarray(np.uint8(array))

#用于管理mnist手写数字图片数据
#下载地址：http://yann.lecun.com/exdb/mnist/
class MnistManager(ImageDataManager):
    
    def __init__(self,datafile_dir=None,external_monitor=None):
        super(MnistManager,self).__init__(
                datafile_dir=datafile_dir,
                default_datafile_dir="D:\\training_data\\used\\MNIST\\data",
                image_shape=(28,28),
                label_length=1,
                labels=10,
                train_samples=60000,
                test_samples=10000,
                filelist_check=('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',
                                'train-images.idx3-ubyte','train-labels.idx1-ubyte'),
                external_monitor=external_monitor
                )
    
    #从数据文件加载数据    
    def load_from_datafile(self,chinese_label=False):
        self.feedback('reading data---')
        filelist=self.check_file()
    
        #从文件名中分离分组信息
        filelist_attr=[]        
        for i in range(len(filelist)):
            fname_ext=filelist[i].split(".",1)
            if len(fname_ext)==1:
                continue
            group_type=fname_ext[0].split("-",1)
            if len(group_type)==1:
                continue
            filelist_attr.append([group_type[0],group_type[1],fname_ext[1]])
        
        filelist_attr=pd.DataFrame(filelist_attr,columns=["group","type","ext"])  

        train_images=np.zeros((self.train_samples,)+self.image_shape)
        train_labels=np.zeros(self.train_samples).astype('<U%d'%self.label_length)
        test_images=np.zeros((self.test_samples,)+self.image_shape)
        test_labels=np.zeros(self.test_samples).astype('<U%d'%self.label_length)
        
        #遍历文件
        for file_idx,filename in enumerate(filelist):
                            
            if filelist_attr.loc[file_idx,'type']=='images':
                #打开文件                              
                f=open(self.datafile_dir+"\\"+filename,'rb')
                buf_images=f.read()
                f.close()
                
                #读取头信息     
                index_i,index_l=0,0     
                magic,images,rows,columns=struct.unpack_from('>IIII',buf_images,index_i)
                index_i+=struct.calcsize('>IIII')
                
                #将array转换为图片       
                for i in range(images):
                    data=struct.unpack_from('>%dB'%(rows*columns),buf_images,index_i)
                    if filelist_attr.loc[file_idx,'group']=='train':
                        train_images[i,:,:]=np.array(data).reshape(rows,columns)
                    else:
                        test_images[i,:,:]=np.array(data).reshape(rows,columns)
                    index_i+=struct.calcsize('>%dB'%(rows*columns))
            
            else:
                #打开文件
                f=open(self.datafile_dir+"\\"+filename,'rb')
                buf_labels=f.read()
                f.close()
    
                #读取头信息  
                magic,labels=struct.unpack_from('>II',buf_labels,index_l)
                index_l+=struct.calcsize('>II')
                       
                #将二进制数据转换为标签            
                for i in range(labels):
                    data=int(struct.unpack_from('>B',buf_labels,index_l)[0])
                    if filelist_attr.loc[file_idx,'group']=='train':
                        train_labels[i]=str(data)
                    else:
                        test_labels[i]=str(data)
                    index_l+=struct.calcsize('>B')
        
        self.feedback('completed')
        return train_images,train_labels,test_images,test_labels
    
#用于管理cifar图片数据集
#下载地址：https://www.cs.toronto.edu/~kriz/cifar.html
class CifarManager(ImageDataManager):
    
    def __init__(self,datafile_dir=None,external_monitor=None):
        super(CifarManager,self).__init__(
                datafile_dir=datafile_dir,
                default_datafile_dir="D:\\training_data\\used\\cifar-10-batches-py",
                image_shape=(32,32,3),
                label_length=10,
                labels=10,
                train_samples=50000,
                test_samples=10000,
                filelist_check=('data_batch_1','data_batch_2','data_batch_3',
                                'data_batch_4','data_batch_5','test_batch',
                                'batches.meta'),
                external_monitor=external_monitor
                )
        self.labels={b'airplane':'飞机',
                     b'automobile':'汽车',
                     b'bird':'鸟',
                     b'cat':'猫',
                     b'deer':'鹿',
                     b'dog':'狗',
                     b'frog':'青蛙',
                     b'horse':'马',
                     b'ship':'船',
                     b'truck':'卡车'}
    
    #使用pickle加载数据集文件
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict0 = pickle.load(fo, encoding='bytes')
        return dict0
    
    #从数据文件加载数据
    def load_from_datafile(self,chinese_label=False):
        self.feedback('reading data---')
        filelist=self.check_file()
        
        train_images=np.zeros((self.train_samples,)+self.image_shape)
        train_labels=np.zeros(self.train_samples).astype('<U%d'%self.label_length)
        test_images=np.zeros((self.test_samples,)+self.image_shape)
        test_labels=np.zeros(self.test_samples).astype('<U%d'%self.label_length)

        offset=0
        
        #遍历分组
        for filename in filelist:
            if filename=='batches.meta':
                continue
            #加载数据
            buf=self.unpickle(self.datafile_dir+"\\"+filename)
            data,labels=buf[b'data'],np.array(buf[b'labels'])
            
            #重组图片数据    
            image_bytes=np.zeros((len(data),32,32,3))
            image_bytes[:,:,:,0]=data[:,:1024].reshape((-1,32,32))
            image_bytes[:,:,:,1]=data[:,1024:2048].reshape((-1,32,32))
            image_bytes[:,:,:,2]=data[:,2048:].reshape((-1,32,32))
            
            #合并数据集
            if filename=='test_batch':
                test_images[:,:,:,:]=image_bytes
                test_labels[:]=labels
            else:
                train_images[offset:offset+10000,:,:,:]=image_bytes
                train_labels[offset:offset+10000]=labels
                offset+=10000
        
        buf=self.unpickle(self.datafile_dir+"\\batches.meta")
        label_names=buf[b'label_names']
        #转换标签        
        for i in range(len(label_names)):
            if chinese_label==False:
                label_name=label_names[i]
            else:
                label_name=self.labels[label_names[i]]
            train_labels[train_labels==str(i)]=label_name
            test_labels[test_labels==str(i)]=label_name
            
        self.feedback('completed')
            
        return train_images,train_labels,test_images,test_labels   