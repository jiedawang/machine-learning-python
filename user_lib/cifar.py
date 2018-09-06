# -*- coding: utf-8 -*-
from PIL import Image
import pandas as pd
import numpy as np
import os
import struct
import shutil
import pickle

#用于管理cifar图片数据集
#下载地址：https://www.cs.toronto.edu/~kriz/cifar.html
class CifarManager:
    
    def __init__(self,file_dir=None):
        if type(file_dir)==type(None):
            print('\nThe default path of cifar is set to:')
            print("D:\\training_data\\used\\cifar-10-batches-py")
            file_dir="D:\\training_data\\used\\cifar-10-batches-py"
        self.file_dir=file_dir
    
    #使用pickle加载数据集文件
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict0 = pickle.load(fo, encoding='bytes')
        return dict0
    
    #加载以pickle打包的数据
    def load_from_pickle(self,file_dir):
        print('\nreading data---')
        
        #校对文件      
        filelist=os.listdir(file_dir)
        filelist_check=['data_batch_1','data_batch_2','data_batch_3',
                        'data_batch_4','data_batch_5','test_batch',
                        'batches.meta']
        
        for filename in filelist_check:
            if filename not in filelist:
                raise IOError('missing file: %s'%filename)
        filelist=filelist_check
        
        train_images=np.zeros((50000,32,32,3))
        train_labels=np.zeros((50000,)).astype('<U10')
        test_images=np.zeros((10000,32,32,3))
        test_labels=np.zeros((10000,)).astype('<U10')
        offset=0
        
        #遍历分组        
        for filename in filelist_check:
            if filename=='batches.meta':
                continue
            #加载数据
            buf=self.unpickle(file_dir+"\\"+filename)
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
        
        buf=self.unpickle(file_dir+"\\batches.meta")
        label_names=buf[b'label_names']
        #转换标签        
        for i in range(len(label_names)):
            train_labels[train_labels==str(i)]=label_names[i]
            test_labels[test_labels==str(i)]=label_names[i]
            
        print('completed')
            
        return train_images,train_labels,test_images,test_labels
        
    #转换为图片
    def to_images(self,rewrite=False):
        file_dir=self.file_dir
        
        #创建输出文件夹
        folders_name=[file_dir+"\\train",file_dir+"\\test"]
        for folder_name in folders_name:
            if os.path.exists(folder_name)==False:
                os.makedirs(folder_name)
            else:
                if rewrite==True:
                    print('\nfolder already exists,deleting---')
                    shutil.rmtree(folder_name)
                    print('folder has been deleted\n')
                    os.makedirs(folder_name)
                else:
                    raise IOError('\nfolder already exists')
        
        #加载数据到numpy数组
        train_images,train_labels,test_images,test_labels=self.load_from_pickle(file_dir)
            
        #转换为图片保存   
        for i in range(train_labels.shape[0]):
            image=Image.fromarray(np.uint8(train_images[i]))
            label=train_labels[i]
            print('train set: save %d image as %s'%(i,label))
            image.save(file_dir+"\\train\\%d_%s.png"%(i,label),'png')
            
        for i in range(test_labels.shape[0]):
            image=Image.fromarray(np.uint8(test_images[i]))
            label=test_labels[i]
            print('test set: save %d image as %s'%(i,label))
            image.save(file_dir+"\\test\\%d_%s.png"%(i,label),'png')
            
        print('completed')
    
    #获取图片名列表            
    def images_list(self):
        if os.path.exists(self.file_dir+"\\train")==False|\
            os.path.exists(self.file_dir+"\\test")==False:
            raise IOError('The image folder does not exist, please run .to_images() first.')
        train_list=os.listdir(self.file_dir+"\\train")
        test_list=os.listdir(self.file_dir+"\\test")
        return train_list,test_list
    
    #以numpy数组的形式读取图片数据到内存
    def read_as_array(self,sort_by_labels=True,from_images=False):
        if from_images==True:
            train_list,test_list=self.images_list()
            train_images=np.zeros((len(train_list),32,32,3))
            train_labels=np.zeros(len(train_list)).astype('<U10')
            test_images=np.zeros((len(test_list),32,32,3))
            test_labels=np.zeros(len(test_list)).astype('<U10')
            print('\nreading train data---')
            for i in range(len(train_list)):
                image=Image.open(self.file_dir+"\\train\\"+train_list[i])
                label=train_list[i].split('.',1)[0].split('_',1)[1]
                train_images[i,:,:,:]=np.asarray(image)
                train_labels[i]=label
            print('completed')
            print('\nreading test data---')
            for i in range(len(test_list)):
                image=Image.open(self.file_dir+"\\test\\"+test_list[i])
                label=test_list[i].split('.',1)[0].split('_',1)[1]
                test_images[i,:,:,:]=np.asarray(image)
                test_labels[i]=label
            print('completed')
        else:
            train_images,train_labels,test_images,test_labels=\
                self.load_from_pickle(self.file_dir)
        train_sort=np.argsort(train_labels)
        test_sort=np.argsort(test_labels)
        train_images=train_images[train_sort,:,:,:]
        train_labels=train_labels[train_sort]
        test_images=test_images[test_sort,:,:,:]
        test_labels=test_labels[test_sort]
        return train_images,train_labels,test_images,test_labels
    
    #图片转数组
    def image_to_array(self,image):
        return np.asarray(image)
    
    #数组转图片
    def array_to_image(self,array):
        if len(array.shape)!=2:
            raise ValueError('Unsupport shape, need (n,m)')
        return Image.fromarray(np.uint8(array))
