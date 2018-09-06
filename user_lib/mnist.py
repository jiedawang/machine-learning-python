# -*- coding: utf-8 -*-
from PIL import Image
import pandas as pd
import numpy as np
import os
import struct
import shutil

#用于管理mnist手写数字图片数据
#下载地址：http://yann.lecun.com/exdb/mnist/
class MnistManager:
    
    def __init__(self,bytefile_dir=None):
        if type(bytefile_dir)==type(None):
            print('\nThe default path of mnist is set to:')
            print("D:\\training_data\\used\\MNIST\\data")
            bytefile_dir="D:\\training_data\\used\\MNIST\\data"
        self.bytefile_dir=bytefile_dir
    
    #将二进制文件转换为图片
    def to_images(self,rewrite=False):
        bytefile_dir=self.bytefile_dir
        
        #校对文件      
        filelist=os.listdir(bytefile_dir)
        filelist_check=['t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte',
                        'train-images.idx3-ubyte','train-labels.idx1-ubyte']
        
        for filename in filelist_check:
            if filename not in filelist:
                raise IOError('missing file: %s'%filename)
        filelist=filelist_check
        
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
        grouplist=filelist_attr["group"].drop_duplicates().tolist()
        
        #遍历分组        
        for n in range(len(grouplist)):
            #创建输出文件夹
            folder_name=bytefile_dir+"\\"+grouplist[n]  
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
            
            #读取二进制数据至缓存（图片和标签数据一起读取）
            images_file_idx=filelist_attr[(filelist_attr['group']==grouplist[n])&
                                          (filelist_attr['type']=='images')].\
                                          index.values[0]
            labels_file_idx=filelist_attr[(filelist_attr['group']==grouplist[n])&
                                          (filelist_attr['type']=='labels')].\
                                          index.values[0]
                
            f=open(bytefile_dir+"\\"+filelist[images_file_idx],'rb')
            buf_images=f.read()
            f.close()
                    
            f=open(bytefile_dir+"\\"+filelist[labels_file_idx],'rb')
            buf_labels=f.read()
            f.close()
            
            #读取头信息     
            index_i,index_l=0,0     
            magic,images,rows,columns=struct.unpack_from('>IIII',buf_images,index_i)
            index_i+=struct.calcsize('>IIII')
            magic,labels=struct.unpack_from('>II',buf_labels,index_l)
            index_l+=struct.calcsize('>II')
            
            #将二进制数据转换为图片       
            for i in range(images):
                image=Image.new('L',(columns,rows))
                for y in range(rows):
                    for x in range(columns):
                        image.putpixel((x,y),int(struct.unpack_from('>B',buf_images,index_i)[0]))
                        index_i+=struct.calcsize('>B')
                label=int(struct.unpack_from('>B',buf_labels,index_l)[0])
                index_l+=struct.calcsize('>B')
                print('save %d image as %d'%(i,label))
                image.save(bytefile_dir+"\\"+grouplist[n]+"\\%d_%d.png"%(i,label),'png')
    
    #获取图片名列表            
    def images_list(self):
        if os.path.exists(self.bytefile_dir+"\\train")==False|\
            os.path.exists(self.bytefile_dir+"\\t10k")==False:
            raise IOError('The image folder does not exist, please run .to_images() first.')
        train_list=os.listdir(self.bytefile_dir+"\\train")
        test_list=os.listdir(self.bytefile_dir+"\\t10k")
        return train_list,test_list
    
    #以numpy数组的形式读取图片到内存
    def read_as_array(self,sort_by_labels=True):
        train_list,test_list=self.images_list()
        train_images=np.zeros((len(train_list),28,28))
        train_labels=np.zeros(len(train_list)).astype('int')
        test_images=np.zeros((len(test_list),28,28))
        test_labels=np.zeros(len(test_list)).astype('int')
        print('\nreading train data---')
        for i in range(len(train_list)):
            image=Image.open(self.bytefile_dir+"\\train\\"+train_list[i])
            label=train_list[i].split('.',1)[0].split('_',1)[1]
            train_images[i,:,:]=np.asarray(image)
            train_labels[i]=int(label)
        print('completed')
        print('\nreading test data---')
        for i in range(len(test_list)):
            image=Image.open(self.bytefile_dir+"\\t10k\\"+test_list[i])
            label=test_list[i].split('.',1)[0].split('_',1)[1]
            test_images[i,:,:]=np.asarray(image)
            test_labels[i]=int(label) 
        print('completed')
        train_sort=np.argsort(train_labels)
        test_sort=np.argsort(test_labels)
        train_images=train_images[train_sort,:,:]
        train_labels=train_labels[train_sort]
        test_images=test_images[test_sort,:,:]
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
        