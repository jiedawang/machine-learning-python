# -*- coding: utf-8 -*-
from PIL import Image
import pandas as pd
import os
import struct

bytefile_dir="D:\\训练数据集\\手写数字图片\\MNIST\\data"

filelist=os.listdir(bytefile_dir)
filelist_attr=[]

for i in range(len(filelist)):
    temp=filelist[i].split(".",1)
    temp2=temp[0].split("-",1)
    filelist_attr.append([temp2[0],temp2[1],temp[1]])
    
filelist_attr=pd.DataFrame(filelist_attr,columns=["name","type","ext"])

namelist=filelist_attr["name"].drop_duplicates().tolist()

for n in range(len(namelist)):
    images_file_idx=filelist_attr[(filelist_attr['name']==namelist[n])&
                                  (filelist_attr['type']=='images')].\
                                  index.values[0]
    labels_file_idx=filelist_attr[(filelist_attr['name']==namelist[n])&
                                  (filelist_attr['type']=='labels')].\
                                  index.values[0]

    f=open(bytefile_dir+"\\"+filelist[images_file_idx],'rb')
    buf_images=f.read()
    f.close()
    
    f=open(bytefile_dir+"\\"+filelist[labels_file_idx],'rb')
    buf_labels=f.read()
    f.close()
    
    index_i=0
    index_l=0
    
    magic,images,rows,columns=struct.unpack_from('>IIII',buf_images,index_i)
    index_i+=struct.calcsize('>IIII')
    magic,labels=struct.unpack_from('>II',buf_labels,index_l)
    index_l+=struct.calcsize('>II')
    
    if os.path.exists(bytefile_dir+"\\"+namelist[n])==False:
        os.makedirs(bytefile_dir+"\\"+namelist[n])
    
    for i in range(images):
        image=Image.new('L',(columns,rows))
        for y in range(rows):
            for x in range(columns):
                image.putpixel((x,y),int(struct.unpack_from('>B',buf_images,index_i)[0]))
                index_i+=struct.calcsize('>B')
        label=int(struct.unpack_from('>B',buf_labels,index_l)[0])
        index_l+=struct.calcsize('>B')
        print('save %d image as %d'%(i,label))
        image.save(bytefile_dir+"\\"+namelist[n]+"\\%d_%d.png"%(i,label),'png')
    

