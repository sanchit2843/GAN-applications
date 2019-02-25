import shutil
import os
import wget
#!pip install -i https://test.pypi.org/simple/ supportlib
import supportlib.gettingdata as getdata

getdata.tarextract('/content/images.tar')
wget.download('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
path = '' #path to extracted data
path_to_tarfile = ''
des = './train' #path to destination to copy all data
getdata.tarextract(path_to_tarfile)

!mkdir train
from tqdm import tqdm
for i in tqdm(os.listdir(path)):
  path1 = os.path.join(path , i)
  for j in os.listdir(path1):
    path2 = os.path.join(path1,j)
    path3 = '{}_{}.jpg'.format(i,j)
    des1 = os.path.join(des,path3)
    shutil.copyfile(path2 , des1)
