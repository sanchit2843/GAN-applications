!pip install wget
!pip install -i https://test.pypi.org/simple/ supportlib
import shutil
import os
import wget
from tqdm import tqdm
import supportlib.gettingdata as getdata
from params import *

wget.download(url)

getdata.tarextract(path_to_tarfile)
#create a seperate training folder and copy content of all breed folders in single folder
!mkdir train
#copy content in single folder
for i in tqdm(os.listdir(path)):
  path1 = os.path.join(path , i)
  for j in os.listdir(path1):
    path2 = os.path.join(path1,j)
    path3 = '{}_{}.jpg'.format(i,j)
    des1 = os.path.join(des,path3)
    shutil.copyfile(path2 , des1)
