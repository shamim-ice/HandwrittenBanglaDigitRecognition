import pandas as pd
from ImageToAraayPreprocessor import ImageToArray
from prework_numtadb import SimplePreprossesing
from train_dataload import SimpleDatasetLoader
from sklearn.preprocessing import LabelBinarizer
import numpy as np


sp = SimplePreprossesing(32,32)
iap = ImageToArray()
sdl = SimpleDatasetLoader([sp, iap])

print('[INFO] loading image...')
p = 'E:/Code/project/numtaDB/training-a'
csvdata = pd.read_csv('E:/Code/project/numtaDB/training-a.csv')
csvdata = csvdata[['filename', 'digit']]
sdl.Load(p, csvdata, verbos=500)

p = 'E:/Code/project/numtaDB/training-b'
csvdata = pd.read_csv('E:/Code/project/numtaDB/training-b.csv')
csvdata = csvdata[['filename', 'digit']]
sdl.Load(p, csvdata, verbos=500)

p = 'E:/Code/project/numtaDB/training-c'
csvdata = pd.read_csv('E:/Code/project/numtaDB/training-c.csv')
csvdata = csvdata[['filename', 'digit']]
sdl.Load(p, csvdata, verbos=500)

p = 'E:/Code/project/numtaDB/training-d'
csvdata = pd.read_csv('E:/Code/project/numtaDB/training-d.csv')
csvdata = csvdata[['filename', 'digit']]
sdl.Load(p, csvdata, verbos=500)

p = 'E:/Code/project/numtaDB/training-e'
csvdata = pd.read_csv('E:/Code/project/numtaDB/training-e.csv')
csvdata = csvdata[['filename', 'digit']]
data, labels = sdl.Load(p, csvdata, verbos=500)

data = np.array(data)
print(data.shape)
labels = np.array(labels)

data = data.astype('float') / 255.0
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

pd.to_pickle(data, 'E:/Code/htmljscss/Data_32_ostu.pkl')
pd.to_pickle(labels, 'E:/Code/htmljscss/Labels_32_ostu.pkl')
