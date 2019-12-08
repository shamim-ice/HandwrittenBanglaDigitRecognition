from ImageToAraayPreprocessor import ImageToArray
from prework_numtadb import SimplePreprossesing
from test_dataload import SimpleDatasetLoader
from keras.models import load_model
import cv2
import os

classLabels=['zero','one','two','three','four','five','six','seven','eight','nine']
print('[INFO] loading image...')
imagePaths=os.listdir('E:/Code/project/numtaDB/testing-new')

sp=SimplePreprossesing(28,28)
iap=ImageToArray()
sdl=SimpleDatasetLoader(preprocessors=[sp,iap])
data=sdl.Load(imagePaths,verbos=100)
data=data.astype('float')/255.0

print('[INFO] loading pre-trained network...')
model=load_model('E:/Code/project/val_loss_mdfy_lenet.hdf5')

print('[INFO] predicting...')
preds=model.predict(data,batch_size=128).argmax(axis=1)


for i,imagePath in enumerate(imagePaths):
    img=cv2.imread('E:/Code/project/numtaDB/testing-new/'+imagePath)
    cv2.putText(img,'Label:{}'.format(classLabels[preds[i]]),
                (0,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(0)

