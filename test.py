import numpy as np
from keras.models import load_model
import bangla
import cv2

model=load_model('val_loss32.hdf5')
x = cv2.imread('image_1.png', 0)
# x = cv2.imread('E:/Code/htmljscss/testing-new/' +filename , 0)

# x = np.invert(x)
# x=cv2.Canny(x,127,255)
# x= cv2.threshold(x, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
gauss = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 15)
_, x = cv2.threshold(gauss, 90, 255, cv2.THRESH_BINARY_INV)

x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA)
x = x.reshape(1, 32, 32, 1)
out = model.predict(x)
print(out)
digit = np.argmax(out, axis=1)
print(digit)
digit=digit[0]
print(bangla.convert_english_digit_to_bangla_digit(digit))



