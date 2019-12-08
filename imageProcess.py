import cv2
image=cv2.imread('E:/Code/htmljscss/imageData/e00020.png', 0)
image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
#image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite('E:/Code/htmljscss/imageData/e.png', image)
cv2.imshow('Image', image)
cv2.waitKey(0)