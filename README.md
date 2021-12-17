# HANDWRITTEN BANGLA DIGIT RECOGNITION USING CNN WITH WEB APPLET:

Bangla handwritten digit recognition is an efficient starting point for building an Optical Character Reader in the Bengali language. Lack of large dataset, Bangla digit recognition was not standardized previously. Handwritten digit recognition complexity varies among various languages due to distinct shapes and numbers of a digit. Recently, Convolutional Neural Network (CNN) is found efficient for English handwritten both digit and character recognition and parallelly Bangla handwritten both digit and character recognition. In this paper, a CNN based Bangla handwritten digit recognition is investigated. The proposed model normalizes the digit images, different kinds of image preprocessing techniques are used for processing images and then finally employ CNN to classify individual digit. The CNN model has shown excellent performance.

# Dataset:
The sample image of digit are:

![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/a00007.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/b00006.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/c00022.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/e00020.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/d00754.png)

# Data Preprocessing:

The original size of NumtaDB is 180 by 180 pixels which are too complex for preprocessing efficiently. So we reduce the size of images to 32 by 32. We also convert all RGB images to GRAY scale images. The color channel converted from three to one.

Images lose many potential data due to image resizing. Inter-area interpolation is the best method for image decimation. We use inter-area interpolation after image resizing.

![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/a.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/b.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/c.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/e.png)
![Alt text](https://github.com/shamim-ice/HandwrittenBanglaDigitRecognition/blob/master/d.png)
