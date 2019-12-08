from keras.preprocessing.image import img_to_array


class ImageToArray:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def Preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)
