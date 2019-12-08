import numpy as np
import cv2


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
        self.data = []
        self.labels = []

    def Load(self, imagePaths, verbos=-1):
        for (i, imagePath) in enumerate(imagePaths):
            img = cv2.imread('E:/Code/project/numtaDB/testing-new/' + imagePath, 0)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    img = p.Preprocess(img)
            self.data.append(img)


            if verbos > 0 and i > 0 and (i + 1) % verbos == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))

        return np.array(self.data)
