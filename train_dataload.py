import cv2


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
        self.data = []
        self.labels = []
        self.itr = 0

    def Load(self, path, csvdata, verbos=-1):
        for i in range(len(csvdata)):
            imagePath = csvdata.filename[i]
            img = cv2.imread(path + '/' + imagePath, 0)
            label = csvdata.digit[i]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    img = p.Preprocess(img)

            self.labels.append(label)
            self.data.append(img)

            if verbos > 0 and i > 0 and (i + 1) % verbos == 0:
                print('[INFO] processed {}'.format(self.itr + i + 1))

        self.itr += i + 1
        return self.data, self.labels
