import numpy as numpy
import os
import cv2
import configs
import imgaug as ia
import pandas as pd
from keras.utils import Sequence
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image

random_seed = 123


ia.seed(random_seed)

def getTrainDataset(opts):
    train_path = os.path.join(opts['BASE_DIR'], 'train')
    data = pd.read_csv(os.path.join(opts['BASE_DIR'],'train.csv'))

    paths = []
    labels = []

    for data_id, data_labels in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(opts['num_classes'])
        for each_label in data_labels:
            y[int(each_label)] = 1
        paths.append(os.path.join(train_path, data_id))
        labels.append(y)
    
    return np.array(paths), np.array(labels)

def getTestDataset(opts):
    train_path = os.path.join(opts['BASE_DIR'], 'test')
    data = pd.read_csv(os.path.join(opts['BASE_DIR'],'sample_submission.csv'))

    paths = []
    labels = []

    for data_id in data['Id']:
        y = np.ones(opts['num_classes'])
        paths.append(os.path.join(train_path, data_id))
        labels.append(y)
    
    return np.array(paths), np.array(labels)

def get_train_test_generator(opts):
    random_seed = opts['random_seed']
    val_ratio = opts['val_ratio']
    batch_size = opts['batch_size']

    paths, labels = getTrainDataset(opts)
    # divide to 
    keys = np.arange(paths.shape[0], dtype=np.int)  
    np.random.seed(random_seed)
    np.random.shuffle(keys)
    lastTrainIndex = int((1-val_ratio) * paths.shape[0])

    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]

    test_paths, test_labels = getTestDataset(opts)

    tg = ProteinDataGenerator(opts, pathsTrain, labelsTrain, batch_size, opts['SHAPE'], use_cache=False, augment = False, shuffle = False)
    vg = ProteinDataGenerator(opts, pathsVal, labelsVal, batch_size, opts['SHAPE'], use_cache=False, shuffle = False)

    return tg, vg, paths, labels, test_paths, test_labels

class ProteinDataGenerator(Sequence):
    def __init__(self, opts, paths, labels, batch_size, shape, shuffle=False,\
            use_cache = False, augment = True):
        self.opts = opts
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.augment = augment
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))

        # generate data here
        if self.use_cache == True:
            X = self.cache[indexes]
            for index, each_path in enumerate(paths[np.where(self.is_cached[indexes]==0)]):
                image = self.__load_image(each_path)
                self.is_cached[indexes[index]] = 1
                self.cache[indexes[index]] = image
                X[index] = image
        else:
            for index, each_path in enumerate(paths):
                X[index] = self.__load_image(each_path)

        y = self.labels[indexes]

        if self.augment == True:
            seq = iaa.Sequence([
                iaa.OneOf([
                    # horizontal filps
                    iaa.Fliplr(0.5), 
                    # random crops
                    iaa.Crop(percent=(0, 0.1)),
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5, iaa.GuassianBlur(sigma=(0, 0,5))),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={'x':(0.8, 1.2), 'y':(0.8, 1.2)},
                        translate_percent={'x':(-0.2, 0.2), 'y':(-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
        
            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), axis=0)
            y = np.concatenate((y, y, y, y), axis=0)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        #create a generator that iterate over the Sequence
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R), 
            np.array(G), 
            np.array(B),
            np.array(Y)), -1)
        
        im = cv2.resize(im, (self.opts['SHAPE'][0], self.opts['SHAPE'][1]))
        im = np.divide(im, 255)
        return im
