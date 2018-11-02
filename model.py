import configs
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from loss import f1_loss, f1
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score as off1
from datahandler import get_train_test_generator, ProteinDataGenerator, getTestDataset, getTrainDataset

class GAPNetModel():
    def __init__(self, opts):
        self.opts = opts
        self.train_gen, self.val_gen, self.train_paths, self.train_labels, self.test_paths,\
                                                    self.test_labels = get_train_test_generator(self.opts)

        init = Input(self.opts['SHAPE'])
        x = BatchNormalization(axis=-1)(init)
        x = Conv2D(32, (3, 3), strides=(2, 2))(x)
        x = ReLU()(x)

        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        feature_map_1 = Dropout(self.opts['drop_ratio'])(x)

        x = BatchNormalization(axis=-1)(feature_map_1)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = ReLU()(x)

        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)

        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        feature_map_2 = Dropout(self.opts['drop_ratio'])(x)
        
        x = BatchNormalization(axis=-1)(feature_map_2)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3, 3))(x)
        x = ReLU()(x)
        feature_map_3 = Dropout(self.opts['drop_ratio'])(x)

        gap1 = GlobalAveragePooling2D()(feature_map_1)
        gap2 = GlobalAveragePooling2D()(feature_map_2)
        gap3 = GlobalAveragePooling2D()(feature_map_3)

        x = Concatenate()([gap1, gap2, gap3])
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(28)(x)
        x = Activation('sigmoid')(x)
        
        self.model = Model(init, x)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy', f1])

    def run(self, epochs=30, use_multiprocessing=False, workers=1, verbose=1):
        save_model_path = self.opts['save_model_path']
        batch_size = self.opts['batch_size']
        checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss',verbose=verbose,\
            save_best_only=True, save_weights_only=False, mode='min', period=1)
        # reduce_lr not used in current version
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, model='min')
        hist = self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=len(self.train_gen),
            validation_data=self.val_gen,
            validation_steps=8,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=1,
            callbacks=[checkpoint]
        )

        # fine-tuning part
        for each_layer in self.model.layers:
            each_layer.trainable = False

        self.model.layers[-1].trainable = True
        self.model.layers[-2].trainable = True
        self.model.layers[-3].trainable = True
        self.model.layers[-4].trainable = True
        self.model.layers[-5].trainable = True
        self.model.layers[-6].trainable = True
        self.model.layers[-7].trainable = True

        self.model.compile(loss=f1_loss, optimizer=Adam(lr=1e-4), metrics=['accuracy', f1])
        hist = self.model.fit_generator(
            self.train_gen,
            steps_per_epoch=len(self.train_gen),
            validation_data=self.val_gen,
            validation_steps=8,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=1,
            max_queue_size=4,
        )

        last_train_index = int((1-self.opts['val_ratio']) * self.train_paths.shape[0])

        best_model = load_model(save_model_path, custom_objects={'f1':f1})

        full_var_gen = ProteinDataGenerator(self.opts, self.train_paths[last_train_index:], \
                                self.train_labels[last_train_index:], batch_size, self.opts['SHAPE'])

        t_now, f1_now = self.getOptimalT(self.model, full_var_gen)
        t_best, f1_best = self.getOptimalT(best_model, full_var_gen)

        if f1_now > f1_best:
            self.T = t_now
            self.best_model = best_model
        else:
            self.T = t_best
            self.best_model = self.model
        return
    
    def predict(self):
        batch_size = self.opts['batch_size']
        test_gen = ProteinDataGenerator(self.opts, self.test_paths, self.test_labels, batch_size, self.opts['SHAPE'])
        submit = pd.read_csv(self.opts['base_dir'] + '/sample_submission.csv')
        pred = np.zeros((self.test_paths.shape[0], 28))

        for i in tqdm(range(len(test_gen))):
            images, labels = test_gen[i]
            score = self.best_model.predict(images)
            pred[i*batch_size:i*batch_size+score.shape[0]] = score

        PP = np.array(pred)
        prediction = []

        for row in tqdm(range(submit.shape[0])):
            
            str_label = ''
            
            for col in range(PP.shape[1]):
                if(PP[row, col] < self.T[col]):
                    str_label += ''
                else:
                    str_label += str(col) + ' '
            prediction.append(str_label.strip())
    
        submit['Predicted'] = np.array(prediction)
        submit.to_csv('predict.csv', index=False)

    def get_best_model(self):
        return load_model(self.opts['save_model_path'], custom_objects={'f1':f1})  

    def getOptimalT(self, model, val_gen):
        last_full_val_pred = np.empty((0, 28))
        last_full_val_labels = np.empty((0, 28))

        for i in tqdm(range(len(val_gen))):
            data_im, data_label = val_gen[i]
            scores = model.predict(data_im)
            last_full_val_pred = np.append(last_full_val_pred, scores, axis=0)
            last_full_val_labels = np.append(last_full_val_labels, data_label, axis=0)
        print(last_full_val_pred.shape, last_full_val_labels.shape)

        rng = np.arange(0, 1, 0.001)
        f1s = np.zeros((rng.shape[0], 28))
        for j,t in enumerate(tqdm(rng)):
            for i in range(28):
                p = np.array(last_full_val_pred[:,i]>t, dtype=np.int8)
                #scoref1 = K.eval(f1_score(fullValLabels[:,i], p, average='binary'))
                scoref1 = off1(last_full_val_labels[:,i], p, average='binary')
                f1s[j,i] = scoref1
                
        print(np.max(f1s, axis=0))
        print(np.mean(np.max(f1s, axis=0)))

        T = np.empty(28)
        for i in range(28):
            T[i] = rng[np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0]]
        #print('Choosing threshold: ', T, ', validation F1-score: ', max(f1s))
        print(T)
        return T, np.mean(np.max(f1s, axis=0))
