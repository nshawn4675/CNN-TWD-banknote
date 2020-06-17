'''
build CNN with following techniques:
    1. drop out.
    2. data augmentation.
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image\
    import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers\
    import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def load_data(money_type, pic_size):
    """
    Parameters
    ----------
    money_type : list
        [100, 500, 1000]

    Returns
    -------
    imgs_arr : dict
        {"file_name": img_arr, ...}
    money_imgs : dict
        {100: ['1.jpg', '2.jpg', ...], 500:[...], ...}
    ground_truth : dict
        {'1.jpg': 100, ... , '35.jpg': 500, ...}
    """
    
    imgs_arr = {}
    money_imgs = {}
    ground_truth = {}
    print("loading money data...")
    
    for money in money_type:
        dir_path = '.\\money\\'+str(money)
        imgs_path = [dir_path+'\\'+name for name in os.listdir(dir_path)]
        money_imgs[money] = imgs_path
        for img_path in imgs_path:
            img = mpimg.imread(img_path)
            imgs_arr[img_path] = img_to_array(img)
            ground_truth[img_path] = money
            
    print("done\n")
    return imgs_arr, money_imgs, ground_truth
    

def build_train_test_data_set(MONEY_TYPE, money_imgs, imgs_arr, ground_truth):
    """
    Parameters
    ----------
    MONEY_TYPE : list
        [100, 500, 1000]
    money_imgs : dict
        {100: ['1.jpg', ...], 500: ['21.jpg', ...], ...}
    imgs_arr : dict
        {"file_name": img_arr, ...}
    ground_truth : dict
        {'1.jpg': 100, ... , '35.jpg': 500, ...}
        
    Returns
    -------
    train_data : dict
        {'X': [img_arr1, img_arr2, ...], 'Y':[y1, y2, ...]}
    test_data : TYPE
        {'X': [img_arr1, img_arr2, ...], 'Y':[y1, y2, ...]}
    """
    print("building train/test data set...")
    
    train_data = {'X': [], 'Y': []}
    test_data = {'X': [], 'Y': []}
    money_cls = {100: 0, 500: 1, 1000: 2}
    for money in MONEY_TYPE:
        random.shuffle(money_imgs[money])
        split_idx = int(len(money_imgs[money])/2)
        for file_name in money_imgs[money][:split_idx]:
            train_data['X'].append(imgs_arr[file_name])
            train_data['Y'].append(ground_truth[file_name])
        for file_name in money_imgs[money][split_idx:]:
            test_data['X'].append(imgs_arr[file_name])
            test_data['Y'].append(ground_truth[file_name])
    train_data['X'] = np.array(train_data['X'])
    train_data['Y'] = np.array(train_data['Y'])
    for idx in range(len(train_data['Y'])):
        train_data['Y'][idx] = money_cls[train_data['Y'][idx]]
    train_data['Y'] = to_categorical(train_data['Y'], num_classes = len(MONEY_TYPE))
    
    test_data['X'] = np.array(test_data['X'])
    test_data['Y'] = np.array(test_data['Y'])
    for idx in range(len(test_data['Y'])):
        test_data['Y'][idx] = money_cls[test_data['Y'][idx]]
    test_data['Y'] = to_categorical(test_data['Y'], num_classes = len(MONEY_TYPE))
        
    print("done\n")
    return train_data, test_data


def build_CNN(IMG_SIZE, num_classes):
    """
    Parameters
    ----------
    IMG_SIZE : number
        number of image size.
    num_classes : number
        number of classes.

    Returns
    -------
    model : 
        tf.keras.layer model.
    """
    model = Sequential()
    
    # Convolution
    model.add(Conv2D(32,(5,5),
                     strides=(1,1),
                     input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=None))
    
    model.add(Conv2D(64,(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None))
    
    
    model.add(Conv2D(128,(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None))

    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3,activation='softmax'))
    
    #model.summary()
    return model


if __name__ == "__main__":
    IMG_SIZE = 128
    MONEY_TYPE = [100, 500, 1000]
    
    # load images
    imgs_arr, money_imgs, ground_truth = load_data(MONEY_TYPE, IMG_SIZE)
    
    # build train/test set
    train_data, test_data = build_train_test_data_set(
                                MONEY_TYPE, money_imgs, imgs_arr, ground_truth)

    # data augmentation.
    datagen = ImageDataGenerator(rotation_range=40,
                                 horizontal_flip=True,
                                 rescale=1/255
                                 #width_shift_range=0.2,
                                 #height_shift_range=0.2,
                                 #shear_range=0.2,
                                 #zoom_range=0.2,
                                 #fill_mode='nearest',
                                 )
    datagen.fit(train_data['X'])
    
    # validation data normalization
    test_data['X'] /= 255
    
    # build CNN model
    model = build_CNN(IMG_SIZE, len(MONEY_TYPE))
    # model.summary()
    model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
    #plot_model(model, 'model.png', show_shapes=True)
    
    # train & validate CNN model with augmented training data
    history = model.fit(
        datagen.flow(train_data['X'], train_data['Y'], batch_size = 5),
        #train_data['X'], train_data['Y'],
        epochs=50, validation_data=(test_data['X'], test_data['Y']))
    
    # plot training & validation history.
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    #plt.plot(history.history['loss'], label='loss')
    #plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
