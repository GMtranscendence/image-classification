import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Flatten, Activation

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import smart_resize

import numpy as np
import cv2 as cv
import splitfolders

def load_data(image_size, batch_size):

    directory = 'flower_photos_selected'
    splitfolders.ratio(directory, output='dataset', seed=1337, ratio=(.7, .2, .1))
    

    train_augmentation = ImageDataGenerator(
        
        rescale=1./255,
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True 
    
    )
    val_augmentation = ImageDataGenerator(rescale=1./255)

    train_ds = train_augmentation.flow_from_directory(
        
        "dataset/train",
        seed=1337,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'

    )
    val_ds =val_augmentation.flow_from_directory(

        "dataset/val",
        seed=1337,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    
    )

    return (train_ds, val_ds) 

def train(x_train, x_validation):

    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
     
    
    model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation = "softmax"))
    model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train, validation_data=x_validation, epochs=21, verbose=2)
    model.save('models/model2')

    return model

def test(model, x_test):

    x_test = tf.keras.preprocessing.image_dataset_from_directory(
        
        "dataset/val",
        seed=1337,
        image_size=(150,150),
        batch_size=32,
        labels='inferred',
        label_mode='categorical'

    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    x_test = x_test.map(lambda x, y: (normalization_layer(x), y))

    labels = np.argmax([y for _, y in x_test], axis=-1).flatten()
#    print(labels)

    prediction = model.predict(x_test)
    pred_digits = np.argmax(prediction, axis=1)
#    print(pred_digits)

    matrix = confusion_matrix(labels, pred_digits)
    report = classification_report(labels, pred_digits, target_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])

    print('-------Evaluation--------')
    print(f'\nconfusion matrix: \n{matrix}\n')
    print(f'{report}')
    print('-------------------------')

    return matrix, report

def infer(model, image_path):

    LABEL_NAMES = {'0': 'daisy', '1': 'dandelion', '2': 'rose', '3': 'sunflower', '4': 'tulip'} 
    
    image = load_img(image_path)
    image = cv.imread(image_path)
    img = np.array(image)
    label = image_path.split('/')[2]
    img = smart_resize(img, (150,150))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred_digit = np.argmax(prediction, axis=1)
    print(f'actual: {label}\npredicted: {LABEL_NAMES[str(pred_digit[0])]}')

if __name__ == '__main__':

    # data info
    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 32
    MODES = {'train': 1, 'test': 2, 'infer': 3}
    LABEL_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'] 

    le = LabelEncoder()
    labels = le.fit_transform(LABEL_NAMES)
    labels = to_categorical(labels, 5)
#    mode = MODES['train']
#    mode = MODES['test']
    mode = MODES['infer']

    # load data
    train_ds, validation_ds = load_data(IMAGE_SIZE, BATCH_SIZE) 

    try:
        model = keras.models.load_model('models/model2')
    except OSError:
        print('Incorrect model path')

    if mode == 1:
        train(train_ds, validation_ds)
    elif mode == 2:
        test(model, validation_ds)
    elif mode == 3:
        infer(model, 'dataset/test/sunflowers/6199086734_b7ddc65816_m.jpg')













