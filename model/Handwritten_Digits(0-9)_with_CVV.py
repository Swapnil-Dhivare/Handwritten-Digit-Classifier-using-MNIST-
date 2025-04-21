import warnings
warnings.filterwarnings("ignore")
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd 
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle


from tensorflow.keras import datasets, layers, models
from image_preprocessing import preprocessing

def load_data():
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths to the data files
    file_path_train = os.path.join(current_dir, 'data', 'mnist_train.csv')
    file_path_test = os.path.join(current_dir, 'data', 'mnist_test.csv')
    
    if os.path.exists(file_path_train) and os.path.exists(file_path_test):
        train_data = pd.read_csv(file_path_train)
        test_data = pd.read_csv(file_path_test)
        
        X_train = train_data.drop(['label'], axis=1)
        y_train = train_data['label']
        X_test = test_data.drop(['label'], axis=1)
        y_test = test_data['label']
        
        return (X_train, y_train, X_test, y_test)
    else:
        print(f"Looking for files at:\n{file_path_train}\n{file_path_test}")
        raise FileNotFoundError("One or Both MNIST dataset files are missing")

def Normalize_data(X_train,X_test):
    #Instead of using StandardScaler() we can just do x/255.0 as typically images contain values from 0 to 255
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0


    return (X_train_scaled,X_test_scaled)

def Build_model(train_generator,X_test,y_test):
    import warnings
    warnings.filterwarnings("ignore")
    
    """
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    X_test = X_test.values.reshape(-1, 28, 28, 1)
    """


    model = models.Sequential()
    model.add( layers.Conv2D(32, (3,3),activation = "relu" ,input_shape= (28,28,1)) )
    model.add( layers.MaxPooling2D((2,2)))
    
    model.add( layers.Conv2D(64, (3,3),activation = "relu" ) )
    model.add( layers.MaxPooling2D((2,2)))
    
    #print(model.summary())
    
    # Now our output shape is 3D, we have to flatten it to 1d so that we can feed it as input into our dense layer
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation="softmax"))
    
    print(model.summary())
    
    #Compile and Train The Model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = "adam",
        metrics = ["accuracy"])
    
    model.fit(train_generator, batch_size=128, epochs=15, validation_data = (X_test,y_test))
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    model.save('model.h5')
    
    
        
    return model

def plot(X_train,y_train):
    class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        #Each MNIST image is 28×28 pixels, but in the dataset, it's flattened into a 1D array of 784 values (since 28×28 = 784).
        #passing the data to plt.imshow(), reshape it into a 2D (28, 28) format
        plt.imshow(X_train[i].reshape(28,28),cmap="gray")
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[int(y_train[i].item())]) 
        #Ensure that y_train[i] is an integer by converting it explicitly using .item()
    plt.show()   

def Augment(X_train,y_train):
    datagen = ImageDataGenerator(
    rotation_range=15,          # Rotate the image by up to 15 degrees
    width_shift_range=0.1,      # Shift the width by up to 10% of the image width
    height_shift_range=0.1,     # Shift the height by up to 10% of the image height
    zoom_range=0.1,             # Zoom in by up to 10%
    horizontal_flip=False        # Since MNIST digits are symmetric
    )
    
    #fit the ImageDataGenerator to the training data
    datagen.fit(X_train)
    
    '''
    # Let's visualize some augmented images
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
        # Create a grid of 3x3 images
        fig, ax = plt.subplots(3, 3, figsize=(8, 8))
        for i in range(9):
            ax[i//3, i%3].imshow(X_batch[i].reshape(28, 28), cmap='gray')
            ax[i//3, i%3].axis('off')
        plt.suptitle('Augmented Images')
        plt.show()
        break  # Only show one batch
    '''
    return datagen.flow(X_train, y_train, batch_size=128)

     


     
def main():
    model_path = "model.h5"
    
    if not os.path.exists(model_path):
        try:
            X_train, y_train, X_test, y_test = load_data()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

        X_train, X_test = Normalize_data(X_train, X_test)
        train_generator = Augment(X_train, y_train)

        model = Build_model(train_generator, X_test, y_test)
        


if __name__ == "__main__":
    main()
