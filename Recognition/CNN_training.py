#importing libraries
import os
import glob
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as k
from keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


# Loading libraries for KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import DistanceMetric


#libraries to be imported for Detection
import cv2
from mtcnn import MTCNN





class CNN_Training():
    def __init__(self):
        self.CNN()
    


    def CNN(self):
        #Define VGG_FACE_MODEL architecture
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))



        #loading downloaded vgg face model weights
        model.load_weights('vgg_face_weights.h5')


        #model.save("model.h5")
        global vgg_face
        # Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
        vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

        #self.training_Data(vgg_face)


        
    def training_Data(self,vgg_face):
        vgg_face=vgg_face


        #preparing training data

        x_train=[]
        y_train=[]
        person_folders=glob.glob('D:\project\main project\Facial Recognition\Recognition\Training images\*')
        person_names=[]
        for i,folder in enumerate(person_folders):
            temp=[i for i in folder.split(' \ ')]
            temp_2=[j for j in temp[-1].split('\\')]
            person_names.append(temp_2[-1]) #names of persons
            
            #for file names
            for f in glob.glob(folder+'\*'):
                img=load_img(f,target_size=(224,224))
                img=img_to_array(img)
                img=np.expand_dims(img,axis=0)
                img=preprocess_input(img)
                img_encode=vgg_face(img)
                x_train.append(np.squeeze(k.eval(img_encode)).tolist())
                y_train.append(i)
                

        # for i in range(len(person_names)):
        #     print(person_names[i])
        per_dict=dict()
        for i,person in enumerate(person_names):
            per_dict[i]=person
        
        global names
        # names=dict()
        # names=per_dict

        Dict_name='Labels_dictionary.pkl'
        pickle.dump(per_dict, open(Dict_name,'wb'))
            


        x_train=np.array(x_train)
        y_train=np.array(y_train)
        #x_train=pd.DataFrame(x_train)
        # print(x_train)
        # print('successfully Executed')

        #Output face encodings from CNN VGGFace model
        np.save('train_data',x_train)
        #Trained Person names list 
        np.save('train_labels',y_train)
