# Loading libraries for KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import DistanceMetric


def knn_Training(self):
        #Load saved data
        x_train=np.load('train_data.npy')
        y_train=np.load('train_labels.npy')


        # Fitting clasifier to the Training set
        # Instantiate learning model (k = 3)
        classifier = KNeighborsClassifier(n_neighbors=3,metric='euclidean',p=2)
        # Fitting the model
        classifier.fit(x_train, y_train)
        

        filename='KNN.sav'
        pickle.dump(classifier,open(filename,'wb'))
