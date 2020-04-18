#libraries to be imported for Detection
import cv2
from mtcnn import MTCNN





class MTCNN_Classifier():
    """
        perform Face Detection
        -->Bounding Box which contains 
                            x,y coordinates at top left corner,Width and Height
        -->Keypoints(left eye, right eye, nose, mouth_left, mmouth_right)
    """
    def __init__(self):
        """ -->initializing the required data variables
            -->implementing detection algorithm
        """
        self.count=0
        self.cam_port=0
        self.detection()

    def Capture(self):
        #capturing frame by frame
        __,frame=self.cap.read()
        return frame

    def face_Extraction(self,frame,count):
        #extracting the faces from image
        self.count=count
        self.frame=frame
        self.cropped=self.frame[self.bounding_box[1]:self.bounding_box[1]+self.bounding_box[3],
        self.bounding_box[0]:self.bounding_box[0]+self.bounding_box[2]]
        """ if count<=10:
            cv2.imwrite('frame%d.jpg' % self.count,self.cropped)
            self.count+=1
            #print(count) """
        self.y_pred=self.extracting_Face_Encodings(self.cropped)
        return self.y_pred


    
    def extracting_Face_Encodings(self,cropped):
        global vgg_face

        self.cropped=cropped
        self.x_test=[]
        """ temp='D:\\project\\main project\\Facial Recognition\\Recognition\\Test images'
        temp_2=temp+'\\'+'temp_img'+'.jpg'
        cv2.imwrite(temp_2,self.cropped)
        #giving path of saved image
        img=load_img(temp_2,target_size=(224,224))
        os.remove(temp_2) """
        img=cv2.resize(self.cropped, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=preprocess_input(img)
        img_encode=vgg_face(img)
        self.x_test.append(np.squeeze(k.eval(img_encode)).tolist())
        #self.x_test.append(k.eval(img_encode[0]))
        #print(self.x_test)
        self.y_pred=self.Testing(self.x_test)
        return self.y_pred
        

    def Testing(self,x_test):
        self.x_test=x_test
        classifier_model=pickle.load(open('KNN.sav','rb'))
        self.y_pred = classifier_model.predict([x_test[0]])
        global names
        global pred
        pred=classifier_model.predict_proba([x_test[0]])
        print(pred)
        #print(names[int(y_pred)])
        names=pickle.load(open('labels_dictionary.pkl','rb'))
        
        return self.y_pred
        




    def detection(self):
        """ Generating bounding box by implementing p-net --> r-net --> o-net
        """
        global pred_results
        pred_results=[]
        num=0
        #creating detector object for MTCNN class
        self.detector=MTCNN()
        self.cap=cv2.VideoCapture(self.cam_port)
        while True:
            self.frame=self.Capture()
            #Detecting faces from frames
            self.result=self.detector.detect_faces(self.frame)
            if self.result!=[]:
                for face in self.result:
                    self.bounding_box=face['box']
                    self.keypoints=face['keypoints']
                    cv2.rectangle(self.frame,
                                (self.bounding_box[0], self.bounding_box[1]),
                                (self.bounding_box[0]+self.bounding_box[2], self.bounding_box[1] + self.bounding_box[3]),
                                (0,255,0),2)
                    #calling the extraction method to extract face images
                    self.y_pred=self.face_Extraction(self.frame,self.count)
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (self.bounding_box[0],self.bounding_box[1])
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2
                    global names
                    global pred
                    probability=pred[0].max()
                    if int(probability)>=1:
                        cv2.putText(self.frame,'HI'+' '+names[int(self.y_pred)],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
                    else:
                        cv2.putText(self.frame,'UnKnown',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

                    
                    #Finding Accuracy Score
                    global y_test
                    y_test=[1,1,1,1,1,1,1,1]
                    
                    """ for i in names.keys():
                        y_test.append(i) """
                    if num==8:
                        num=0
                        accuracy=accuracy_score(y_test,pred_results)*100
                        y_test=[]
                        pred_results=[]
                        print(accuracy)
                    else:
                        num+=1
                        pred_results.append(int(self.y_pred))
                     

                    




            #display resulting frame
            cv2.imshow('Hello!',self.frame)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        return None


#creating object and executing final detection and Testing process

object=MTCNN_Classifier()



