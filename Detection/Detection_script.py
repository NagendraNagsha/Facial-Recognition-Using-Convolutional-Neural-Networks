#libraries to be imported
import cv2
from mtcnn import MTCNN
import tensorflow as tf

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

    def face_Extraction(self,frame):
        #extracting the faces from image
        
        self.frame=frame
        self.cropped=self.frame[self.bounding_box[1]:self.bounding_box[1]+self.bounding_box[3],
        self.bounding_box[0]:self.bounding_box[0]+self.bounding_box[2]]
        """ if count<=10:
            cv2.imwrite('frame%d.jpg' % self.count,self.cropped)
            self.count+=1
            #print(count) """
        return self.cropped


    def detection(self):
        """ Generating bounding box by implementing p-net --> r-net --> o-net
        """
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
                    self.cropped=self.face_Extraction(self.frame)
                    



            #display resulting frame
            cv2.imshow('Hello!',self.frame)
            if cv2.waitKey(1) &0xFF == ord('c'):
                if self.count<=100:
                    cv2.imwrite('frame%d.jpg' % self.count,self.cropped)
                    self.count+=1
            """ if cv2.waitKey(1) &0xFF ==ord('q'):
                break """
                
        
        self.cap.release()
        cv2.destroyAllWindows()
        return None

#creating object and executing detection process
object=MTCNN_Classifier()















