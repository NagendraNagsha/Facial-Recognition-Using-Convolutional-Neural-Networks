import cv2
import glob
import os
from PIL import Image
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

detector=MTCNN()
folders_path=glob.glob('D:\project\main project\Facial Recognition\Recognition\images\*')


#Operation to find person names
image_names_list=[]
folder_names_list=[]
person_names=[]
new=[]
file_names=[]   
file_names_list=[]
for folder in folders_path:
    temp=[i for i in folder.split(' \ ')]
    temp_2=[j for j in temp[-1].split('\\')]
    person_names.append(temp_2[-1]) #names of persons
    
    for f in glob.glob(folder+'\*'):
        file_names_list.append(f)
        image_names_list.append(f)



    


#code for filenames
new=[]
for i in file_names_list:
    temp=[j for j in i.split(' \ ')]
    new.append(temp[0])
new_files=[]
for i in new:
    new_files.append(i)
for i in new_files:
    temp=[j for j in i.split(' \ ')]
    temp_2=[k for k in temp[0].split('\\')]
    temp_3=temp_2[-1].split('.')
    file_names.append(temp_3[0])#file names were stored in files list


read_images=[]
for image in image_names_list:
    img=cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_CUBIC)
    read_images.append(img)
#person names and filenames
# print(file_names)
# print(person_names)




res_img=[]
for i in range(len(read_images)):
    print(read_images[i])
    
    result=detector.detect_faces(read_images[i])
    print(result,i)
    if result!=[]:
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']
        #print(bounding_box)#contains [top_left_corner  top_right_corner  width  height]
        #print(keypoints)
        #Radius of the scale and thickness of the line
        cv2.rectangle(read_images[i],(bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]),(0,155,255),2)
        cropped=read_images[i][bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
        read_images[i]=cropped
        """ #resizing the image
        scale_percent=60
        width=int(read_images[i].shape[0]*scale_percent/100)
        height=int(read_images[i].shape[1]*scale_percent/100)
        dim=(width,height)
        resized=cv2.resize(read_images[i],dim,interpolation=cv2.INTER_AREA)
        read_images[i]=resized """
        

    res_img.append(result)


count=0
#saving the cropped images in training folders
for i in range(len(person_names)):
    temp='D:\\project\\main project\\Facial Recognition\\Recognition\\Training images'
    os.chdir(temp)
    os.mkdir(person_names[i])
    new_path=temp+'\\'+person_names[i]
    for i in range(15):
        temp_2=new_path+'\\'+file_names[count]+'.jpg'
        cv2.imwrite(temp_2,read_images[count])
        count+=1







""" for i in range(len(read_images)):
    #print(result)
    cv2.imshow('image',read_images[i])
    cv2.waitKey(0)  """






















