#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:31:20 2023

@author: hp
"""
import mtcnn
import os
import pandas as pd
import matplotlib.pyplot as plt
import face_recognition
from skimage import exposure
import numpy as np
from PIL import Image
import api
from numpy import asarray
import math
from itertools import combinations
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2

DIM = 128
RGB = 3

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignment_procedure(img, left_eye, right_eye):
    #this function aligns given face in img based on left and right eye coordinates
     
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
     
    #-----------------------
    #find rotation direction
     
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
     
    #-----------------------
    #find length of triangle edges
     
    a = euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = euclidean_distance(np.array(right_eye), np.array(left_eye))
     
    #-----------------------
     
    #apply cosine rule
     
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
     
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
     
        #-----------------------
        #rotate base image
     
        if direction == -1:
            angle = 90 - angle
     
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
     
    #-----------------------
     
    return img #return img anyway

def alignFaceMTCNN(img):
    align_faces=[]    
    detections = detector.detect_faces(img)
    #check dua wajah, nanti pakai face sj
    for detection in detections:
        score = detection["confidence"]        
        if score > 0.9:            
            x, y, w, h = detection["box"]
            detected_face = img[abs(y):abs(y+h), abs(x):abs(x+w)]
            keypoints = detection["keypoints"]
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]            
            align_face = alignment_procedure(detected_face, left_eye, right_eye)                                                                         
            if len(align_face)!=0:
                align_face = resizeImg(align_face) 
                align_faces.append(align_face)
                print(score)
                # plt.imshow(detected_face)
                # plt.show()                      
        # break
    return align_faces

def resizeImg(img,required_size=(224, 224)):
    face_image = Image.fromarray(img)
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)
    
    return face_array

def pushToArrImg(align_faces,face_images):
    no_align_faces=True
    for align_face in align_faces:
        no_align_faces=False
        face_array = resizeImg(align_face) 
        face_images.append(face_array)                                              
    return face_images,no_align_faces

def getFace(img,face_location):
    top, right, bottom, left = face_location
    # plt.imshow(img[top:bottom, left:right])
    # plt.show()                                                  
    detected_face  = img[top:bottom, left:right] 
    return detected_face

def repairRotation(img):
    print("recostruct image...(rotasi 90")
    img  = Image.fromarray(img)
    img  = np.array(img.rotate(90))
    plt.imshow(img)
    plt.show()
    return img

def repairWithClahe(img):
    print("recostruct image...hist")
    img  = exposure.equalize_adapthist(img/255)
    img  = img*255
    img  = img.astype(np.uint8)  
    plt.imshow(img)
    plt.show()
    return img

def rekonstruksiImage(img):
    img2=repairWithClahe(img)
    face_bounding_boxes = face_recognition.face_locations(img2)   
    if len(face_bounding_boxes)==0:
        img2=repairRotation(img)
        face_bounding_boxes = face_recognition.face_locations(img2)
        if len(face_bounding_boxes)==0:
            img2=img
    return img2,face_bounding_boxes

# def extract_face_from_image(image_path, required_size=(224, 224),needAlign=True):
#   # load image and detect faces
#     noFace=True
#     face_images = []
#     img = plt.imread(image_path)
#     face_bounding_boxes = face_recognition.face_locations(img)   
#     if len(face_bounding_boxes)==0:
#         print("wajah tidak terdeteksi")
#         img,face_bounding_boxes = rekonstruksiImage(img)    
#     for face_location in face_bounding_boxes:
#         detected_face = getFace(img,face_location)
#         # plt.imshow(detected_face)
#         # plt.show()
#         if needAlign:
#             align_faces    = api.face_alignment(detected_face)            
#             face_images,no_align_faces = pushToArrImg(align_faces,face_images)
#             if no_align_faces==False:
#                 noFace=False                
#         else:
#             face_array = resizeImg(detected_face) 
#             face_images.append(face_array)
#             noFace=False
#     if noFace==True :
#         print("face tdk dapat di align, try with mtcnn")                                  
#         align_faces = alignFaceMTCNN(img)
#         face_images,no_align_faces = pushToArrImg(align_faces,face_images)
            

#     return face_images
def loadImageDetectFace(path, face_images, fr=False):
    ftr,files,idfiles,scoreFace,idNumber  =[],[],[],[],0    
    for file in os.listdir(path):
        # print("nmFile ...." + file)
        # extracted_faces = face_images
        # extracted_face = face_images(path, required_size=(224, 224))
        extracted_face = face_images
        if fr == True:
            for detected_face in extracted_face:
                fe = getEncoding(detected_face)
                if len(fe) > 0:
                    scoreFace.append(fe)
                    ftr.append(detected_face)
                    files.append(file)
                    idfiles.append(idNumber)
                    idNumber += 1
        else:
            model_scores = get_model_scores(extracted_face)
            for i in range(len(model_scores)):
                scoreFace.append(model_scores[i])
                ftr.append(extracted_face[i])
                files.append(file)
                idfiles.append(idNumber)
                idNumber += 1
                
    comb = combinations(idfiles, 2)
    similars = np.zeros(len(idfiles))
    for c in comb:
        matches = False
        if fr == True:
            matches = face_recognition.compare_faces(
                np.array(scoreFace[c[0]]), np.array(scoreFace[c[1]])
                
            )
        else:
            diff = cosine(scoreFace[c[0]], scoreFace[c[1]])
            if diff <= 0.4:
                matches = True

        if matches:
            similars[c[0]] += 1
            similars[c[1]] += 1

    df = pd.DataFrame({"id": np.array(idfiles), "mirip": similars})
    df = df.sort_values(by=["mirip"], ascending=False)
    df = df[df["mirip"] > 1]
    df.reset_index(drop=True, inplace=True)

    return df

    #get base on 3 folder->3 face sj
    # selectedFtr,selectedLbl,selectedUsia,selectNm=[],[],[],[]
    # selectedFtr, selectNm = [], []
    # cntFace = 0
    # for d in range(len(df)):
    #     idfile = df['id'].loc[d]
    #     if files[idfile] not in selectNm:
    #         selectedFtr.append(ftr[idfile])
    #         # selectedLbl.append(jk)
    #         # selectedUsia.append(umur)
    #         selectNm.append(files[idfile])
    #         cntFace += 1
    #     if cntFace >= 3:
    #         break

    # # return selectedFtr,selectedLbl,selectedUsia,selectNm
    # return selectedFtr, selectNm


def getEncoding(detected_face):
    face_endcoding = face_recognition.face_encodings(detected_face)                    
    if len(face_endcoding)==0:
        print("wajah tidak dapat di encoding")        
        plt.imshow(detected_face)
        plt.show()      
    return face_endcoding  

def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

    # perform prediction
    return model(samples)

if __name__ == '__main__':
    path = "lfw\Abdullah_Gul"
    detector = mtcnn.MTCNN()
    idfile = 0
    for dirpath, subdirs, files in os.walk(path):
        # print(files)
        for file in files:
            if ".jpg" in file:
                # print(file)
                # continue
                idfile = idfile + 1
                # print(file)
                image_path = os.path.join(dirpath, file)
                needAlign = True

                noFace = True
                face_images = []
                print(image_path)
                img = plt.imread(image_path)
                face_bounding_boxes = face_recognition.face_locations(img)

                # deteksi menggunakan face_recognition
                if len(face_bounding_boxes) == 0:
                    print("wajah tidak terdeteksi")
                    img, face_bounding_boxes = rekonstruksiImage(img)
                for face_location in face_bounding_boxes:
                    detected_face = getFace(img, face_location)
                    plt.imshow(detected_face)
                    plt.show()
                    if needAlign:
                        align_faces = api.face_alignment(detected_face)
                        face_images, no_align_faces = pushToArrImg(
                            align_faces, face_images)
                        if no_align_faces == False:
                            noFace = False
                    else:
                        face_array = resizeImg(detected_face)
                        face_images.append(face_array)
                        noFace = False
                # face recognition gagal, digunakan MTCNN
                if noFace == True:
                    print("face tdk dapat di align, try with mtcnn")
                    align_faces = alignFaceMTCNN(img)
                    face_images.extend(align_faces)
                    face_images, no_align_faces = pushToArrImg(
                        align_faces, face_images)
                    
                # Load images and calculate similarity scores
                # print(image_path)
                similarity_df = loadImageDetectFace(path, face_images)

                # # Print similarity results
                print(similarity_df)
                continue

            if idfile > 0:
                break

        if idfile > 0:
            break
