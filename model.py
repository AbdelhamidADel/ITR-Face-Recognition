# Libraries 
import numpy as np
import cv2
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.pipeline import make_pipeline
import pickle
import urllib.request

# Model
with open('saved_model.pkl','rb') as file:
    model = pickle.load(file)

# Pywavelet
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# cropping images 
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

#url
def url(user_input):
    try:
        image_url=user_input
        save_name = 'test.jpg' 
        urllib.request.urlretrieve(image_url, save_name)
        cropped_img=get_cropped_image_if_2_eyes(save_name)
        scalled_raw_img = cv2.resize(cropped_img, (32, 32))
        img_har = w2d(cropped_img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        url_result=np.array(combined_img).reshape(1, 4096)
        final_pred=model.predict(url_result)
        if final_pred == 0:
            url_name="She is Alexia Putellas"

        elif final_pred == 1:
            url_name="He is Gianluigi Donnarumma"

        elif final_pred == 2:
            url_name="He is Lionel Messi"

        elif final_pred == 3:
            url_name="He is Pedri"

        elif final_pred == 4:
            url_name="He is Robert Lewandowski"

        return url_name
    except:
        return 'choose another picture'