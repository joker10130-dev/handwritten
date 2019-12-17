# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:32:49 2019

@author: User
"""

import cv2
import numpy as np
from keras.models import load_model

# load model
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

labels=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
cam = cv2.VideoCapture(0)

cv2.namedWindow("handwritten")

img_counter = 0

while True:
    
    ret, frame = cam.read()
    
    h, w, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fx, fy = w//6, h//6
    rec = cv2.rectangle(frame, (100,100), (150, 150),(0,255,0),2)
    rect_img = frame[100:150,100:150]
    frame[100 : 150, 100 : 150] = rect_img
    cv2.imshow("handwritten", frame)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 99:
        # SPACE pressed
        img_name = "handwritten-{}.png".format(img_counter)
        img = cv2.adaptiveThreshold(rect_img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.bitwise_not(img)
        
        
        img = cv2.resize(img, (28,28))
        img = np.reshape(img, [1, 28, 28, 1])
        prob = model.predict(img)
        pred = int(np.argmax(prob, axis=1))
        response = labels[pred]
        print("predict: "+response)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
    
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its value

cam.release()

cv2.destroyAllWindows()