import copy
import cv2
import numpy as np
from keras.models import load_model
import time
from alexav3 import SyntaxAnalyzer, AudioPlayer
import pywhatkit
from datetime import datetime
import subprocess
import pygame
import time


# General Settings
prediction = ''
action = ''
score = 0

gesture_names = {0: 'google',
                 1: 'explorador',
                 2: 'word',
                 3: 'excel',
                 4: 'powerpoint',
                 5: 'volumeup',
                 6: 'volumelow'}

model = load_model('C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/accessible-_-jarvis/saved_model.hdf5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    if result == 'google':
        subprocess.Popen("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
        audio_player = AudioPlayer()
        audio_player.play("Google se ha abierto") 
    elif result == 'word':
        audio_player = AudioPlayer()
        audio_player.play("Word se ha abierto")
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.exe")
    elif result == 'excel':
        audio_player = AudioPlayer()
        audio_player.play("Excel se ha abierto")
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.exe")
    elif result == 'powerpoint':
        audio_player = AudioPlayer()
        audio_player.play("PowerPoint se ha abierto")
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.exe")
    elif result == 'volumeup':
        audio_player = AudioPlayer()
        audio_player.play("Up")
    elif result == 'volumelow':
        audio_player = AudioPlayer()
        audio_player.play("low")
        pygame.mixer.init()
        # Disminuir el volumen en un 10%
        pygame.mixer.music.set_volume(-0.1)
    return result, score


# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works


def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    # Run once background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('ori', thresh) #mandar este frame para capturar el gesto en im

        # get the contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
    elif k == ord('r'):  # press 'r' to reset the background
        time.sleep(1)
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('Reset background')
    elif k == 32:
        # If space bar pressed
        cv2.imshow('original', frame)
        # copies 1 channel BW image to all 3 RGB channels
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_rgb_image_vgg(target)
    
