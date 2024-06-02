import multiprocessing as mp
import copy
import cv2
import numpy as np
from keras.models import load_model
import time
from voice_assistant import VoiceAssistant  # Importar la clase VoiceAssistant
import os

class GestureRecognition():
    def __init__(self, terminate_flag):
        self.terminate_flag = terminate_flag
        self.cap_region_x_begin = 0.5  # punto de inicio/ancho total
        self.cap_region_y_end = 0.8  # punto de inicio/ancho total
        self.threshold = 60  # umbral binario
        self.blurValue = 41  # parámetro de GaussianBlur
        self.bgSubThreshold = 50
        self.learningRate = 0

        self.isBgCaptured = 0  # bool, si el fondo está capturado
        self.camera = cv2.VideoCapture(0)
        self.camera.set(10, 200)

        self.bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
        time.sleep(2)
        self.isBgCaptured = 1
        print('Background captured')

        self.model = load_model('C:/Users/zS22000728/Documents/VS/InterfacesNaturalesDeUsuario/proyecto_amorcita/saved_model.hdf5')

        self.gesture_names = {0: 'A', 1: 'S', 2: 'L'}

        self.a_count = 0  # Contador para el gesto 'A'
        self.s_count = 0  # Contador para el gesto 'S'
        self.voice_assistant = VoiceAssistant(terminate_flag)  # Crear una instancia de VoiceAssistant

    def predict_rgb_image_vgg(self, image):
        image = np.array(image, dtype='float32')
        image /= 255
        pred_array = self.model.predict(image)
        result = self.gesture_names[np.argmax(pred_array)]
        score = float("%0.2f" % (max(pred_array[0]) * 100))
        return result, score

    def remove_background(self, frame):
        fgmask = self.bgModel.apply(frame, learningRate=self.learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    def gesture_recognition(self, terminate_flag):
        while self.camera.isOpened():
            ret, frame = self.camera.read()
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # filtro de suavizado
            frame = cv2.flip(frame, 1)  # voltear la imagen

            cv2.rectangle(frame, (int(self.cap_region_x_begin * frame.shape[1]), 0),
                          (frame.shape[1], int(self.cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

            cv2.imshow('original', frame)

            if self.isBgCaptured == 1:
                img = self.remove_background(frame)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)

                ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                thresh1 = copy.deepcopy(thresh)
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                length = len(contours)
                maxArea = -1
                if length > 0:
                    for i in range(length):  # encontrar el contorno más grande (según el área)
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

            k = cv2.waitKey(10)
            if k == 27 or terminate_flag.is_set():  # presionar ESC o señal para salir de todas las ventanas en cualquier momento
                break
            elif k == 32:
                # Si se presiona la barra espaciadora
                cv2.imshow('original', frame)
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = self.predict_rgb_image_vgg(target)
                if prediction == 'A':
                    self.a_count += 1
                    if self.a_count == 1:
                        print("Realizar acción para el gesto A")
                        # Detener la reproducción actual
                        self.voice_assistant.ss.stop
                        self.voice_assistant.reask_routine = True
                elif prediction == 'S':
                    self.s_count += 1
                    if self.s_count == 1:
                        print("Realizar acción para el gesto S")
                elif prediction == 'L':
                    print("Realizar acción para el gesto L")

        self.camera.release()
        cv2.destroyAllWindows()