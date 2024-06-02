import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import speech_recognition as sr
import nltk
import pygame
from gtts import gTTS
import os
import pywhatkit
import pygetwindow as gw
import ctypes

# Configuración general
prediction = ''
action = ''
score = 0
a_count = 0  # Contador para el gesto 'A'
s_count = 0  # Contador para el gesto 'S'

gesture_names = {0: 'A', 1: 'S', 2: 'L'}

model = load_model('C:/Users/zS22000728/Documents/VS/InterfacesNaturalesDeUsuario/proyecto_amorcita/saved_model.hdf5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

class SpeechRecognizer():
    def __init__(self):
        self.__r = sr.Recognizer()
        self.__mic = sr.Microphone()

    def recognize(self):
        with self.__mic as source:
            print("Ajustando al ruido de fondo...")
            self.__r.adjust_for_ambient_noise(source)
            print("Dime algo:")
            audio = self.__r.listen(source)

        try:
            recognized_speech = self.__r.recognize_google(audio, language='es-ES').lower()
            print("Reconoció:", recognized_speech)
            return recognized_speech
        except sr.UnknownValueError:
            print("Google Speech Recognition no pudo entender el audio")
        except sr.RequestError as e:
            print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google:", e)
        
        return None

class SpeechSynthesizer():
    def speak(self, audioString):
        path_mp3 = "audio.mp3"
        print(audioString)
        tts = gTTS(text=audioString, lang='es')
        tts.save(path_mp3)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(path_mp3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(5)
        if os.path.exists(path_mp3):
            pygame.mixer.quit()
            os.remove(path_mp3)
            print("El archivo 'audio.mp3' ha sido eliminado.")
        else:
            print("El archivo 'audio.mp3' no existe.")

class VoiceAssistant():
    def __init__(self):
        self.speechRecognizer = SpeechRecognizer()
        self.ss = SpeechSynthesizer()

    def dictar_rutina(self, parte):
        pywhatkit.playonyt("rutina de " + parte)

    def close_youtube_window(self):
        for window in gw.getWindowsWithTitle('YouTube'):
            window.close()

    def assist(self):
        self.ss.speak("Bienvenida Isabel al asistente de gym, ¿qué rutina deseas hacer hoy?")
        recognized_speech = self.speechRecognizer.recognize()
        if recognized_speech:
            tokens = nltk.word_tokenize(recognized_speech)
            gramatica = nltk.CFG.fromstring("""
                O -> Sustantivo
                Sustantivo -> Pierna | brazo | cardio
                Pierna -> "pierna"
                brazo -> "brazo"
                cardio -> "cardio"
            """)
            
            rd_parser = nltk.RecursiveDescentParser(gramatica)
            try:
                for tree in rd_parser.parse(tokens):
                    print(tree)
                    tree.pretty_print()
                    if "pierna" in recognized_speech:
                        self.ss.speak("Vamos a empezar con la rutina de pierna.")
                        self.dictar_rutina("pierna")
                    elif "brazo" in recognized_speech:
                        self.ss.speak("Vamos a empezar con la rutina de brazo.")
                        self.dictar_rutina("brazo")
                    elif "cardio" in recognized_speech:
                        self.ss.speak("Vamos a empezar con la rutina de cardio.")
                        self.dictar_rutina("cardio")
                    else:
                        self.ss.speak("Lo siento, no entendí tu solicitud. Por favor, intenta de nuevo.")
                    return
            except ValueError as e:
                print(f"Error: {e}")
                self.ss.speak("Lo siento, no entendí tu solicitud. Por favor, intenta de nuevo.")

# Inicialización de la cámara
cap_region_x_begin = 0.5  # punto de inicio/ancho total
cap_region_y_end = 0.8  # punto de inicio/ancho total
threshold = 60  # umbral binario
blurValue = 41  # parámetro de GaussianBlur
bgSubThreshold = 50
learningRate = 0

isBgCaptured = 0  # bool, si el fondo está capturado
triggerSwitch = False  # si es verdadero, el simulador de teclado funciona

camera = cv2.VideoCapture(0)
camera.set(10, 200)

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
time.sleep(2)
isBgCaptured = 1
print('Background captured')

# Instancia del asistente de voz
voiceAssistant = VoiceAssistant()

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # filtro de suavizado
    frame = cv2.flip(frame, 1)  # voltear el frame horizontalmente
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)
    
    # Ejecutar una vez capturado el fondo
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # recortar la ROI

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    if k == 27:  # presionar ESC para salir de todas las ventanas en cualquier momento
        break
    elif k == 32:
        # Si se presiona la barra espaciadora
        cv2.imshow('original', frame)
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_rgb_image_vgg(target)
        if prediction == 'A':
            a_count += 1
            if a_count == 1:
                voiceAssistant.assist()
        elif prediction == 'S':
            s_count += 1
            if s_count == 1:
                voiceAssistant.ss.speak("Es todo por hoy, bonito día.")
                voiceAssistant.close_youtube_window()  # Cerrar la ventana de YouTube
        elif prediction == 'L':
            voiceAssistant.ss.speak("Cambio de rutina, ¿qué rutina deseas hacer?")
            recognized_speech = voiceAssistant.speechRecognizer.recognize()
            if recognized_speech:
                if "pierna" in recognized_speech:
                    voiceAssistant.ss.speak("Vamos a empezar con la rutina de pierna.")
                    voiceAssistant.dictar_rutina("pierna")
                elif "brazo" in recognized_speech:
                    voiceAssistant.ss.speak("Vamos a empezar con la rutina de brazo.")
                    voiceAssistant.dictar_rutina("brazo")
                elif "cardio" in recognized_speech:
                    voiceAssistant.ss.speak("Vamos a empezar con la rutina de cardio.")
                    voiceAssistant.dictar_rutina("cardio")
                else:
                    voiceAssistant.ss.speak("Lo siento, no entendí tu solicitud. Por favor, intenta de nuevo.")

camera.release()
cv2.destroyAllWindows()
