import threading
import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import subprocess
import pyautogui
import os
import speech_recognition as sr
import pywhatkit
import nltk
import pygame
from gtts import gTTS
import sys

# Variable global para señal de terminación
terminate_flag = threading.Event()

# AudioPlayer class definition
class AudioPlayer:
    def __init__(self):
        self.__path_mp3 = "audio.mp3"
        self.__audioString = ""
    
    def play(self, audioString):
        self.__audioString = audioString
        print(self.__audioString)
        tts = gTTS(text=self.__audioString, lang='es')
        tts.save(self.__path_mp3)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(self.__path_mp3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(5)
        if os.path.exists(self.__path_mp3):
            pygame.mixer.quit()
            os.remove(self.__path_mp3)
            print("El archivo 'audio.mp3' ha sido eliminado.")
        else:
            print("El archivo 'audio.mp3' no existe.")

# SpeechToText class definition
class SpeechToText:
    def __init__(self):
        self.__audio_player = AudioPlayer()
        self.__mic= sr.Microphone()
        self.__recognizer = sr.Recognizer()
        self.__text = ""
    
    def transcribe_audio(self):
        with self.__mic as source:
            print("Ajustando para ruido ambiente...")
            self.__recognizer.adjust_for_ambient_noise(source)
            print("Listo, puedes hablar.")

        print("Escuchando...")
        with self.__mic as source:
            audio = self.__recognizer.listen(source)

        print("Enviando a Google para transcripción...")
        try:
            self.__text = self.__recognizer.recognize_google(audio, language="es-ES").lower()
            print("Texto transcrito:", self.__text)
            return self.__text
        except sr.UnknownValueError:
            print("No se pudo entender el audio, ¿puedes repetir por favor?")
        except sr.RequestError as e:
            print("Error al solicitar la transcripción; {0}".format(e))
        return None

# SyntaxAnalyzer class definition
class SyntaxAnalyzer:
    def __init__(self):
        self.__grammar = nltk.CFG.fromstring("""
        O -> Sujeto Verbo Sustantivo
        Sujeto -> "jarvis"
        Verbo -> "abre" | "abrir" | "inicia" | "inicializa" | "pon" | "iniciar"
        Artículo -> 
        Sustantivo -> "youtube" | "google" | "explorador" | "word" | "excel" | "powerpoint"
        """)
        self.__speech_to_text = SpeechToText()
        self.__audio_player = AudioPlayer()
        self.__text = ""

    def analyze_syntax(self):
        rd_parser = nltk.RecursiveDescentParser(self.__grammar)
        while not terminate_flag.is_set():  # Loop to keep listening
            try:
                self.__text = self.__speech_to_text.transcribe_audio()
                if self.__text:
                    tokens = self.__text.split()
                    for tree in rd_parser.parse(tokens):
                        for subtree in tree.subtrees():
                            if subtree.label() == 'Sustantivo':
                                sustantivo_value = subtree[0]
                                print("Sustantivo:", sustantivo_value)
                                if sustantivo_value == "youtube":
                                    self.__audio_player.play("¿Qué video deseas reproducir?")
                                    self.__text = self.__speech_to_text.transcribe_audio()
                                    pywhatkit.playonyt(self.__text)
                                elif sustantivo_value == "google":
                                    subprocess.Popen("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
                                    self.__audio_player.play("Google se ha abierto")
                                elif sustantivo_value == "explorador":
                                    os.system("explorer")
                                    self.__audio_player.play("El explorador de archivos se ha abierto")
                                elif sustantivo_value == "word":
                                    subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.exe")
                                    self.__audio_player.play("Word se ha abierto")
                                elif sustantivo_value == "excel":
                                    subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.exe")
                                    self.__audio_player.play("Excel se ha abierto")
                                elif sustantivo_value == "powerpoint":
                                    subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.exe")
                                    self.__audio_player.play("PowerPoint se ha abierto")
            except ValueError:
                print("No se reconoce como oración del lenguaje")
            except sr.UnknownValueError:
                print("Google Speech Recognition no pudo entender el audio")
            except sr.RequestError as e:
                print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))

# Gesture recognition setup
gesture_names = {0: 'google', 1: 'explorador', 2: 'word', 3: 'excel', 4: 'powerpoint', 5: 'volumeup', 6: 'volumedown'}
model = load_model('C:/Users/zS22000728/Documents/VS/InterfacesNaturalesDeUsuario/accessible-_-jarvis/saved_model.hdf5')

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    if result == 'google':
        subprocess.Popen("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
    elif result == 'explorador':
        os.system("explorer")
    elif result == 'word':
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.exe")
    elif result == 'excel':
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.exe")
    elif result == 'powerpoint':
        subprocess.Popen("C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.exe")
    elif result == 'volumeup':
        pyautogui.press('volumeup')
    elif result == 'volumedown':
        pyautogui.press('volumedown')
    return result, score

cap_region_x_begin = 0.5
cap_region_y_end = 0.8
threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0
isBgCaptured = 0
triggerSwitch = False

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def gesture_recognition():
    global isBgCaptured, bgModel, triggerSwitch
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(10, 200)
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    time.sleep(2)
    isBgCaptured = 1
    print('Background captured')
    while camera.isOpened() and not terminate_flag.is_set():
        ret, frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0), 
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):
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
        if k == 32:
            target = np.stack((thresh,) * 3, axis=-1)
            target = cv2.resize(target, (224, 224))
            target = target.reshape(1, 224, 224, 3)
            predict_rgb_image_vgg(target)
        elif k == 27:
            camera.release()
            cv2.destroyAllWindows()
            terminate_flag.set()  

def voice_recognition():
    syntax_analyzer = SyntaxAnalyzer()
    syntax_analyzer.analyze_syntax()

if __name__ == "__main__":
    thread1 = threading.Thread(target=gesture_recognition)
    thread2 = threading.Thread(target=voice_recognition)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
