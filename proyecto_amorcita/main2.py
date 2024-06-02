import threading
import numpy as np
import cv2
import copy
import time
from keras.models import load_model
import speech_recognition as sr
import nltk
import pygame
from gtts import gTTS
import os

class GestureRecognition():
    def __init__(self, terminate_flag, voice_assistant_instance):
        self.terminate_flag = terminate_flag
        self.voice_assistant = voice_assistant_instance
        self.cap_region_x_begin = 0.5  
        self.cap_region_y_end = 0.8  
        self.threshold = 60  
        self.blurValue = 41  
        self.bgSubThreshold = 50
        self.learningRate = 0

        self.isBgCaptured = 0  
        self.camera = cv2.VideoCapture(0)
        self.camera.set(10, 200)

        self.bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
        time.sleep(2)
        self.isBgCaptured = 1
        print('Background captured')

        self.model = load_model('C:/Users/zS22000728/Documents/VS/InterfacesNaturalesDeUsuario/proyecto_amorcita/saved_model.hdf5')

        self.gesture_names = {0: 'A', 1: 'S', 2: 'L'}

        self.a_count = 0  

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
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  
            frame = cv2.flip(frame, 1)  

            if self.isBgCaptured == 1:
                img = self.remove_background(frame)
                img = img[0:int(self.cap_region_y_end * frame.shape[0]),
                          int(self.cap_region_x_begin * frame.shape[1]):frame.shape[1]]  

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)
                ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

            k = cv2.waitKey(10)
            if k == 27 or terminate_flag.is_set():  
                break
            elif k == 32:
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = self.predict_rgb_image_vgg(target)
                if prediction == 'A':
                    self.a_count += 1
                    if self.a_count == 1:
                        print("Realizar acción para el gesto A")
                        self.voice_assistant.assist()  
        self.camera.release()
        cv2.destroyAllWindows()

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
            print("El archivo audio.mp3' ha sido eliminado.")
        else:
            print("El archivo 'audio.mp3' no existe.")

class VoiceAssistant():
    def __init__(self,terminate_flag):
        self.terminate_flag = terminate_flag
        self.speechRecognizer = SpeechRecognizer()
        self.ss = SpeechSynthesizer()
        self.ejercicios = {
            "pierna": ["Sentadillas", "Estocadas", "Peso muerto", "Elevación de talones"],
            "brazo": ["Flexiones", "Curl de bíceps", "Fondos", "Press de hombros"],
            "cardio": ["Correr", "Saltar la cuerda", "Burpees", "Mountain climbers"]
        }

    def dictar_rutina(self, parte):
        ejercicios = self.ejercicios.get(parte, [])
        for ejercicio in ejercicios:
            self.ss.speak(f"A continuación, haz 4 series de 12 repeticiones de {ejercicio}. Empieza con un máximo de peso y disminuye el peso en la última serie.")
            time.sleep(2)  # Pausa de 2 segundos entre ejercicios

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
            except ValueError as e:
                print(f"Error: {e}")
                self.ss.speak("Lo siento, no entendí tu solicitud. Por favor, intenta de nuevo.")

def voice_recognition(terminate_flag):
    voiceAssistant = VoiceAssistant(terminate_flag)
    while not terminate_flag.is_set():
        voiceAssistant.assist()

def gesture_recognition(terminate_flag):
    voice_assistant_instance = VoiceAssistant(terminate_flag)
    gesture_recognition_instance = GestureRecognition(terminate_flag, voice_assistant_instance)
    gesture_recognition_instance.gesture_recognition(terminate_flag)

if __name__ == "__main__":
    terminate_flag = threading.Event()
    thread1 = threading.Thread(target=gesture_recognition, args=(terminate_flag,))
    thread2 = threading.Thread(target=voice_recognition, args=(terminate_flag,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()                            
               
