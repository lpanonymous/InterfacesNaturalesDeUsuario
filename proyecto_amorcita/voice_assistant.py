import speech_recognition as sr
import nltk
import pygame
from gtts import gTTS
import os
import time

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
    def __init__(self):
        self.path_mp3 = "audio.mp3"
        pygame.init()
        pygame.mixer.init()
         
    
    def stop(self):
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.playing = False  # Actualizar el indicador
            print("Reproducción anterior detenida.")
            while pygame.mixer.music.get_busy():  # Asegurar que la reproducción se detenga completamente
                pygame.time.Clock().tick(5)
    
    def speak(self, audioString):
        # Asegurarse de que el mezclador esté inicializado
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        self.stop()  # Detener la reproducción anterior si está en progreso
        print(audioString)
        tts = gTTS(text=audioString, lang='es')
        tts.save(self.path_mp3)
        pygame.mixer.music.load(self.path_mp3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(5)
        if os.path.exists(self.path_mp3):
            pygame.mixer.quit()
            os.remove(self.path_mp3)
            print("El archivo 'audio.mp3' ha sido eliminado.")
        else:
            print("El archivo 'audio.mp3' no existe.")

class VoiceAssistant():
    def __init__(self, terminate_flag):
        self.terminate_flag = terminate_flag
        self.speechRecognizer = SpeechRecognizer()
        self.ss = SpeechSynthesizer()
        self.ejercicios = {
            "pierna": ["Sentadillas", "Estocadas", "Peso muerto", "Elevación de talones"],
            "brazo": ["Flexiones", "Curl de bíceps", "Fondos", "Press de hombros"],
            "cardio": ["Correr", "Saltar la cuerda", "Burpees", "Mountain climbers"]
        }
        self.reask_routine = False  # Bandera para indicar si se debe volver a preguntar por la rutina

    def dictar_rutina(self, parte):
        ejercicios = self.ejercicios.get(parte, [])
        for ejercicio in ejercicios:
            self.ss.speak(f"A continuación, haz 4 series de 12 repeticiones de {ejercicio}. Empieza con un máximo de peso y disminuye el peso en la última serie.")
            time.sleep(2)  # Pausa de 2 segundos entre ejercicios

    def assist(self):
        if self.reask_routine:
            self.reask_routine = False
            return
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
