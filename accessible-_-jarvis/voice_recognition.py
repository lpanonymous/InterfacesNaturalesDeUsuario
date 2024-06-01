import speech_recognition as sr
import pywhatkit
import nltk
import subprocess
import os
import pygame
from gtts import gTTS

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

class SpeechToText:
    def __init__(self, terminate_flag):
        self.__audio_player = AudioPlayer()
        self.__mic= sr.Microphone()
        self.__recognizer = sr.Recognizer()
        self.__text = ""
        self.terminate_flag = terminate_flag
    
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

class SyntaxAnalyzer:
    def __init__(self, terminate_flag):
        self.__grammar = nltk.CFG.fromstring("""
        O -> Sujeto Verbo Sustantivo
        Sujeto -> "jarvis"
        Verbo -> "abre" | "abrir" | "inicia" | "inicializa" | "pon" | "iniciar"
        Artículo -> 
        Sustantivo -> "youtube" | "google" | "explorador" | "word" | "excel" | "powerpoint"
        """)
        self.__speech_to_text = SpeechToText(terminate_flag)
        self.__audio_player = AudioPlayer()
        self.__text = ""
        self.terminate_flag = terminate_flag

    def analyze_syntax(self):
        rd_parser = nltk.RecursiveDescentParser(self.__grammar)
        while not self.terminate_flag.is_set():  # Loop to keep listening
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

def voice_recognition(terminate_flag):
    syntax_analyzer = SyntaxAnalyzer(terminate_flag)
    syntax_analyzer.analyze_syntax()
