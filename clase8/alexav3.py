import speech_recognition as sr
import pywhatkit
import nltk
import pygame
from gtts import gTTS
import os

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
            self.__audio_player.play("No se pudo entender el audio, ¿puedes repetir por favor?")
        except sr.RequestError as e:
            print("Error al solicitar la transcripción; {0}".format(e))

        return None

class SyntaxAnalyzer:
    def __init__(self):
        self.__grammar = nltk.CFG.fromstring("""
        O -> Sujeto Verbo Sustantivo
        Sujeto -> "Alexa" | "alexa"
        Sustantivo -> YT | youtube
        Verbo -> "reproducir" | "reproduce"
        YT -> "youtube"
        """)
        self.__speech_to_text = SpeechToText()
        self.__audio_player = AudioPlayer()
        self.__text = ""

    def analyze_syntax(self):
        rd_parser = nltk.RecursiveDescentParser(self.__grammar)
        try:
            self.__text = self.__speech_to_text.transcribe_audio()
            if self.__text:
                tokens = self.__text.split()
                for tree in rd_parser.parse(tokens):
                    print(tree)
                    tree.pretty_print()
                    self.__audio_player.play("¿Qué video deseas reproducir?")
                    self.__text = self.__speech_to_text.transcribe_audio()
                    pywhatkit.playonyt(self.__text)
        except ValueError:
            print("No se reconoce como oración del lenguaje")
        except sr.UnknownValueError:
            print("Google Speech Recognition no pudo entender el audio")
        except sr.RequestError as e:
            print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))

# Ejemplo de uso:
if __name__ == "__main__":
    syntax_analyzer = SyntaxAnalyzer()
    syntax_analyzer.analyze_syntax()