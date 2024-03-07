import speech_recognition as sr
import pywhatkit
import nltk
import pygame
from gtts import gTTS
import os

# Definir la gramática
grammar = nltk.CFG.fromstring("""
    O -> Verbo Sustantivo
    Sustantivo -> YT | youtube
    Verbo -> "reproducir" | "reproduce"
    YT -> "youtube"
    """)

def speak(audioString):
    path_mp3 = "audio.mp3"
    print(audioString)
    tts = gTTS(text=audioString, lang='es')
    tts.save(path_mp3)
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(path_mp3)
    pygame.mixer.music.play()
    # Mantén el programa en ejecución para que no termine inmediatamente después de reproducir el archivo
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)
    # Verificar si el archivo existe antes de intentar eliminarlo
    if os.path.exists(path_mp3):
        pygame.mixer.quit()
        # Eliminar el archivo
        os.remove(path_mp3)
        print("El archivo 'audio.mp3' ha sido eliminado.")
    else:
        print("El archivo 'audio.mp3' no existe.")

def transcribir_audio():
    #Transcribe el audio del micrófono en texto utilizando la API de Google.
    # Inicializar micrófono y reconocedor de voz
    mic = sr.Microphone()
    recognizer = sr.Recognizer()

    # Ajustar para ruido ambiental
    with mic as source:
        print("Ajustando para ruido ambiente...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listo, puedes hablar.")

    # Escuchar audio desde el micrófono
    print("Escuchando...")
    with mic as source:
        audio = recognizer.listen(source)

    # Transcribir audio a texto utilizando la API de Google
    print("Enviando a Google para transcripción...")
    try:
        text = recognizer.recognize_google(audio, language="es-ES").lower()
        print("Texto transcrito:", text)
        return text
    except sr.UnknownValueError:
        print("No se pudo entender el audio, ¿puedes repetir por favor?")
        speak("No se pudo entender el audio, ¿puedes repetir por favor?")
    except sr.RequestError as e:
        print("Error al solicitar la transcripción; {0}".format(e))

    return None

def analizador_sintactico():
    # Crear un analizador sintáctico para la gramática definida
    rd_parser = nltk.RecursiveDescentParser(grammar)
    while True:
        try:
            text = transcribir_audio()
            if text:
                tokens = text.split()
                for tree in rd_parser.parse(tokens):
                    print(tree)
                    tree.pretty_print()
                    # Reproducir un video en YouTube
                    speak("¿Qué video deseas reproducir?")
                    text = transcribir_audio()
                    pywhatkit.playonyt(text)
        except ValueError:
            print("No se reconoce como oración del lenguaje")
            speak("No te entendí, ¿puedes repetir por favor?")
        except sr.UnknownValueError:
            print("Google Speech Recognition no pudo entender el audio")
        except sr.RequestError as e:
            print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))

if __name__ == "__main__":
    # Ejecuta el reconocimiento de voz y la síntesis de voz
    analizador_sintactico()
