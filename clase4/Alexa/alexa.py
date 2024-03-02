import speech_recognition as sr
from nltk.tokenize import word_tokenize
import pywhatkit
import nltk
import pygame
from gtts import gTTS

def speak(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='es')
    tts.save("audio.mp3")
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("audio.mp3")
    pygame.mixer.music.play()
    # Mantén el programa en ejecución para que no termine inmediatamente después de reproducir el archivo
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def transcribir_audio():
    """
    Transcribe el audio del micrófono en texto utilizando la API de Google.

    Returns:
        text (str): El texto transcribido.
    """
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
        text = recognizer.recognize_google(audio, language="es-ES")
        print("Texto transcrito:", text)
        return text
    except sr.UnknownValueError:
        print("No se pudo entender el audio")
    except sr.RequestError as e:
        print("Error al solicitar la transcripción; {0}".format(e))

    return None

# Definir la gramática
grammar = nltk.CFG.fromstring("""
  O -> Reproducir Objeto
  Objeto -> YT | YouTube | youtube
  Reproducir -> "reproducir" | "poner" | "Reproducir"
  YT -> "YouTube"
""")

# Crear un analizador sintáctico para la gramática definida
parser = nltk.ChartParser(grammar)
try:
    text = transcribir_audio()
    tokens = text.split()
    rd_parser = nltk.RecursiveDescentParser(grammar)
    for tree in rd_parser.parse(tokens):
        print(tree)
        tree.pretty_print()
        # Play a Video on YouTube
        speak("¿Qué video deseas reproducir?")
        text= transcribir_audio()
        pywhatkit.playonyt(text)
except ValueError:
    print("No se reconoce como oración del lenguaje")
except sr.UnknownValueError:
    print("Google Speech Recognition no pudo entender el audio")
except sr.RequestError as e:
    print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))
