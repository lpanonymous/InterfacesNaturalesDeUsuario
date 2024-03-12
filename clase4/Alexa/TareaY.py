import speech_recognition as sr
import nltk
import pywhatkit
import pygame
from gtts import gTTS
import os

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

def recognize_speech():
    # Crea una instancia del reconocedor de voz
    r = sr.Recognizer()

    # Crea una instancia del objeto Microphone
    mic = sr.Microphone()

    # Ajusta el reconocedor de voz para el ruido de fondo
    with mic as source:
        print("Ajustando al ruido de fondo...")
        r.adjust_for_ambient_noise(source)

    # Graba y reconoce la voz
    print("Dime algo:")
    with mic as source:
        audio = r.listen(source)

    try:
        # Reconoce la voz usando Google Speech Recognition
        reconocida_habla = r.recognize_google(audio, language='es-ES').lower()
        print("Reconoció: " + reconocida_habla)

        # Analiza la habla reconocida en una estructura de árbol
        tokens = nltk.word_tokenize(reconocida_habla)
        # Definir la gramática
        gramatica = nltk.CFG.fromstring("""                                   
         O -> Verbo Sustantivo
         Sustantivo -> Canción | video | youtube
         Verbo -> "reproducir" | "poner"| "reproduce"
         Canción -> "canción"
        """)
        
        rd_parser = nltk.RecursiveDescentParser(gramatica)
        for tree in rd_parser.parse(tokens):
            print(tree)
            tree.pretty_print()
            # Play a Video on YouTube
            speak("¿Qué video deseas reproducir?")
            text= transcribir_audio()
            pywhatkit.playonyt(text)
    except ValueError:
            print("No se reconoce como oración del lenguaje")
            speak("No te entendí, ¿puedes repetir por favor?")
            recognize_speech()
    except sr.UnknownValueError:
        print("Google Speech Recognition no pudo entender el audio")
    except sr.RequestError as e:
        print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))
 
        
if __name__ == "__main__":
    # Ejecuta el reconocimiento de voz y la síntesis de voz
    recognize_speech()