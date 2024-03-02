import speech_recognition as sr
from nltk.tokenize import word_tokenize
import nltk
from nltk import grammar
# Inicializar el reconocedor
r = sr.Recognizer()

# Configurar el micrófono
mic = sr.Microphone()

# Ajustar para el ruido ambiental
with mic as source:
    print("Ajustando para ruido ambiente...")
    r.adjust_for_ambient_noise(source)
    print("Listo, puedes hablar.")

# Escuchar audio desde el micrófono
print("Escuchando...")
with mic as source:
    audio = r.listen(source)

# Transcribir el audio utilizando Google Speech Recognition
try:
    print("Enviando a Google para transcripción...")
    text = r.recognize_google(audio, language="es-ES")
    print("Texto transcrito:", text)

    # Procesamiento de texto básico utilizando NLTK
    tokens = word_tokenize(text)
    print("\nTokens:")
    print(tokens)
    print("\nTokens etiquetados:")
    print(nltk.pos_tag(tokens))
except sr.UnknownValueError:
    print("Google Speech Recognition no pudo entender el audio")
except sr.RequestError as e:
    print("No se pudieron obtener resultados del servicio de reconocimiento de voz de Google; {0}".format(e))
