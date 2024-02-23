import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
print("speak:")
with mic as source:
    print("ajustando")
    r.adjust_for_ambient_noise(source)
    print("escuchando")
    audio =r.listen(source)

# recognize speech using Google Speech Recognition
try:
    print("enviando a google")
    text = r.recognize_google(audio)
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print(text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))