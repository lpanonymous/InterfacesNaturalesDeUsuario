import speech_recognition as sr
class SpeechRecognizer():
    def __init__(self):
        self.__text = ""
    def recognize(self):
        self.__r = sr.Recognizer()
        self.__mic = sr.Microphone()
        print("speak:")
        with self.__mic as source:
            print("ajustando")
            self.__r.adjust_for_ambient_noise(source)
            print("escuchando")
            self.__audio =self.__r.listen(source)

        # recognize speech using Google Speech Recognition
        try:
            print("enviando a google")
            self.__text = self.__r.recognize_google(self.__audio)
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            print(self.__text)
            return(self.__text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))


class VoiceAssistant():
    def __init__(self):
        self.__wordsequence = ""
        self.__speechrecognizer = SpeechRecognizer()
        #nlp = NaturalLanguageProcessor()
        #sa = ActionSelector()
        #ss=SpeechSynthetizer()

    def assist(self):
        self.__wordsequence = self.__speechrecognizer.recognize()
        print("la secuencia detectada es:", self.__wordsequence)
        #semInterpretation = nlp.process(wordsequence)
        #msg = sa.actionselection(semInterpretation)
        #ss.synthetize(msg)

if __name__ == "__main__":
    va = VoiceAssistant()
    va.assist()
    