import nltk
from voice_assistant import VoiceAssistant

def voice_recognition(terminate_flag):
    voiceAssistant = VoiceAssistant(terminate_flag)
    while not terminate_flag.is_set():
        voiceAssistant.assist()
