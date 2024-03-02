#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:30:50 2018

@author: ebenitez
"""

from gtts import gTTS
import os
 
def speak(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='es')
    tts.save("audio.mp3")
    os.system("mpg123 audio.mp3")
    
if __name__ == "__main__":
    #print(os.environ['PATH'])
    #os.environ['PATH'] += ":"+"/usr/local/bin"
    #print(os.environ['PATH'])
    #print(os.getcwd())
            
    #text to speech    
    speak('esto es una prueba')
