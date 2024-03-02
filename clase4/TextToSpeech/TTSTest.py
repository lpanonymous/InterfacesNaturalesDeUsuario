#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 00:30:50 2018

@author: ebenitez
"""

from gtts import gTTS
import pygame
 
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
if __name__ == "__main__":
    #print(os.environ['PATH'])
    #os.environ['PATH'] += ":"+"/usr/local/bin"
    #print(os.environ['PATH'])
    #print(os.getcwd())
            
    #text to speech    
    speak('esto es una prueba')
