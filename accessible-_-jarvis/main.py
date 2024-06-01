import threading
from gesture_recognition import GestureRecognition
from voice_recognition import voice_recognition

if __name__ == "__main__":
    terminate_flag = threading.Event()
    # Crear los hilos para reconocimiento de gestos y de voz
    gesture_recognition_instance = GestureRecognition()
    thread1 = threading.Thread(target=gesture_recognition_instance.gesture_recognition, args=(terminate_flag,))
    thread2 = threading.Thread(target=voice_recognition, args=(terminate_flag,))

    # Iniciar los hilos
    thread1.start()
    thread2.start()

    # Esperar a que los hilos terminen
    thread1.join()
    thread2.join()
