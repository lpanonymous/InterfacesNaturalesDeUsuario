import threading
from gesture_recognition import GestureRecognition
from voice_recognition import voice_recognition

if __name__ == "__main__":
    terminate_flag = threading.Event()
    
    # Instancia de GestureRecognition
    gesture_recognition_instance = GestureRecognition(terminate_flag)
    
    # Definir los objetivos de los hilos
    gesture_recognition_target = lambda: gesture_recognition_instance.gesture_recognition(terminate_flag)
    
    thread1 = threading.Thread(target=gesture_recognition_target)
    thread2 = threading.Thread(target=voice_recognition, args=(terminate_flag,))

    # Iniciar los hilos
    thread1.start()
    thread2.start()

    # Esperar a que los hilos terminen
    thread1.join()
    thread2.join()
