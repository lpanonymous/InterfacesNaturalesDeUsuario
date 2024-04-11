import numpy as np
from PIL import Image
from keras.models import load_model

# Definir el diccionario de gestos y el mapeo de gestos
gestures = {'L_': 'L', 'fi': 'Fist', 'ok': 'Okay', 'pe': 'Peace', 'pa': 'Palm'}
gestures_map = {'Fist': 0, 'L': 1, 'Okay': 2, 'Palm': 3, 'Peace': 4}

def process_image(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

# Cargar el modelo guardado
loaded_model = load_model('C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase6/signrecognitionv2/modelo.h5')

# Procesar una imagen de prueba
test_image_path = 'C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase6/signrecognitionv2/silhouettes/fist_001.jpg'
test_image = process_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0)  # Agregar dimensión para el lote de muestras
test_image = np.expand_dims(test_image, axis=-1)  # Agregar dimensión para el canal de color

# Hacer predicciones sobre la imagen de prueba
predictions = loaded_model.predict(test_image)

# Interpretar las predicciones para obtener el resultado
gesture_names = {0: 'Fist', 1: 'L', 2: 'Okay', 3: 'Palm', 4: 'Peace'}
predicted_class = np.argmax(predictions)
predicted_gesture = gesture_names[predicted_class]
confidence = predictions[0][predicted_class]

print('Gesto predicho:', predicted_gesture)
print('Confianza:', confidence)
