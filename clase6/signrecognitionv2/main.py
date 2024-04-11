import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Definir el diccionario de gestos y el mapeo de gestos
gestures = {'L_': 'L', 'fi': 'Fist', 'ok': 'Okay', 'pe': 'Peace', 'pa': 'Palm'}
gestures_map = {'Fist': 0, 'L': 1, 'Okay': 2, 'Palm': 3, 'Peace': 4}

def process_image(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

def walk_file_tree(relative_path):
    X_data = []
    y_data = []

    for directory, subdirectories, files in os.walk(relative_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = gestures[file[0:2]]
                X_data.append(process_image(path))
                y_data.append(gestures_map[gesture_name])
            else:
                continue
                
    return X_data, y_data

# Ruta de las imágenes
relative_path = 'C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase6/signrecognitionv2/silhouettes'

# Obtener los datos
X_data, y_data = walk_file_tree(relative_path)

# Convertir a arrays numpy
X_data = np.array(X_data, dtype='float32') / 255.0
y_data = np.array(y_data)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=12, stratify=y_data)

# Ajustar la forma de los datos de entrada
X_train = np.expand_dims(X_train, axis=-1)  # Agregar dimensión de canal
X_test = np.expand_dims(X_test, axis=-1)  # Agregar dimensión de canal

# Construir el modelo
model = Sequential()
model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25, seed=21))
model.add(Dense(units=5, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=4, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Evaluar el modelo
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))

# Guardar el modelo
model.save('C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase6/signrecognitionv2/modelo.h5')
print('Modelo guardado exitosamente.')

