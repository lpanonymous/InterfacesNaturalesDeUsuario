import cv2 as cv

# Cargar la imagen
img = cv.imread("C:/Users/zS22000728/Downloads/amorcita.jpeg")

# Verificar si la imagen se cargo correctamente
if img is not None:
    # Mostrar la imagen
    cv.imshow("Ventana de visualización", img)

    # Esperar una tecla para cerrar la ventana
    k = cv.waitKey(0)

    # Cerrar la ventana
    cv.destroyAllWindows()
else:
    print("Error al cargar la imagen")