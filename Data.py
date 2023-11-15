#
import cv2
import os

# importar la clase
import HandTracking as sm

#creacion de la carpeta
nombre = 'Letra_DDDDDDD'
direccion = 'C:\\Users\\deadp\\Documents\\UABC\\Proyecto_Traductor_de_señas\\Database'
carpeta = direccion + '/' + nombre

#Modo entrenamiento =True, Modo solo captura = False
train = False

#Si no esta creada la carpeta 
if not os.path.exists(carpeta):
    print("CARPETA CREADA: ", carpeta)
    #creamos la carpeta
    os.makedirs(carpeta)

# Lectura de la camara 
cap = cv2.VideoCapture(0)
# Cambiar la resolucion 
cap.set(3, 1280)
cap.set(4, 720)

# Declaramos contador
cont = 0

# Declara detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizar la lectura de la caputra(de la camara)
    ret, frame = cap.read()

    # Extraer informacion de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posiciones de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos= True, dibujarBox= False, color=[0,225,0])

    # Si hay mano
    if mano == 1:
        # Extraer la informacion del cuadro
        xmin, ymin, xmax, ymax = bbox
        
        # Asigancion de margen para mejor captura de mano
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40
        
        # Realizaremos recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]

        # Redimensionamiento
        recorte = cv2.resize(recorte, (640,640), interpolation=cv2.INTER_CUBIC)

        if train == True:
            # Almacenar nuestras imagenes
            cv2.imwrite(carpeta + "/DDDDDDD_{}.jpg".format(cont), recorte)
        
            # Aumentamos contador
            cont = cont + 1

        cv2.imshow("RECORTE", recorte)

        #cv2.rectangle(frame,(xmin, ymin), (xmax, ymax), [0,255,0], 2)

    # Mostrar FPS
    cv2.imshow("LENGUAJE SEÑAS", frame)
    
    # Leer nuestro teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 299:
        break

cap.release()
cv2.destroyAllWindows()