import os
import sys
import subprocess

import cv2
from ultralytics import YOLO

def main():

    modeloSinEntrenar = "yolo11n.yaml"
    modeloEntrenado = "yolo11n.pt"
    numEpochs = 2 # En la docu se recomienda... empezar en 100!

    model = YOLO(modeloEntrenado)


    # 1) Uso las imagenes que defino en mi .yaml
    model.train(data="dataCheetah.yaml", epochs=numEpochs, imgsz=640, batch=-1)

    # 2) Vamos a ver que tal predice...
    results = model.predict(
        source="test/images",
        conf=0.2,
        show=True,
        save = True
    )

    # Cojo la imagn del resultado, uso la librería cv2 para sacarlo por pantalla
    img = cv2.resize(results[0].plot(), (640, 640))

    cv2.imshow("Prediccion YOLO", img)
    cv2.waitKey(0)  # CUANDO PULSE UNA TECLA SE CIERRA LA IMAGEN!
    cv2.destroyAllWindows()


    print("Los resultados tienen esta forma...")
    print(results)


if __name__ == "__main__":
    main()
