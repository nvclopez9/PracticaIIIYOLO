import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo preentrenado
model = YOLO("yolo11n.pt")


def detectar(img):
    # Llamo al modelo, le digo que NO me dibuje cajitas con una probabilidad de menos del 25%
    results = model.predict(source=img, conf=0.25, save=False, verbose=False)

    # Cojo la primera imagen
    print(results)
    result = results[0]
    boxes = result.boxes  # Esto son las cajas...

    # Selecciono la imagen generada
    imagen = result.plot()  # Esta imagen contiene las cajas ya metidas!
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # Con esto cambio el orden de colores de BGR a RGB
    # SI quitáis esta linea de arriba veréis la imagen como en "azul". Es por una librería de Python
    # Que usa YOLO por debajo, que representa las imagenes en BGR en vez de en RGB.

    num_objetos = len(result.boxes) # boxes es un array de todas las cajas que se han identificado

    if num_objetos == 0:
        mensaje = "No se ha detectado ningún objeto."
    else:
        mensaje = f"Se han detectado {num_objetos} objetos."



    return imagen, mensaje


# Interfaz Gradio
demo = gr.Interface(
    fn=detectar,
    inputs=gr.Image(type="filepath"), # Aquí quiero la ruta de la imagen
    outputs=[
        gr.Image(label="Resultado"),
        gr.Textbox(label="¿Cuántos objetos hemos detectado?"),
    ],
    title="YOLO",
    description="Sube una imagen para detectar objetos usando el modelo preentrenado (dataset COCO)"
)

if __name__ == "__main__":
    demo.launch()
