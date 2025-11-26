import numpy as np
import cv2
from tensorflow.keras.models import load_model

modelo = load_model("deportes_model.h5")

deportes = [
    "americano",
    "basket",
    "beisball",
    "boxeo",
    "ciclismo",
    "f1",
    "futbol",
    "golf",
    "natacion",
    "tenis"
]

def predecir_imagen(ruta, modelo, deportes):
    img = cv2.imread(ruta)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # EL MODELO ESPERA 28 x 28
    img_r = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    if len(img_r.shape) == 2:
        img_r = cv2.merge([img_r, img_r, img_r])

    img_n = img_r.astype("float32") / 255.0
    batch = np.expand_dims(img_n, axis=0)

    p = modelo.predict(batch, verbose=0)[0]
    clase = np.argmax(p)
    conf = p[clase]

    orden = np.argsort(p)[::-1]
    print("Top 3 clases:")
    for i in orden[:3]:
        print(f"  {deportes[i]}: {p[i]*100:.2f}%")

    return deportes[clase], conf


ruta_test = "Test4.jpg"
deporte, confianza = predecir_imagen(ruta_test, modelo, deportes)

print("Deporte:", deporte)
print("Confianza:", f"{confianza * 100:.2f}%")
