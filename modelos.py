import google.generativeai as genai
import os

# PEGA TU API KEY AQUÍ
genai.configure(api_key="AIzaSyCPVFidTPfkWMIwoW2CTrL8JGOoGvM-XTQ")

print("Listando modelos disponibles para tu clave...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")