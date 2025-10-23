from django.shortcuts import render
import pickle
import os
import numpy as np

# Cargar el modelo entrenado
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelo_path = os.path.join(BASE_DIR, 'modelo_titanic.pkl')
with open(modelo_path, 'rb') as archivo:
    modelo = pickle.load(archivo)

def index(request):
    """Vista principal con el formulario y resultado"""
    resultado = None
    
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            pclass = request.POST.get('pclass')
            sex = request.POST.get('sex')
            age = request.POST.get('age')
            sibsp = request.POST.get('sibsp')
            parch = request.POST.get('parch')
            fare = request.POST.get('fare')
            embarked = request.POST.get('embarked')
            
            # Validar que todos los campos tengan valores
            if pclass and sex and age and sibsp is not None and parch is not None and fare and embarked:
                # Convertir a los tipos correctos
                pclass = int(pclass)
                sex = int(sex)
                age = float(age)
                sibsp = int(sibsp)
                parch = int(parch)
                fare = float(fare)
                embarked = int(embarked)
                
                # Preparar los datos para el modelo
                datos = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
                
                # Hacer la predicción
                prediccion = modelo.predict(datos)[0]
                
                # Preparar el resultado
                resultado = {
                    'sobrevive': prediccion == 1
                }
        except (ValueError, TypeError):
            # Si hay error en la conversión, no hacer nada
            pass
    
    return render(request, 'index.html', {'resultado': resultado})