from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model

'''
Programa que genera de manera aleatoria numeros entre 0 y 1, y los clasifica en dos clases dependiendo de 
si la suma de los dos numeros es mayor a 1 utilizando un MLP.
'''

# Generación de datos sintéticos
np.random.seed(42)
x_data = np.random.rand(1000, 2)  # 1000 muestras con 2 características
y_data = (np.sum(x_data, axis=1) > 1).astype(int)  # Etiqueta: 1 si la suma es mayor que 1, 0 en caso contrario

# División de los datos en entrenamiento (80%) y prueba (20%)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Construcción del modelo
model = Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),  # Capa oculta con 64 neuronas y activación ReLU
    layers.Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide
])

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Pérdida en prueba: {test_loss:.4f}")
print(f"Precisión en prueba: {test_accuracy:.4f}")

# Predicción en nuevas muestras
x_sample = np.array([[0.6, 0.6], [0.2, 0.1], [0.8, 0.9]])
y_pred_prob = model.predict(x_sample)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Predicciones:")
for i, sample in enumerate(x_sample):
    print(f"Entrada: {sample}, Probabilidad: {y_pred_prob[i][0]:.4f}, Clase: {y_pred[i][0]}")