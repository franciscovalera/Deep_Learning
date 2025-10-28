''' Se pretende crear un autocodificador que detecte anomalias en un conjunto de datos. 
Se va a utilizar la libreria polar por su superior rendimiento comparado a pandas'''

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from collections import Counter



# Leer el CSV con Polars
data = pl.read_csv('./imgs/ecg.csv')

# Separar señales y etiquetas
X = data.select(pl.all().exclude(data.columns[-1])).to_numpy()
y = data.select(data.columns[-1]).to_numpy().flatten()

# Reetiquetar: 0 -> normal, 1 -> anómalo
# En el dataset, 1 es normal y 0 es anómalo, así que invertimos
y = np.where(y == 1, 0, 1)  # 0: normal, 1: anómalo

# Separar registros normales y anómalos
normal_mask = y == 0
X_normal = X[normal_mask]
X_anomalo = X[~normal_mask]
y_normal = y[normal_mask]
y_anomalo = y[~normal_mask]

# Particionar normales: 70% train, 30% test
X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(
    X_normal, y_normal, test_size=0.3, random_state=42
)

# El conjunto de test incluye 30% normales + todos los anómalos
X_test = np.vstack([X_normal_test, X_anomalo])
y_test = np.concatenate([y_normal_test, y_anomalo])

# El conjunto de entrenamiento solo contiene normales
X_train = X_normal_train
y_train = y_normal_train

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Normales en train: {len(X_train)}, Normales en test: {np.sum(y_test==0)}, Anómalos en test: {np.sum(y_test==1)}")


# Definir hiperparámetros y estructuras de la red
INPUT_SHAPE = (X_train.shape[1],)
ACTIVATION = 'relu'
EPOCHS = 100
BATCH_SIZE = 64

# Dimensiones de la pirámide (potencias de 2)
LATENT_DIMS = [2, 4, 8, 16, 32]
DIM_1 = 64
DIM_2 = 32

# Diccionarios para guardar modelos e historiales
encoders = {}
decoders = {}
autoencoders = {}
histories = {}

# Entrenamiento y visualización para diferentes dimensiones latentes
for latent_dim in LATENT_DIMS:
    # Codificador
    input_ecg = keras.layers.Input(shape=INPUT_SHAPE)
    x = keras.layers.Dense(DIM_1, activation=ACTIVATION)(input_ecg)
    x = keras.layers.Dense(DIM_2, activation=ACTIVATION)(x)
    encoded = keras.layers.Dense(latent_dim, name='latent')(x)
    encoder = keras.Model(input_ecg, encoded, name=f'encoder_{latent_dim}')

    # Decodificador
    latent_input = keras.layers.Input(shape=(latent_dim,))
    x = keras.layers.Dense(DIM_2, activation=ACTIVATION)(latent_input)
    x = keras.layers.Dense(DIM_1, activation=ACTIVATION)(x)
    reconstructed = keras.layers.Dense(INPUT_SHAPE[0], name='output')(x)
    decoder = keras.Model(latent_input, reconstructed, name=f'decoder_{latent_dim}')

    # Autoencoder
    autoencoder_input = keras.layers.Input(shape=INPUT_SHAPE)
    encoded_z = encoder(autoencoder_input)
    decoded = decoder(encoded_z)
    autoencoder = keras.Model(autoencoder_input, decoded, name=f'autoencoder_{latent_dim}')

    autoencoder.compile(optimizer='adam', loss='mae')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = autoencoder.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # Guardar modelos e historiales
    encoders[latent_dim] = encoder
    decoders[latent_dim] = decoder
    autoencoders[latent_dim] = autoencoder
    histories[latent_dim] = history

    # Visualización de ejemplos
    idx = np.random.choice(len(X_train), 3, replace=False)
    X_examples = X_train[idx]
    encoded_examples = encoder.predict(X_examples)
    decoded_examples = autoencoder.predict(X_examples)

    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(X_examples[i], label='Original')
        plt.plot(decoded_examples[i], label='Reconstruido')
        plt.title(f'Latent dim: {latent_dim}\nCodificación: {np.round(encoded_examples[i], 2)}')
        plt.xlabel('Tiempo')
        plt.ylabel('Amplitud')
        plt.legend()
    plt.suptitle(f'Autoencoder (Latent dim={latent_dim}) - Ejemplos de reconstrucción')
    plt.tight_layout()
    plt.show()

    # Métricas adicionales
    decoded_train = autoencoder.predict(X_train)
    r2 = np.mean([r2_score(X_train[i], decoded_train[i]) for i in range(len(X_train))])
    print(f"Latent dim: {latent_dim} | R2 medio: {r2*100:.2f}% | Compresión: {latent_dim/np.prod(INPUT_SHAPE)*100:.2f}%")
    print(f"Coeficientes no nulos: {np.count_nonzero(np.sum(encoded_examples, axis=0))}/{latent_dim}\n")


#Ahora vamos a evaluar el autoencoder con mayor dimensión latente entrenado utilizando el conjunto de test.

# Seleccionamos el autoencoder con mayor dimensión latente entrenado (por ejemplo, 32)
latent_dim = 32
autoencoder = autoencoders[latent_dim]

# Reconstrucción de los conjuntos de test y entrenamiento
decoded_test = autoencoder.predict(X_test)
decoded_train = autoencoder.predict(X_train)

# Cálculo del MAE para cada muestra
mae_test = np.mean(np.abs(X_test - decoded_test), axis=1)
mae_train = np.mean(np.abs(X_train - decoded_train), axis=1)

# Separar índices de normales y anómalos en test
idx_normal = np.where(y_test == 0)[0]
idx_anomalo = np.where(y_test == 1)[0]

# Mostrar un ejemplo de cada tipo (normal y anómalo) con su reconstrucción
plt.figure(figsize=(12, 5))

# Ejemplo normal
i_normal = idx_normal[0]
plt.subplot(1, 2, 1)
plt.plot(X_test[i_normal], label='Original normal')
plt.plot(decoded_test[i_normal], label='Reconstruido')
plt.title(f'ECG normal (MAE={mae_test[i_normal]:.4f})')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()

# Ejemplo anómalo
i_anomalo = idx_anomalo[0]
plt.subplot(1, 2, 2)
plt.plot(X_test[i_anomalo], label='Original anómalo')
plt.plot(decoded_test[i_anomalo], label='Reconstruido')
plt.title(f'ECG anómalo (MAE={mae_test[i_anomalo]:.4f})')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()
plt.show()

# Calcular y mostrar el MAE medio para cada tipo
mae_normal = mae_test[idx_normal]
mae_anomalo = mae_test[idx_anomalo]
print(f"MAE medio (normales test): {mae_normal.mean():.4f}")
print(f"MAE medio (anómalos test): {mae_anomalo.mean():.4f}")
print('Queremos, precisamente mucha diferencia entre ambas muestras')

# Histograma de errores MAE
plt.figure(figsize=(8, 5))
plt.hist(mae_train, bins=40, alpha=0.6, label='Train (normales)', color='C0')
plt.hist(mae_normal, bins=40, alpha=0.6, label='Test (normales)', color='C1')
plt.hist(mae_anomalo, bins=40, alpha=0.6, label='Test (anómalos)', color='C3')
plt.xlabel('MAE de reconstrucción')
plt.ylabel('Frecuencia')
plt.title('Histograma de errores MAE de reconstrucción')
plt.legend()
plt.show()


# Visualización para razonar la separabilidad
plt.figure(figsize=(8, 5))
plt.hist(mae_normal, bins=40, alpha=0.7, label='Normales (test)', color='C1')
plt.hist(mae_anomalo, bins=40, alpha=0.7, label='Anómalos (test)', color='C3')
plt.axvline(np.percentile(mae_normal, 95), color='k', linestyle='--', label='Umbral 95% normales')
plt.xlabel('MAE de reconstrucción')
plt.ylabel('Frecuencia')
plt.title('Separabilidad de errores MAE (normales vs anómalos) en electrocardiogramas')
plt.legend()
plt.show()

# Estimar un umbral: percentil 95 de los normales del test, esto se ha realizado una vez se ha visto el histograma
umbral = np.percentile(mae_normal, 95)
print(f"Umbral de detección (percentil 95% de normales): {umbral:.4f}")

# Clasificación: 0=normal, 1=anómalo
y_pred = (mae_test > umbral).astype(int)

# Mostrar conteo de clasificados
print("Conteo de predicciones en test:", Counter(y_pred))
