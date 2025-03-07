from keras import layers, models
import mrcfile 
import numpy as np
from scipy.ndimage import zoom
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
Se crea un modelo de segmentación de imágenes con una U-Net para segmentar tomogramas de células.
'''


#primero cargo los datos

with mrcfile.open('./data/tomo.mrc') as f:
    original = f.data.astype(np.float32)
with mrcfile.open('./data/tomo_gt.mrc') as f:
    gt = f.data.astype(np.int32)

# Normalizar (ejemplo: escalar a [0, 1])
original = (original - original.min()) / (original.max() - original.min())

# Extraer cortes X-Y (eje Z)
slices_xy = [original[:, :, z] for z in range(original.shape[2])]
slices_xy_gt = [gt[:, :, z] for z in range(gt.shape[2])]

# Factores de escala para reducir de (100, 400) a (50, 50)
scale_factors = (100 / original.shape[0], 100 / original.shape[1])  # (escala en altura, escala en ancho)

# Redimensionar cada corte manteniendo coherencia
slices_xy_resized = [zoom(slice, scale_factors, order=1) for slice in slices_xy]  # Interpolación bilineal
slices_xy_gt_resized = [zoom(slice, scale_factors, order=0) for slice in slices_xy_gt]  # Vecino más cercano (mantiene clases)

# Convertir a arrays numpy y añadir dimensión de canal
X = np.expand_dims(np.array(slices_xy_resized), axis=-1)  # (n_slices, 50, 50, 1)
y = np.expand_dims(np.array(slices_xy_gt_resized), axis=-1)  # (n_slices, 50, 50, 1)

#definimos la arquitectura de la red neuronal, en este caso una U-Net para segmentación de imágenes con 6 clases.


def build_unet_segmentation(input_shape, num_classes=6):
    inputs = layers.Input(input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottom
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2], axis=-1)
    c4 = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1], axis=-1)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    
    # Output for segmentation
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#  la estructura cuenta con 3 capas de convolución y pooling en el encoder, una capa en el bottom y 2 capas de convolución en el decoder.

# Dividir datos (¡eliminar la dimensión extra en y!)
X_train, X_val, y_train, y_val = train_test_split(
    X, 
    y.squeeze(),  # Eliminar la última dimensión (cambia shape de (n, H, W, 1) a (n, H, W)
    test_size=0.2, 
    random_state=42
)

# Crear modelo
model = build_unet_segmentation((X.shape[1], X.shape[2], 1))
model.summary()

# Entrenar
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, 
    y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    epochs=50,
    batch_size=16
)

# Evaluar
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")


# Function to visualize the results
def visualize_results(X, y_true, y_pred, index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original slice
    axes[0].imshow(X[index].squeeze(), cmap='gray')
    axes[0].set_title('Original Slice')
    
    # Ground truth
    axes[1].imshow(y_true[index], cmap='gray')
    axes[1].set_title('Ground Truth')
    
    # Predicted segmentation
    axes[2].imshow(y_pred[index], cmap='gray')
    axes[2].set_title('Predicted Segmentation')
    
    plt.show()

# Predict on validation set
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=-1)  # Convert one-hot encoded predictions to class labels

# Visualize a couple of slices
visualize_results(X_val, y_val, y_pred, index=0)
visualize_results(X_val, y_val, y_pred, index=1)