import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATASET: Generador de datos sintéticos (Cuadrados)
# =====================================================================
# En un proyecto real, aquí cargarías tus fotos desde una carpeta.
# Para este ejemplo, generamos imágenes sobre la marcha.
class SquareDataset(Dataset):
    def __init__(self, num_samples=100, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        # Le dice a PyTorch cuántos ejemplos hay en total
        return self.num_samples

    def __getitem__(self, idx):
        # Esta función genera/carga UNA sola imagen y su máscara correspondiente.
        
        # Crear imagen base en negro (ruido de fondo)
        image = np.random.normal(0, 0.1, (1, self.image_size, self.image_size)).astype(np.float32)
        # Crear máscara base en negro
        mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)

        # Definir tamaño y posición aleatoria de un cuadrado
        size = np.random.randint(10, 20)
        x = np.random.randint(0, self.image_size - size)
        y = np.random.randint(0, self.image_size - size)

        # Dibujar el cuadrado en la imagen (color blanco = 1.0)
        image[0, y:y+size, x:x+size] = 1.0
        # Dibujar el cuadrado en la máscara (color blanco = 1.0)
        # La máscara es nuestra "verdad absoluta" (Ground Truth)
        mask[0, y:y+size, x:x+size] = 1.0 

        # PyTorch espera que los datos sean Tensores
        return torch.from_numpy(image), torch.from_numpy(mask)


# =====================================================================
# 2. MODELO: Arquitectura U-Net Básica
# =====================================================================
class BasicUNet(nn.Module):
    def __init__(self):
        super(BasicUNet, self).__init__()
        
        # --- ENCODER (Camino de bajada - Extrae características) ---
        # Toma la imagen (1 canal) y aplica filtros para buscar patrones
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Reduce la imagen a la mitad de tamaño
        self.pool1 = nn.MaxPool2d(2) 
        
        # --- BOTTLENECK (Fondo de la U) ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # --- DECODER (Camino de subida - Reconstruye la imagen) ---
        # Duplica el tamaño de la imagen
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) 
        
        # Al subir, concatenaremos con la información del encoder (Skip Connection)
        # Por eso recibe 32 canales (16 de la subida + 16 del encoder)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Capa final para sacar un solo canal (probabilidad de ser cuadrado o no)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)
        # Sigmoide aprieta los valores entre 0 y 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Paso hacia adelante (cómo fluye la información por la red)
        
        # 1. Bajada
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        # 2. Fondo
        b = self.bottleneck(p1)
        
        # 3. Subida con Skip Connection
        u1 = self.up1(b)
        cat1 = torch.cat([u1, e1], dim=1) # <- ¡El secreto de la U-Net!
        d1 = self.dec1(cat1)
        
        # 4. Salida
        out = self.out_conv(d1)
        return self.sigmoid(out)


# =====================================================================
# 3. ENTRENAMIENTO
# =====================================================================
def train_model():
    print("Preparando datos...")
    # Creamos el dataset y el DataLoader (este último hace los "lotes" o batches)
    dataset = SquareDataset(num_samples=1000, image_size=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("Inicializando modelo U-Net...")
    model = BasicUNet()
    
    # Función de pérdida (Binary Cross Entropy) compara píxel por píxel si es 0 o 1
    criterion = nn.BCELoss() 
    # Optimizador (el que ajusta los pesos de la red)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 5
    print("Iniciando entrenamiento (esto puede tomar unos segundos)...")
    
    for epoch in range(epochs):
        model.train() # Poner el modelo en modo entrenamiento
        epoch_loss = 0
        
        for images, masks in dataloader:
            # 1. Reiniciar gradientes
            optimizer.zero_grad()
            
            # 2. Forward: pasar las imágenes por la red
            outputs = model(images)
            
            # 3. Calcular el error comparando con la máscara real
            loss = criterion(outputs, masks)
            
            # 4. Backward: calcular cómo cambiar los pesos para mejorar
            loss.backward()
            
            # 5. Optimizar: aplicar los cambios
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}], Pérdida (Error): {epoch_loss/len(dataloader):.4f}")
    
    print("¡Entrenamiento completado!")
    return model, dataset


# =====================================================================
# 4. VISUALIZACIÓN DE RESULTADOS
# =====================================================================
def visualize_results(model, dataset, num_images=3):
    model.eval() # Modo evaluación (no se ajustan pesos)
    
    # Crear una figura con múltiples filas (una por cada imagen)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    # Si solo es una imagen, axes es 1D, lo convertimos a 2D para que el for funcione igual
    if num_images == 1:
        axes = [axes]
        
    with torch.no_grad(): # No calcular gradientes ahorra memoria
        for i in range(num_images):
            # Tomamos una imagen nueva del dataset
            image, real_mask = dataset[i] 
            
            # A PyTorch le gustan los lotes. Añadimos una dimensión extra al principio (1, C, H, W)
            input_tensor = image.unsqueeze(0) 
            predicted_mask = model(input_tensor)
            
            # Convertimos los tensores de PyTorch a arreglos de Numpy para dibujarlos
            img_np = image.squeeze().numpy()
            real_mask_np = real_mask.squeeze().numpy()
            pred_mask_np = predicted_mask.squeeze().numpy()
            
            axes[i][0].imshow(img_np, cmap='gray')
            axes[i][1].imshow(real_mask_np, cmap='gray')
            axes[i][2].imshow(pred_mask_np, cmap='gray')
            
            # Solo poner los títulos en la primera fila para que no se vea sobrecargado
            if i == 0:
                axes[i][0].set_title('Imagen de Entrada (Cuadrado con ruido)')
                axes[i][1].set_title('Máscara Real (Lo que queremos)')
                axes[i][2].set_title('Predicción de la U-Net')
                
            for ax in axes[i]:
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, eval_dataset = train_model()
    visualize_results(trained_model, eval_dataset)
