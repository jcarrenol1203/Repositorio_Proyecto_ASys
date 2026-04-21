import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATASET: Generador de datos sintéticos (Cuadrados)
# =====================================================================


class SquareDataset(Dataset):
    def __init__(self, num_samples=100, image_size=64): # num_samples es el número de imágenes, image_size es el tamaño de la imagen
        self.num_samples = num_samples # Número de imágenes
        self.image_size = image_size # Tamaño de la imagen

    def __len__(self):
        # Le dice a PyTorch cuántos ejemplos hay en total
        return self.num_samples

    def __getitem__(self, idx):
        # Esta función genera/carga UNA sola imagen y su máscara correspondiente.
        
        # Crear imagen base en negro (ruido de fondo)
        image = np.random.normal(0, 0.1, (1, self.image_size, self.image_size)).astype(np.float32) # Crear imagen base en negro (ruido de fondo)
        # Crear máscara base en negro
        mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32) # Crear máscara base en negro

        # Definir tamaño y posición aleatoria para dos cuadrados
        for _ in range(2):
            size = np.random.randint(10, 20) # Definir tamaño y posición aleatoria de un cuadrado
            x = np.random.randint(0, self.image_size - size) # Posición aleatoria en X
            y = np.random.randint(0, self.image_size - size) # Posición aleatoria en Y

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

#Crea la red neuronal nn.Module es la clase base de Pytorch para todas las redes neuronales
class BasicUNet(nn.Module):
    def __init__(self):
        super(BasicUNet, self).__init__()
        
        # --- ENCODER (Camino de bajada - Extrae características) ---
        # Toma la imagen (1 canal) y aplica filtros para buscar patrones
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # 1 canal de entrada, 16 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True), # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # 16 canales de entrada, 16 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True) # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
        )
        # Reduce la imagen a la mitad de tamaño
        self.pool1 = nn.MaxPool2d(2) # Reduce el tamaño de la imagen a la mitad (2x2)
        
        # --- BOTTLENECK (Fondo de la U) ---
        self.bottleneck = nn.Sequential( # Capa que procesa la imagen en su tamaño más pequeño
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16 canales de entrada, 32 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True), # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 canales de entrada, 32 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True) # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
        )
        
        # --- DECODER (Camino de subida - Reconstruye la imagen) ---
        # Duplica el tamaño de la imagen
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # Duplica el tamaño de la imagen y aumenta los canales
        
        # Al subir, concatenaremos con la información del encoder (Skip Connection)
        # Por eso recibe 32 canales (16 de la subida + 16 del encoder)
        self.dec1 = nn.Sequential( # Capa que reconstruye la imagen
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 32 canales de entrada, 16 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True),  # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # 16 canales de entrada, 16 canales de salida, kernel de 3x3, padding de 1 para que no se encoja la imagen
            nn.ReLU(inplace=True) # Función de activación ReLU activa las neuronas (Vuelve a 0 los valores negativos)
        )
        
        # Capa final para sacar un solo canal (probabilidad de ser cuadrado o no)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1) # Capa final para sacar un solo canal (probabilidad de ser cuadrado o no)
        # Sigmoide aprieta los valores entre 0 y 1
        self.sigmoid = nn.Sigmoid() # Función de activación Sigmoid aprieta los valores entre 0 y 1

        #El sigmoide es una funcion que transforma los valores de la red neuronal en valores entre 0 y 1
        #El valor 0 significa que no es un cuadrado
        #El valor 1 significa que es un cuadrado

    def forward(self, x): # Paso hacia adelante (cómo fluye la información por la red)
        
        # 1. Bajada
        e1 = self.enc1(x) # Entrada de la red
        p1 = self.pool1(e1) # Reducción de tamaño
        
        # 2. Fondo
        b = self.bottleneck(p1) # Procesamiento de la imagen
        
        # 3. Subida con Skip Connection
        u1 = self.up1(b) # Aumento de tamaño
        cat1 = torch.cat([u1, e1], dim=1) # <- ¡El secreto de la U-Net! #Concatenacion de la informacion del encoder y del decoder
        d1 = self.dec1(cat1) # Reconstruccion de la imagen
        
        # 4. Salida
        out = self.out_conv(d1) # Salida de la red
        return self.sigmoid(out) # Salida con sigmoide


# =====================================================================
# 3. ENTRENAMIENTO
# =====================================================================

# Implementación explícita de la Función de Pérdida del Artículo (Ecuación 12)
# Mean Square Error (MSE)
class ArticleMSELoss(nn.Module):
    def __init__(self):
        super(ArticleMSELoss, self).__init__()
        
    def forward(self, D_theta_x_o, x_t):
        """
        Calcula matemáticamente: L(θ) = (1 / N) * Σ || x_t - D_θ(x_o) ||^2_2
        
        Donde:
        - D_theta_x_o : Salida de la red U-Net (predicción desde imagen observada con ruido)
        - x_t         : Ground truth (imagen real/máscara objetivo)
        - N           : Número total de imágenes
        """
        # 1. Calculamos la diferencia entre la verdad absoluta y la predicción: (x_t - D_θ(x_o))
        diff = x_t - D_theta_x_o
        
        # 2. Elevamos al cuadrado para calcular la norma L2 al cuadrado: || ... ||^2_2
        squared_diff = torch.pow(diff, 2)
        
        # 3. Sumamos y dividimos por N (Promedio / Mean): (1 / N) * Σ ...
        L_theta = torch.mean(squared_diff)
        
        return L_theta

def train_model():
    print("Preparando datos...")
    # Creamos el dataset y el DataLoader (este último hace los "lotes" o batches)
    dataset = SquareDataset(num_samples=1000, image_size=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("Inicializando modelo U-Net...")
    model = BasicUNet()
    
    # Reemplazamos BCELoss por nuestra función de pérdida extraída del artículo
    criterion = ArticleMSELoss() 
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
def visualize_results(model, dataset, num_images=5):
    model.eval() # Modo evaluación (no se ajustan pesos)
    
    # Crear una figura con múltiples filas (una por cada imagen)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    
    # Si solo es una imagen, axes es 1D, lo convertimos a 2D para que el for funcione igual
    if num_images == 1:
        axes = [axes]
        
    with torch.no_grad(): # No calcular gradientes ahorra memoria
        for i in range(num_images):
            # Tomamos una imagen nueva del dataset
            image, _ = dataset[i] 
            
            # A PyTorch le gustan los lotes. Añadimos una dimensión extra al principio (1, C, H, W)
            input_tensor = image.unsqueeze(0) 
            predicted_mask = model(input_tensor)
            
            # Convertimos los tensores de PyTorch a arreglos de Numpy para dibujarlos
            img_np = image.squeeze().numpy()
            pred_mask_np = predicted_mask.squeeze().numpy()
            
            axes[i][0].imshow(img_np, cmap='gray')
            axes[i][1].imshow(pred_mask_np, cmap='gray')
            
            # Solo poner los títulos en la primera fila para que no se vea sobrecargado
            if i == 0:
                axes[i][0].set_title('Imagen de Entrada (Cuadrado con ruido)')
                axes[i][1].set_title('Predicción de la U-Net')
                
            for ax in axes[i]:
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, eval_dataset = train_model()
    visualize_results(trained_model, eval_dataset, num_images=5)
