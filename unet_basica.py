import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        # 1. Crear la imagen limpia (Ground Truth - x_t en el paper)
        # Empezamos con un fondo negro
        clean_mask = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)

        # Dibujar dos cuadrados aleatorios
        for _ in range(2):
            size = np.random.randint(10, 25)
            x = np.random.randint(0, self.image_size - size)
            y = np.random.randint(0, self.image_size - size)
            clean_mask[0, y:y+size, x:x+size] = 1.0

        # Convertimos a Tensor de PyTorch para aplicar el desenfoque
        clean_tensor = torch.from_numpy(clean_mask) # [1, H, W]
        
        # 2. Simular el proceso de degradación: y = k * x + n
        # a) Desenfoque (Convolución con kernel k)
        # Creamos un kernel Gaussiano de 7x7
        k_size = 7
        sigma = 2.0
        coords = torch.arange(k_size) - (k_size - 1) / 2.0
        g_1d = torch.exp(-coords.pow(2) / (2 * sigma**2))
        g_2d = g_1d.view(-1, 1) * g_1d.view(1, -1)
        kernel = (g_2d / g_2d.sum()).view(1, 1, k_size, k_size)

        # Aplicamos la convolución (desenfoque)
        # unsqueeze(0) añade la dimensión de batch que espera conv2d
        blurred = F.conv2d(clean_tensor.unsqueeze(0), kernel, padding=k_size//2)
        blurred = blurred.squeeze(0) # Quitamos el batch

        # b) Ruido Aditivo (n)
        # Generamos ruido gaussiano aleatorio
        noise = torch.randn_like(blurred) * 0.05
        
        # Imagen observada con degradación (x_o en el paper)
        observed_image = blurred + noise

        # Retornamos la imagen degradada (entrada) y la limpia (objetivo)
        return observed_image, clean_tensor


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
    print("Preparando datos...", flush=True)
    # Creamos el dataset y el DataLoader (este último hace los "lotes" o batches)
    dataset = SquareDataset(num_samples=200, image_size=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print("Inicializando modelo U-Net...", flush=True)
    model = BasicUNet()
    
    # Reemplazamos BCELoss por nuestra función de pérdida extraída del artículo
    criterion = ArticleMSELoss() 
    # Optimizador (el que ajusta los pesos de la red)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 3
    print("Iniciando entrenamiento (esto puede tomar unos segundos)...", flush=True)
    
    total_batches = len(dataloader)
    for epoch in range(epochs):
        model.train() # Poner el modelo en modo entrenamiento
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # 1. Reiniciar gradiente
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
            
            # Porcentaje de avance dentro de la época
            batch_pct = (batch_idx + 1) / total_batches * 100
            print(f"  Época [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{total_batches}] ({batch_pct:.1f}%) - Loss: {loss.item():.4f}", flush=True)
            
        epoch_pct = (epoch + 1) / epochs * 100
        print(f"[OK] Epoca [{epoch+1}/{epochs}] completada ({epoch_pct:.1f}% del entrenamiento) | Perdida promedio: {epoch_loss/total_batches:.4f}", flush=True)
        print("-" * 60, flush=True)
    
    print("¡Entrenamiento completado!", flush=True)
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
                axes[i][0].set_title('Entrada Degradada (Blur + Ruido)')
                axes[i][1].set_title('Reconstrucción (U-Net)')
                
            for ax in axes[i]:
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('resultado_deconvolucion.png')
    print("Gráfica guardada como 'resultado_deconvolucion.png'")
    plt.show()

if __name__ == "__main__":
    trained_model, eval_dataset = train_model()
    visualize_results(trained_model, eval_dataset, num_images=5)
