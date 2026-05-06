import torch # Importa la librería principal de PyTorch (para tensores y redes neuronales)
import torch.nn as nn # Importa el módulo de redes neuronales (Neural Networks) de PyTorch
import torch.optim as optim # Importa algoritmos de optimización (como Adam o SGD) para entrenar la red
from torch.utils.data import Dataset, DataLoader # Importa clases para manejar y cargar conjuntos de datos en lotes
import numpy as np # Importa NumPy para cálculos matemáticos y manejo de matrices
import matplotlib.pyplot as plt # Importa Matplotlib para graficar y visualizar imágenes
try: # Intenta ejecutar el bloque de código siguiente
    import cv2 # Importa OpenCV, una librería muy potente para procesamiento de imágenes
except ImportError: # Si OpenCV no está instalado, captura el error
    cv2 = None # Define cv2 como None para usar una alternativa (NumPy) más adelante
import torch.nn.functional as F # Importa funciones sin estado de PyTorch (como convoluciones o activaciones)

# =====================================================================
# 1. DATASET: Generador de datos sintéticos (Cuadrados, Círculos, Triángulos)
# =====================================================================


class ShapeDataset(Dataset):
    def __init__(self, num_samples=100, image_size=64, min_shapes=3, max_shapes=6):
        self.num_samples = num_samples
        self.image_size = image_size
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Ground Truth RGB (x_t)
        # Inicializamos imagen RGB (H, W, 3)
        clean_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        
        # Fondo con color aleatorio muy tenue
        bg_color = np.random.uniform(0.02, 0.08, size=3).astype(np.float32)
        clean_image += bg_color

        num_shapes = np.random.randint(self.min_shapes, self.max_shapes + 1)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['square', 'circle', 'triangle'])
            color = np.random.uniform(0.4, 0.9, size=3).astype(np.float32) # Color RGB aleatorio
            
            if shape_type == 'square':
                size = np.random.randint(10, 20)
                x, y = np.random.randint(0, self.image_size - size, 2)
                clean_image[y:y+size, x:x+size] = np.maximum(clean_image[y:y+size, x:x+size], color)
            
            elif shape_type == 'circle':
                radius = np.random.randint(6, 14)
                cx, cy = np.random.randint(radius, self.image_size - radius, 2)
                if cv2 is not None:
                    # CV2 usa BGR por defecto, pero aquí solo queremos colores aleatorios
                    cv2.circle(clean_image, (cx, cy), radius, color.tolist(), -1)
                else:
                    Y, X = np.ogrid[:self.image_size, :self.image_size]
                    dist = (X - cx) ** 2 + (Y - cy) ** 2
                    mask = dist <= radius ** 2
                    clean_image[mask] = np.maximum(clean_image[mask], color)
            
            else: # Triangle
                pts = np.random.randint(0, self.image_size, (3, 2))
                if cv2 is not None:
                    cv2.fillPoly(clean_image, [pts], color.tolist())
                else:
                    min_x, min_y = pts.min(axis=0)
                    max_x, max_y = pts.max(axis=0)
                    clean_image[min_y:max_y+1, min_x:max_x+1] = np.maximum(clean_image[min_y:max_y+1, min_x:max_x+1], color)

        # PyTorch espera [Canales, H, W] -> (3, 64, 64)
        clean_tensor = torch.from_numpy(clean_image).permute(2, 0, 1)
        
        # 2. Degradación RGB
        # a) Desenfoque Gaussiano (aplicado a cada canal igual usando groups=3)
        k_size = 9 
        sigma = np.random.uniform(1.2, 2.2)
        coords = torch.arange(k_size) - (k_size - 1) / 2.0
        g_1d = torch.exp(-coords.pow(2) / (2 * sigma**2))
        kernel = (g_1d.view(-1, 1) * g_1d.view(1, -1))
        kernel = (kernel / kernel.sum()).view(1, 1, k_size, k_size)
        kernel_rgb = kernel.repeat(3, 1, 1, 1) # Repetir para los 3 canales RGB

        blurred = F.conv2d(clean_tensor.unsqueeze(0), kernel_rgb, padding=k_size//2, groups=3).squeeze(0)

        # b) Ruido Aditivo Gaussiano fuerte en cada canal
        noise_std = 0.08
        observed_image = blurred + torch.randn_like(blurred) * noise_std
        
        # c) Ruido "Sal y Pimienta" (Salt & Pepper) aplicado a la imagen completa
        sp_noise = torch.rand(self.image_size, self.image_size)
        observed_image[:, sp_noise < 0.01] = 0.0 
        observed_image[:, sp_noise > 0.99] = 1.0

        observed_image = torch.clamp(observed_image, 0, 1)

        return observed_image, clean_tensor
 
 


# =====================================================================
# 2. MODELO: Arquitectura U-Net Básica
# =====================================================================

# Crea la red neuronal. nn.Module es la clase base de PyTorch para todas las redes neuronales
class BasicUNet(nn.Module):
    def __init__(self): 
        super(BasicUNet, self).__init__()
        
        # Encoder Level 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2) 
        
        # Encoder Level 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # --- DECODER Level 2 ---
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 128 (up) + 128 (skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- DECODER Level 1 ---
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 64 (up) + 64 (skip)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Capa final para sacar 3 canales (RGB)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u2 = self.up2(b)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        # Salida
        out = self.out_conv(d1)
        return self.sigmoid(out)


# =====================================================================
# 3. ENTRENAMIENTO
# =====================================================================

# Implementación explícita de la Función de Pérdida (Loss) del Artículo (Ecuación 12)
# Mean Square Error (MSE) o Error Cuadrático Medio
class ArticleMSELoss(nn.Module): # Hereda de nn.Module porque es una función matemática para la red
    def __init__(self): # Constructor
        super(ArticleMSELoss, self).__init__()
        
    def forward(self, D_theta_x_o, x_t): # Define el cálculo matemático del error
        """
        Calcula matemáticamente: L(θ) = (1 / N) * Σ || x_t - D_θ(x_o) ||^2_2
        
        Donde:
        - D_theta_x_o : Salida de la red U-Net (la predicción limpia a partir de imagen con ruido)
        - x_t         : Ground truth (imagen limpia original real)
        - N           : Número total de píxeles/imágenes
        """
        # 1. Calculamos la diferencia exacta entre la verdad absoluta y la predicción: (x_t - D_θ(x_o))
        diff = x_t - D_theta_x_o 
        
        # 2. Elevamos al cuadrado esa diferencia (potencia de 2) para penalizar errores grandes y quitar negativos
        squared_diff = torch.pow(diff, 2)
        
        # 3. Sumamos todo y dividimos por la cantidad total (eso es calcular el promedio o Mean)
        L_theta = torch.mean(squared_diff)
        
        return L_theta # Devuelve un solo número (escalar) que representa qué tan mal lo hizo la red

def visualize_dataset_sample(dataset): # Función que muestra un ejemplo del dataset ANTES de entrenar
    """Muestra una muestra del dataset de entrenamiento: imagen degradada y su ground truth."""
    observed, clean = dataset[0] # Toma la primera muestra del dataset (imagen degradada y limpia)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Crea una figura con 1 fila y 2 columnas

    # Permutamos [C, H, W] -> [H, W, C] para que Matplotlib pueda dibujarlo en color
    axes[0].imshow(observed.permute(1, 2, 0).numpy()) # Muestra la imagen degradada (RGB)
    axes[0].set_title('Entrada Degradada (Color + Ruido)') # Título de la primera columna
    axes[0].axis('off') # Oculta los ejes

    axes[1].imshow(clean.permute(1, 2, 0).numpy()) # Muestra la imagen limpia original (RGB)
    axes[1].set_title('Ground Truth (Imagen Limpia RGB)') # Título de la segunda columna
    axes[1].axis('off') # Oculta los ejes

    plt.suptitle('Ejemplo del Dataset de Entrenamiento', fontsize=14, fontweight='bold', y=0.98) # Título general
    plt.tight_layout() # Ajusta el espaciado
    plt.subplots_adjust(top=0.85) # Deja espacio arriba para que el título no se superponga
    plt.savefig('ejemplo_dataset.png') # Guarda la imagen en disco
    print("Ejemplo del dataset guardado como 'ejemplo_dataset.png'", flush=True)
    plt.show() # Muestra la ventana gráfica


def train_model(): # Función que controla todo el ciclo de aprendizaje de la red
    print("Preparando datos (Aumentando para Color)...", flush=True) 
    # Aumentamos a 800 imágenes para manejar la mayor complejidad cromática
    dataset = ShapeDataset(num_samples=800, image_size=64, min_shapes=3, max_shapes=6)
    # Lotes de 16 para mayor estabilidad en el aprendizaje de colores
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) 

    # Mostrar un ejemplo del dataset antes de entrenar
    print("Mostrando ejemplo del dataset de entrenamiento...", flush=True)
    visualize_dataset_sample(dataset)
    
    print("Inicializando modelo U-Net...", flush=True)
    model = BasicUNet() # Instanciamos nuestra red U-Net
    
    # criterion es la métrica con la que evaluaremos a la red (nuestra clase ArticleMSELoss definida arriba)
    criterion = ArticleMSELoss() 
    # Optimizador Adam con un learning rate más bajo para estabilidad en color
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    epochs = 15 # Aumentamos ligeramente las épocas
    print("Iniciando entrenamiento (esto puede tomar unos segundos)...", flush=True)
    
    total_batches = len(dataloader) # Cantidad total de lotes (ej: 200/8 = 25 lotes por época)
    for epoch in range(epochs): # Bucle que se repite por cada época
        model.train() # Pone el modelo explícitamente en "modo entrenamiento" (necesario para ciertas capas)
        epoch_loss = 0 # Variable para acumular el error total de esta época
        
        for batch_idx, (images, masks) in enumerate(dataloader): # Bucle que toma un lote de 8 imágenes degradadas y sus máscaras limpias
            # 1. Reiniciar gradiente: Borra los cálculos de derivadas del lote anterior (crucial en PyTorch)
            optimizer.zero_grad() 
            
            # 2. Forward: Pasa el lote de imágenes degradadas por la red para obtener sus predicciones
            outputs = model(images) 
            
            # 3. Calcular el error (pérdida): Compara las predicciones de la red con las máscaras limpias reales
            loss = criterion(outputs, masks) 
            
            # 4. Backward: Calcula las derivadas (gradientes) para saber en qué dirección ajustar cada peso de la red
            loss.backward() 
            
            # 5. Optimizar: Aplica matemáticamente los ajustes a los pesos (¡aquí es donde la red aprende!)
            optimizer.step() 
            
            epoch_loss += loss.item() # Suma el error de este lote al acumulado de la época (usa .item() para extraer el número del tensor)
            
            # Calcula el porcentaje de avance dentro de la época actual
            batch_pct = (batch_idx + 1) / total_batches * 100
            # Imprime el progreso lote por lote
            print(f"  Época [{epoch+1}/{epochs}] - Batch [{batch_idx+1}/{total_batches}] ({batch_pct:.1f}%) - Loss: {loss.item():.4f}", flush=True)
            
        epoch_pct = (epoch + 1) / epochs * 100 # Porcentaje total del entrenamiento
        # Al final de la época, imprime el error promedio
        print(f"[OK] Epoca [{epoch+1}/{epochs}] completada ({epoch_pct:.1f}% del entrenamiento) | Perdida promedio: {epoch_loss/total_batches:.4f}", flush=True)
        print("-" * 60, flush=True) # Separador visual en consola
    
    print("¡Entrenamiento completado!", flush=True)
    return model, dataset # Devuelve el modelo ya entrenado (con pesos ajustados) y el dataset usado


# =====================================================================
# 4. VISUALIZACIÓN DE RESULTADOS
# =====================================================================
def visualize_results(model, dataset, num_images=5): # Función para probar y ver visualmente qué tan bien aprendió el modelo
    model.eval() # Pone el modelo en "modo evaluación" (desactiva actualizaciones y fija el comportamiento)
    
    # Crea una cuadrícula de gráficos de Matplotlib: num_images filas y 2 columnas (entrada, predicción)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images)) 
    
    # Si solo pedimos ver 1 imagen, axes es una lista simple (1D). Lo forzamos a ser 2D para no romper el bucle siguiente
    if num_images == 1:
        axes = [axes]
        
    with torch.no_grad(): # Bloque que desactiva el cálculo de gradientes (ahorra muchísima memoria, porque no vamos a entrenar más)
        for i in range(num_images): # Bucle sobre cada imagen que queremos mostrar
            # Toma la imagen i del dataset. Ignora la máscara limpia (porque queremos ver la predicción de la red)
            image, _ = dataset[i] 
            
            # Las redes esperan un batch (Lote, Canales, Alto, Ancho). image tiene forma (3, 64, 64)
            # unsqueeze(0) añade esa dimensión vacía al principio (pasa de [3,64,64] a [1,3,64,64])
            input_tensor = image.unsqueeze(0) 
            predicted_mask = model(input_tensor) # Pasa la imagen por el modelo entrenado y obtiene la reconstrucción
            
            # Permutamos los canales para visualización RGB [H, W, C]
            img_np = image.permute(1, 2, 0).numpy() # Imagen de entrada degradada
            pred_mask_np = predicted_mask.squeeze(0).permute(1, 2, 0).numpy() # Imagen predicha RGB
            
            axes[i][0].imshow(img_np) # Dibuja la imagen degradada en la primera columna
            axes[i][1].imshow(pred_mask_np) # Dibuja la predicción de la U-Net en la segunda columna
            
            # Solo pone títulos en la primera fila para que la imagen final no quede sobrecargada de texto
            if i == 0:
                axes[i][0].set_title('Entrada Degradada (Color)')
                axes[i][1].set_title('Reconstrucción RGB (U-Net)')
                
            for ax in axes[i]:
                ax.axis('off') # Oculta los ejes numéricos (los bordes) de cada gráfico
    
    plt.suptitle(f'Resultados de la U-Net: {num_images} ejemplos', fontsize=16, fontweight='bold', y=0.98) # Título general
    plt.tight_layout() # Ajusta automáticamente el espaciado para que nada se superponga
    plt.subplots_adjust(top=0.92) # Deja un margen superior para que el título principal no choque
    plt.savefig('resultado_deconvolucion.png') # Guarda la imagen resultante en el disco
    print("Gráfica guardada como 'resultado_deconvolucion.png'")
    plt.show() # Muestra la ventana gráfica en la pantalla

if __name__ == "__main__": # Este bloque solo se ejecuta si corremos este archivo directamente (no si lo importamos en otro archivo)
    trained_model, eval_dataset = train_model() # Ejecuta el entrenamiento y guarda el modelo y los datos
    visualize_results(trained_model, eval_dataset, num_images=3) # Ejecuta la visualización final con 3 ejemplos
    
