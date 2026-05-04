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


class ShapeDataset(Dataset): # Define una clase que hereda de Dataset de PyTorch para crear nuestros datos
    def __init__(self, num_samples=100, image_size=64, num_shapes=2): # Método constructor que inicializa el dataset
        self.num_samples = num_samples  # Define el número total de imágenes que tendrá el dataset
        self.image_size = image_size    # Define el tamaño (ancho y alto) de cada imagen cuadrada (ej: 64x64)
        self.num_shapes = num_shapes    # Define cuántas figuras geométricas habrá en cada imagen

    def __len__(self): # Método que PyTorch usa para saber el tamaño del dataset
        return self.num_samples # Retorna el número total de muestras

    def __getitem__(self, idx): # Método que PyTorch usa para obtener una imagen específica mediante su índice (idx)
        # 1. Crear la imagen limpia (Ground Truth - x_t en el paper)
        # Empezamos con un fondo negro (matriz de ceros) del tamaño especificado
        clean_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32) 

        # Dibujar formas aleatorias
        for _ in range(self.num_shapes): # Bucle para dibujar la cantidad de formas indicadas
            shape_type = np.random.choice(['square', 'circle', 'triangle']) # Elige una forma al azar entre las tres opciones
            if shape_type == 'square': # Si la forma elegida es un cuadrado...
                size = np.random.randint(10, 25) # Elige un tamaño de lado aleatorio entre 10 y 25 píxeles
                x = np.random.randint(0, self.image_size - size) # Elige una coordenada X aleatoria que no se salga de la imagen
                y = np.random.randint(0, self.image_size - size) # Elige una coordenada Y aleatoria que no se salga de la imagen
                clean_mask[y:y+size, x:x+size] = 1.0 # Rellena la zona del cuadrado con 1s (color blanco)
            elif shape_type == 'circle': # Si la forma elegida es un círculo...
                radius = np.random.randint(5, 15) # Elige un radio aleatorio entre 5 y 15 píxeles
                cx = np.random.randint(radius, self.image_size - radius) # Elige centro X aleatorio sin salir del borde
                cy = np.random.randint(radius, self.image_size - radius) # Elige centro Y aleatorio sin salir del borde
                if cv2 is not None: # Si OpenCV está instalado...
                    cv2.circle(clean_mask, (cx, cy), radius, 1.0, -1) # Dibuja un círculo relleno (grosor -1) (Circulo solido) de color blanco (1.0)
                else: # Si no está instalado OpenCV, usa NumPy como plan B
                    # Simple numpy fallback: draw a filled circle
                    Y, X = np.ogrid[:self.image_size, :self.image_size] # Crea una cuadrícula de coordenadas Y y X
                    dist = (X - cx) ** 2 + (Y - cy) ** 2 # Calcula la distancia al cuadrado desde cada píxel al centro
                    mask = dist <= radius ** 2 # Crea una máscara booleana para los píxeles dentro del radio
                    clean_mask[mask] = 1.0 # Pinta esos píxeles de blanco (1.0)
            else:  # Si la forma es un triángulo (triangle)
                pts = np.random.randint(0, self.image_size, (3, 2)) # Genera 3 puntos (vértices) aleatorios (X, Y)
                if cv2 is not None: # Si OpenCV está instalado...
                    cv2.fillPoly(clean_mask, [pts], 1.0) # Rellena el polígono formado por los 3 puntos con color blanco
                else: # Si no hay OpenCV, usa NumPy (plan B)
                    # Simple numpy fallback: draw bounding box of triangle
                    min_x, min_y = pts.min(axis=0) # Encuentra la coordenada X e Y mínima de los 3 vértices
                    max_x, max_y = pts.max(axis=0) # Encuentra la coordenada X e Y máxima
                    clean_mask[min_y:max_y+1, min_x:max_x+1] = 1.0 # Dibuja un rectángulo que envuelve al triángulo (aproximación)

        # Añadir canal de profundidad (PyTorch espera [Canales, Alto, Ancho])
        clean_mask = clean_mask[np.newaxis, :, :] # Añade una dimensión extra al principio. Pasa de (64, 64) a (1, 64, 64)

        # Convertimos la matriz de NumPy a un Tensor de PyTorch (el formato que usa la red neuronal)
        clean_tensor = torch.from_numpy(clean_mask)  # [1, H, W] (el 1 es de los canales de color, en este caso solo uno porque es blanco y negro)
        
        # 2. Simular el proceso de degradación: y = k * x + n
        # a) Desenfoque (Convolución con kernel k)
        # Creamos un kernel Gaussiano de 7x7 para emborronar la imagen
        k_size = 7 # Tamaño del kernel (ventana) que emborrona
        sigma = 2.0 # Desviación estándar (qué tan fuerte es el desenfoque)
        coords = torch.arange(k_size) - (k_size - 1) / 2.0 # Crea un vector centrado en 0 (ej: -3, -2, -1, 0, 1, 2, 3)
        g_1d = torch.exp(-coords.pow(2) / (2 * sigma**2)) # Aplica la fórmula de la campana de Gauss en 1D
        g_2d = g_1d.view(-1, 1) * g_1d.view(1, -1) # Multiplica el vector 1D consigo mismo transpuesto para crear una matriz 2D
        kernel = (g_2d / g_2d.sum()).view(1, 1, k_size, k_size) # Normaliza para que sume 1 y le da la forma (Canal_out, Canal_in, Alto, Ancho)

        # Aplicamos la convolución (desenfoque) usando el kernel Gaussiano
        # unsqueeze(0) añade la dimensión temporal de "batch" (lote) que exige F.conv2d: [1, 1, 64, 64]
        blurred = F.conv2d(clean_tensor.unsqueeze(0), kernel, padding=k_size//2) # padding mantiene el tamaño original de la imagen
        blurred = blurred.squeeze(0) # Quitamos la dimensión de lote extra añadida antes

        # b) Ruido Aditivo (n)
        # Generamos ruido gaussiano aleatorio con la misma forma que la imagen emborronada
        noise = torch.randn_like(blurred) * 0.05 # torch.randn_like crea valores normales estándar; multiplicamos por 0.05 para bajar la intensidad
        
        # Imagen observada con degradación (x_o en el paper), sumando imagen borrosa + ruido
        observed_image = blurred + noise 

        # Retornamos la imagen degradada (entrada de la red) y la limpia (objetivo a alcanzar)
        return observed_image, clean_tensor 


# =====================================================================
# 2. MODELO: Arquitectura U-Net Básica
# =====================================================================

# Crea la red neuronal. nn.Module es la clase base de PyTorch para todas las redes neuronales
class BasicUNet(nn.Module):
    def __init__(self): # Constructor donde se definen las capas de la red
        super(BasicUNet, self).__init__() # Inicializa la clase padre (nn.Module)
        
        # --- ENCODER (Camino de bajada - Extrae características) ---
        # Toma la imagen (1 canal en blanco y negro) y aplica filtros para buscar patrones
        self.enc1 = nn.Sequential( # Agrupa varias operaciones que se ejecutarán en secuencia
            # nn.Conv2d: 1 canal de entrada, 16 canales de salida (filtros), kernel de 3x3, padding de 1 para que la imagen no se encoja
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), # Función de activación ReLU: Convierte los valores negativos a 0, dejando pasar los positivos
            # Segunda convolución: 16 canales de entrada (los de antes), 16 de salida, misma configuración
            nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True) # Segunda activación ReLU
        )
        # Reduce la imagen a la mitad de tamaño (Downsampling / Pooling)
        self.pool1 = nn.MaxPool2d(2) # Toma el valor máximo en bloques de 2x2, reduciendo resolución (ej: de 64x64 a 32x32)
        
        # --- BOTTLENECK (Fondo de la U) ---
        # Capa que procesa la imagen en su tamaño más pequeño y comprimido
        self.bottleneck = nn.Sequential( 
            # 16 canales de entrada (del encoder), 32 canales de salida (más filtros para más complejidad), kernel 3x3
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), # Activación ReLU
            # 32 canales de entrada y 32 de salida
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True) # Activación ReLU
        )
        
        # --- DECODER (Camino de subida - Reconstruye la imagen) ---
        # Duplica el tamaño de la imagen usando una convolución transpuesta (o deconvolution)
        # Recibe 32 canales, devuelve 16. Kernel de 2 y stride de 2 hacen que la resolución se duplique (ej: de 32x32 vuelve a 64x64)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) 
        
        # Al subir, concatenaremos con la información del encoder (Skip Connection)
        # Por eso este bloque recibe 32 canales: 16 vienen de la capa up1 y 16 vienen directamente del encoder (enc1)
        self.dec1 = nn.Sequential( # Capa que reconstruye la imagen fusionando contexto y detalles
            # 32 canales combinados de entrada, 16 de salida
            nn.Conv2d(32, 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),  # Activación ReLU
            # 16 canales de entrada, 16 de salida
            nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True) # Activación ReLU
        )
        
        # Capa final para sacar un solo canal (la imagen final reconstruida)
        # De 16 canales pasa a 1 canal (escala de grises), usando un kernel de 1x1 (clasificación píxel por píxel)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1) 
        # Sigmoide aprieta los valores de salida para que estén estrictamente entre 0 y 1
        self.sigmoid = nn.Sigmoid() 

        # Nota: El valor 0 significa negro absoluto (fondo) y 1 significa blanco absoluto (figura)

    def forward(self, x): # Paso hacia adelante (define exactamente cómo fluye la imagen por las capas)
        
        # 1. Bajada (Encoder)
        e1 = self.enc1(x) # Pasa la imagen original 'x' por el primer bloque encoder
        p1 = self.pool1(e1) # Reduce a la mitad la resolución de 'e1'
        
        # 2. Fondo (Bottleneck)
        b = self.bottleneck(p1) # Pasa la imagen reducida por el bloque de fondo
        
        # 3. Subida con Skip Connection (Decoder)
        u1 = self.up1(b) # Duplica la resolución de lo que sale del fondo
        # torch.cat concatena (une) la imagen subida 'u1' con la imagen 'e1' del encoder en la dimensión 1 (canales)
        cat1 = torch.cat([u1, e1], dim=1) # <- ¡El secreto de la U-Net! Mezcla contexto global con detalles locales
        d1 = self.dec1(cat1) # Pasa el resultado concatenado por el bloque decoder
        
        # 4. Salida
        out = self.out_conv(d1) # Reduce los canales de 16 a 1
        return self.sigmoid(out) # Aplica Sigmoide y devuelve el resultado (la imagen limpia predicha)


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

    axes[0].imshow(observed.squeeze().numpy(), cmap='gray') # Muestra la imagen degradada (con blur + ruido)
    axes[0].set_title('Entrada Degradada (Blur + Ruido)') # Título de la primera columna
    axes[0].axis('off') # Oculta los ejes

    axes[1].imshow(clean.squeeze().numpy(), cmap='gray') # Muestra la imagen limpia original (ground truth)
    axes[1].set_title('Ground Truth (Imagen Limpia)') # Título de la segunda columna
    axes[1].axis('off') # Oculta los ejes

    plt.suptitle('Ejemplo del Dataset de Entrenamiento', fontsize=14, fontweight='bold', y=0.98) # Título general
    plt.tight_layout() # Ajusta el espaciado
    plt.subplots_adjust(top=0.85) # Deja espacio arriba para que el título no se superponga
    plt.savefig('ejemplo_dataset.png') # Guarda la imagen en disco
    print("Ejemplo del dataset guardado como 'ejemplo_dataset.png'", flush=True)
    plt.show() # Muestra la ventana gráfica


def train_model(): # Función que controla todo el ciclo de aprendizaje de la red
    print("Preparando datos...", flush=True) # Imprime un mensaje en consola
    # Creamos un dataset de 200 imágenes, de 64x64 píxeles cada una
    dataset = ShapeDataset(num_samples=200, image_size=64)
    # DataLoader envuelve el dataset para entregar las imágenes a la red en "lotes" (batches) de 8 en 8
    # shuffle=True hace que el orden sea aleatorio en cada época para evitar que la red memorice el orden
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) 

    # Mostrar un ejemplo del dataset antes de entrenar
    print("Mostrando ejemplo del dataset de entrenamiento...", flush=True)
    visualize_dataset_sample(dataset)
    
    print("Inicializando modelo U-Net...", flush=True)
    model = BasicUNet() # Instanciamos nuestra red U-Net
    
    # criterion es la métrica con la que evaluaremos a la red (nuestra clase ArticleMSELoss definida arriba)
    criterion = ArticleMSELoss() 
    # Optimizador Adam: es el "profesor" que cambia los pesos de la red basándose en el error (learning rate de 0.005) (lr es que tanto se equivoca al ajustar los pesos)
    optimizer = optim.Adam(model.parameters(), lr=0.005) 
    
    epochs = 3 # Número de veces que la red verá el conjunto completo de datos (épocas)
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
            
            # Las redes esperan un batch (Lote, Canales, Alto, Ancho). image tiene forma (Canales, Alto, Ancho)
            # unsqueeze(0) añade esa dimensión vacía al principio (pasa de [1,64,64] a [1,1,64,64])
            input_tensor = image.unsqueeze(0) 
            predicted_mask = model(input_tensor) # Pasa la imagen por el modelo entrenado y obtiene la reconstrucción
            
            # squeeze() remueve las dimensiones de tamaño 1 (lote y canales). numpy() convierte el tensor en un arreglo normal para dibujar
            img_np = image.squeeze().numpy() # Imagen de entrada degradada
            pred_mask_np = predicted_mask.squeeze().numpy() # Imagen predicha por la U-Net
            
            axes[i][0].imshow(img_np, cmap='gray') # Dibuja la imagen degradada en la primera columna
            axes[i][1].imshow(pred_mask_np, cmap='gray') # Dibuja la predicción de la U-Net en la segunda columna
            
            # Solo pone títulos en la primera fila para que la imagen final no quede sobrecargada de texto
            if i == 0:
                axes[i][0].set_title('Entrada Degradada (Blur + Ruido)')
                axes[i][1].set_title('Reconstrucción (U-Net)')
                
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
