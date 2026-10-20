# IMT2220 – Cálculo para Ciencia de Datos (2026-2)
## Tarea Computacional: Optimización en Problemas de Datos

**Profesor:** Eduardo Cerpa (`eduardo.cerpa@uc.cl`)  
**Ayudante:** Kevin Cardemil (`kevin.cardemil@uc.cl`)  
**Institución:** Pontificia Universidad Católica de Chile  
**Instituto de Ingeniería Matemática y Computacional (IMC)**  

---

## 📌 Descripción General

Este repositorio contiene la solución completa, rigurosa y verificada a la **Tarea Computacional** del curso **IMT2220: Cálculo para Ciencia de Datos**. 

El trabajo aborda dos problemas fundamentales de Ciencia de Datos resueltos exclusivamente mediante herramientas de optimización del curso (cálculo matricial, gradientes, matrices Hessianas, formas cuadráticas, análisis espectral y descenso del gradiente):

1. **Problema 1: Reconstrucción Inversa de Imágenes y Regularización de Tikhonov**
   - Planteamiento mediante mínimos cuadrados y ecuaciones normales $A^T A x = A^T b$.
   - Demostración de convexidad vía Hessiana semidefinida positiva $H_{J_0} = 2 A^T A$ y Taylor exacto.
   - Demostración de inestabilidad de la solución directa $x_0 = A^{-1} b$ por amplificación catastrófica de ruido en direcciones con valores propios diminutos ($\mu_{\min} \approx 2\cdot 10^{-16}$).
   - Regularización de Tikhonov ($L_2$ / Ridge): acotamiento uniforme del ruido vía $\frac{\|\eta\|_2}{2\sqrt{\lambda}}$ y selección de $\lambda$ óptimo mediante el principio de discrepancia de Morozov.
   - Análisis de ruido impulsivo (sensores dañados) y resolución mediante mínimos cuadrados ponderados (**BONUS**).

2. **Problema 2: Red Neuronal con Dos Capas Ocultas y Descenso del Gradiente**
   - Propiedades de la función de activación sigmoide ($\sigma'(t) = \sigma(t)(1-\sigma(t))$, cota máxima de $1/4$ en $t=0$).
   - Deducción analítica manual del gradiente $\nabla_\theta J(\theta) \in \mathbb{R}^6$ mediante la regla de la cadena sobre el grafo computacional (retropropagación).
   - Verificación de derivadas con aproximación por diferencias finitas centradas ($h = 10^{-5}$, error absoluto $E \approx 3\cdot 10^{-11}$).
   - Entrenamiento con descenso del gradiente ($K = 20\,000$ iteraciones, $\alpha = 1$).
   - Análisis dinámico de la función de pérdida: caída inicial, meseta transitoria en la varianza de los datos $\frac{1}{2}\mathrm{Var}(y)$ (predicción de la media) y convergencia final.
   - Estudio de sensibilidad frente al tamaño de paso $\alpha$ y frente al rango de inicialización (gradientes desvanecientes / saturación de la sigmoide).
   - Demostración teórica de la limitación estructural de representatividad: prueba de que la red es invariablemente **monótona** en $\mathbb{R}$ e incapacidad de aproximar funciones no monótonas (en forma de "V").

---

## 📁 Estructura del Repositorio

```text
├── Problema1_Reconstruccion_Imagen.ipynb   # Notebook con código y desarrollo del Problema 1
├── Problema2_Red_Neuronal.ipynb           # Notebook con código y desarrollo del Problema 2
├── datos_P1.npz                           # Datos: matriz A, vector b y vector b_def (descargar de Canvas)
├── datos_P2.csv                           # Datos de entrenamiento para f (descargar de Canvas)
├── datos_P2_g.csv                         # Datos de entrenamiento para f_tilde (descargar de Canvas)
├── TareaComputacional_IMT2220_2026-2.pdf  # Enunciado oficial de la tarea
├── .gitignore                             # Filtro de exclusión de archivos pesados/temporales
└── README.md                              # Documentación completa del proyecto
```

---

## 🔬 Síntesis Matemática y Resultados

### Problema 1: Reconstrucción de Imágenes

- **Ecuaciones normales y Hessiana:**
  $$J_0(x) = \|Ax - b\|_2^2 \implies \nabla J_0(x) = 2 A^T(Ax - b) = 0 \iff A^T A x = A^T b$$
  $$H_{J_0} = 2 A^T A \succeq 0 \quad (\forall h, \; h^T H_{J_0} h = 2\|Ah\|_2^2 \ge 0)$$
  Al ser $J_0$ cuadrático, la aproximación de Taylor de 2º orden es exacta para todo $h$:
  $$J_0(x_0 + h) = J_0(x_0) + \|Ah\|_2^2 \ge J_0(x_0)$$
  Por tanto, **todo punto crítico es un mínimo global**. Como todos los $\mu_i > 0$, el minimizador es estrictamente único.

- **Inestabilidad de la solución directa ($x_0$):**
  $$\langle x_0 - x^*, v_i \rangle = \frac{\langle A v_i, \eta \rangle}{\mu_i} \implies |\langle x_0 - x^*, v_i \rangle| \le \frac{\|\eta\|_2}{\sqrt{\mu_i}}$$
  El error típico en la dirección $v_i$ es $\frac{\sigma}{\sqrt{\mu_i}}$. Para $\mu_{\min} \approx 2\cdot 10^{-16}$, $\frac{\sigma}{\sqrt{\mu_{\min}}} \approx 7\cdot 10^5$. La suma acumulada sobre las 4096 direcciones produce $\|x_0 - x^*\|_2 \approx \sigma\sqrt{\sum_i \frac{1}{\mu_i}} \approx 10^6$, explicando por qué $\|x_0\|_2 = 1\,300\,698$ (cuando una imagen normal con valores en $[0,1]$ tiene norma máxima $\sqrt{4096} = 64$).

- **Regularización de Tikhonov:**
  $$x_\lambda = (A^T A + \lambda I)^{-1} A^T b$$
  $$\langle x_\lambda, v_i \rangle = \frac{\mu_i}{\mu_i + \lambda} \langle x^*, v_i \rangle + \frac{\langle A v_i, \eta \rangle}{\mu_i + \lambda}$$
  Usando $\mu_i + \lambda \ge 2\sqrt{\mu_i \lambda}$, el término de ruido queda acotado uniformemente por:
  $$\left| \frac{\langle A v_i, \eta \rangle}{\mu_i + \lambda} \right| \le \frac{\|\eta\|_2}{2\sqrt{\lambda}}$$
  Aplicando el **criterio de discrepancia de Morozov** ($\|A x_\lambda - b\|_2 \approx \delta = 0.64$), se encuentra $\lambda \approx 4\cdot 10^{-3}$, lo que confirma que $\lambda = 10^{-2.5} \approx 3.16\cdot 10^{-3}$ es la elección óptima (residuo $= 0.616$).

- **Píxeles dañados y BONUS (Mínimos Cuadrados Ponderados):**
  El sensor dañado introduce ruido impulsivo con $\|\tilde{\eta}\|_2 \approx \|\tilde{b} - b\|_2 = 11.20 \gg \delta = 0.64$ (17.5 veces mayor). Tikhonov estándar fracasa porque ningún $\lambda$ puede balancear este ruido sin borrar la imagen. 
  Identificando los 435 píxeles dañados (10.6%) donde $\tilde{b}_p \in \{0, 1\}$ y asignando $w_p = 0$, se resuelve:
  $$(A^T W A + \lambda I) x_W = A^T W \tilde{b}$$
  La solución recupera la imagen con una diferencia relativa de solo **1.5%** respecto a la obtenida con el sensor intacto.

---

### Problema 2: Red Neuronal con Dos Capas Ocultas

- **Arquitectura:**
  $$z_1 = a_1 x + b_1, \quad h_1 = \sigma(z_1)$$
  $$z_2 = a_2 h_1 + b_2, \quad h_2 = \sigma(z_2)$$
  $$\hat{y}(x; \theta) = c + v h_2$$
  Parámetros: $\theta = (a_1, b_1, a_2, b_2, v, c) \in \mathbb{R}^6$.

- **Derivada de la sigmoide:**
  $$\sigma'(t) = \sigma(t)(1 - \sigma(t)), \quad 0 < \sigma'(t) \le \frac{1}{4} \quad (\text{máximo en } t=0)$$

- **Gradiente analítico por Grafo Computacional:**
  Para $e_i = \hat{y}_i - y_i$, definimos las sensibilidades hacia atrás:
  $$\delta_{2,i} = e_i \cdot v \cdot \sigma'(z_{2,i}), \quad \delta_{1,i} = \delta_{2,i} \cdot a_2 \cdot \sigma'(z_{1,i})$$
  $$\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^N \begin{pmatrix} \delta_{1,i} x_i \\ \delta_{1,i} \\ \delta_{2,i} h_{1,i} \\ \delta_{2,i} \\ e_i h_{2,i} \\ e_i \end{pmatrix}$$
  El gradiente analítico coincide con diferencias centradas ($h = 10^{-5}$) con un error máximo $E = 3.13\cdot 10^{-11}$.

- **Dinámica del Descenso del Gradiente ($\alpha = 1$):**
  La curva de pérdida presenta 3 etapas bien diferenciadas:
  1. *Fase de ajuste de escala*: rápida reducción desde $J \approx 3.15$ hasta $\approx 0.46$ en las primeras 10 iteraciones.
  2. *Meseta de predicción trivial*: se estaciona temporalmente en $J \approx 0.4597 \approx \frac{1}{2}\mathrm{Var}(y)$, prediciendo la media constante $\bar{y} \approx 1.20$.
  3. *Escape y convergencia*: entre las iteraciones 100 y 1000 la red orienta sus pesos para capturar la variación de $x$, convergiendo a $J(\theta_{20000}) = 0.001344$.

- **Sensibilidad y saturación:**
  - $\alpha = 0.01$: se queda atrapado en la meseta durante las 20 000 iteraciones ($J \approx 0.4589$).
  - $\alpha = 2.5$: sobrepasa el paso crítico y diverge a `NaN`.
  - Inicialización en $[-5, 5]$: 9 de 10 corridas convergen a $J \approx 0.00134$, pero la semilla 6 queda estancada en $J \approx 0.4597$ debido a que la segunda neurona se satura en $z_2 \approx -7.7$ ($\sigma'(z_2) \approx 0.0004$), conllevando a gradientes que desaparecen ($\|\nabla J\| \approx 3\cdot 10^{-5}$).

- **Teorema de Monotonía de la Red:**
  Por la regla de la cadena:
  $$\frac{d\hat{y}}{dx} = v \cdot a_1 \cdot a_2 \cdot \sigma'(z_1) \cdot \sigma'(z_2)$$
  Dado que $\sigma'(t) > 0$ para todo $t \in \mathbb{R}$, el signo de $\frac{d\hat{y}}{dx}$ es idéntico al signo de la constante $v a_1 a_2$. Por tanto:
  - Si $v a_1 a_2 > 0$, $\hat{y}$ es estrictamente creciente.
  - Si $v a_1 a_2 < 0$, $\hat{y}$ es estrictamente decreciente.
  - Si $v a_1 a_2 = 0$, $\hat{y}$ es constante.
  En cualquier caso, **la función $x \mapsto \hat{y}(x;\theta)$ es monótona**. Como los datos de $\tilde{f}$ forman una curva en "V" (medias por intervalos: $+2.27 \to -0.09 \to +1.19$), ninguna función monótona puede ajustarlos. El error se estanca en $J \approx 0.069$ (50 veces superior), evidenciando una limitación estructural insalvable de esta arquitectura.

---

## 🚀 Requisitos e Instalación

Para ejecutar los notebooks se requiere Python 3.10+ y las bibliotecas estándar permitidas en el enunciado:

```bash
pip install numpy matplotlib pandas
```

> **Nota:** Según las instrucciones oficiales del curso, no se utilizan bibliotecas prohibidas (como `scipy.optimize`, `sklearn`, `torch`, `tensorflow`, `jax`, etc.). Toda la propagación hacia adelante, retropropagación de gradientes, descenso del gradiente y funciones de pérdida fueron programadas manualmente desde cero.

---

## 🖨️ Instrucciones de Entrega (Canvas)

Para generar los archivos PDF requeridos:
1. Abra cada notebook (`.ipynb`) en VS Code y ejecute todas las celdas de principio a fin (**Run All**).
2. Exporte cada notebook a HTML:
   - Menú `···` en la barra superior del notebook $\to$ **Exportar** $\to$ **HTML**.
3. Abra el archivo `.html` generado en el navegador web (Chrome/Edge) y elija **Imprimir $\to$ Guardar como PDF**.
4. Suba a Canvas los 5 archivos requeridos:
   - `Informe_IMT2220_Tarea.pdf` (desarrollo y respuestas teóricas)
   - `Problema1_Reconstruccion_Imagen.ipynb`
   - `Problema2_Red_Neuronal.ipynb`
   - `Problema1_Reconstruccion_Imagen.pdf` (notebook renderizado con gráficos y salidas)
   - `Problema2_Red_Neuronal.pdf` (notebook renderizado con gráficos y salidas)
