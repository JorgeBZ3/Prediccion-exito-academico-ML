# Predicción del éxito académico en educación superior

Proyecto final de **Aprendizaje Automático** · Grado en Ingeniería Matemática e Inteligencia Artificial · Comillas ICAI · Curso 2025/2026

Análisis de un dataset real de **4.424 estudiantes** para anticipar el abandono y el rendimiento académico a partir de datos disponibles antes o al terminar el primer semestre. El proyecto combina tres enfoques: clasificación, regresión y aprendizaje no supervisado.

## Resumen de resultados

| Tarea | Modelo | Resultado principal |
|---|---|---|
| Clasificación multiclase (Abandono / Matriculado / Graduado) | Regresión logística multinomial **implementada desde cero** | Accuracy 72,8 % · F1-macro 0,61 en test |
| Regresión de la nota media del 2.º semestre | OLS, gradiente descendente y SVR (kernel RBF) | SVR: R² = 0,447 frente a 0,356 de OLS |
| Segmentación de perfiles | K-Means + PCA (k = 4) | Silhouette 0,27, estable con 10 semillas |

**Hallazgo principal:** el rendimiento en el primer semestre (`nota_media_1sem` y `asignaturas_1sem_aprobadas`) es el predictor más potente en las tres tareas. El análisis no supervisado identifica, sin usar la variable objetivo, un grupo de alto riesgo (870 estudiantes) en el que el **79,3 % acaba abandonando** y que es detectable con datos del primer semestre.

## Datos

- **Fuente:** Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student Dropout and Academic Success*. Data, 7(11), 146. [doi.org/10.3390/data7110146](https://doi.org/10.3390/data7110146). Dataset público disponible en el UCI Machine Learning Repository.
- **Tamaño:** 4.424 instancias y 37 variables (demográficas, académicas previas, socioeconómicas, macroeconómicas y de rendimiento del 1.º y 2.º semestre). Sin valores nulos.
- **Variable objetivo desbalanceada:** Graduado 49,9 % · Abandono 32,1 % · Matriculado 17,9 %. Por eso la métrica principal es el **F1-macro** y no la accuracy.

## Metodología

**Preprocesado común**
- One-hot encoding (de 25 a 147 columnas en clasificación).
- Estandarización con `StandardScaler` ajustado **solo con el conjunto de entrenamiento**, para evitar fuga de información.
- Permutación aleatoria y split 80/20 reproducible (3.539 / 885 instancias en clasificación).
- **Prevención de data leakage** en regresión: se excluyen todas las variables del 2.º semestre y la variable objetivo.
- En regresión se filtran los 870 estudiantes con nota 0 en el 2.º semestre, porque representan ausencia de actividad y no rendimiento académico.

### Tarea 1 · Clasificación
Regresión logística multinomial (softmax) escrita desde cero a partir de una versión binaria propia: pesos matriciales (n, K), entropía cruzada categórica y decisión por `argmax`. Incluye regularización L1, L2 y ElasticNet. El modelo final usa Ridge (C = 0,1), tasa de aprendizaje 0,1 y 1.000 iteraciones, con convergencia estable (la pérdida baja de 1,099 a 0,627).

| Clase | Precisión | Recall | F1 |
|---|---|---|---|
| Abandono | 0,77 | 0,72 | 0,74 |
| Graduado | 0,76 | 0,91 | 0,83 |
| Matriculado | 0,36 | 0,18 | 0,24 |

La clase *Matriculado* es la más difícil porque es transitoria y se solapa con las otras dos. La diferencia entre el F1-macro de train (0,65) y test (0,61) indica un ligero sobreajuste, esperable con 147 variables.

### Tarea 2 · Regresión
Predicción de `nota_media_2sem` con solo información previa al 2.º semestre (2.843 instancias de entrenamiento y 711 de test).

| Modelo | R² train | R² test | RMSE test | MAE test |
|---|---|---|---|---|
| Regresión lineal (OLS) | 0,421 | 0,356 | 1,131 | 0,851 |
| Regresión lineal (gradiente descendente) | 0,358 | 0,303 | 1,176 | 0,895 |
| SVR (kernel RBF) | 0,565 | 0,447 | 1,048 | 0,787 |

El SVR mejora a los modelos lineales en todas las métricas, lo que sugiere relaciones no lineales entre los predictores y la nota.

### Tarea 3 · Aprendizaje no supervisado
K-Means sobre 14 variables de perfil (7 numéricas y 7 categóricas binarias) tras PCA con 11 componentes (93 % de la varianza). Se eligió **k = 4** por silhouette (máximo de 0,2656). Se validó con 10 semillas distintas (desviación estándar < 0,01) y se contrastó con clustering jerárquico (Ward, silhouette 0,2547), con más del 90 % de coincidencia en tres de los cuatro clusters.

| Cluster | Perfil | Estudiantes | Abandono | Graduado |
|---|---|---|---|---|
| 0 | Intermedio | 110 | 29,1 % | 49,1 % |
| 1 | Alto riesgo | 870 | 79,3 % | 9,3 % |
| 2 | Exitoso | 2.895 | 16,3 % | 63,5 % |
| 3 | Maduro (edad media 35,5) | 549 | 41,2 % | 43,2 % |

## Limitaciones

- Los modelos son lineales o de baja complejidad. 
- El dataset no recoge motivación, carga laboral ni situación familiar, que previsiblemente explican parte de la varianza residual.

## Cómo ejecutarlo

Probado con **Python 3.12**.

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<nombre-del-repo>.git
cd <nombre-del-repo>

# 2. Crear y activar el entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Instalar dependencias
python -m pip install -r requirements.txt
```


```bash
# Los resultados del proyecto se encuentran en los notebooks
# abrir los notebooks de cada tarea
```

### Dependencias principales

`numpy 2.5.3` · `pandas 3.0.6` · `scikit-learn 1.9.1` · `matplotlib 3.11.2` · `seaborn 0.13.2` · `ipykernel==7.4.0`

## Estructura del repositorio

<!-- TODO: ajusta a tu estructura real -->
```
.
├── data/                  # dataset 
├── notebooks/             # análisis y experimentos de cada tarea
├── src/                   # implementación propia de la regresión logística multinomial
├── requirements.txt
└── README.md
```

