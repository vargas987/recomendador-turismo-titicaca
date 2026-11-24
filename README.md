# 🏔️ Titicaca Smart Tourism

**Sistema inteligente de recomendación turística con aprendizaje continuo (MLOps) y soporte multilingüe (español, quechua y aymara) para la predicción explicable de la satisfacción del visitante en el destino Puno – Lago Titicaca.**

Autor: **Elroy Vargas** — E.P. de Ingeniería de Sistemas, Universidad Peruana Unión (UPeU), Juliaca, Puno, Perú.

🔗 **App en vivo:** https://vargas987.github.io/recomendador-turismo-titicaca/ *(se activa con GitHub Pages)*

---

## 📌 Resumen

Aplicación web (PWA instalable) que recomienda atractivos del Lago Titicaca y **predice la satisfacción del visitante** con aprendizaje automático **explicable (SHAP)**, usando **datos reales**:

- **Satisfacción** → 500 reseñas de Google Maps (319 en el periodo 2020–2025) de 5 atractivos.
- **Demanda/contexto** → estadísticas oficiales de **MINCETUR – datosTurismo**.

Es la parte práctica de una investigación que incluye una **revisión sistemática de literatura (RSL, 366 estudios, PRISMA)** y un **paper en formato IEEE**.

---

## 🧠 Modelo de satisfacción (resultados reales)

| Modelo | Accuracy | AUC | F1 (satisf.) | F1-macro |
|---|---|---|---|---|
| Random Forest | 0.854 | 0.650 | 0.920 | 0.523 |
| **XGBoost** ⭐ | 0.812 | **0.717** | 0.890 | 0.624 |
| Texto (TF-IDF) | 0.817 | 0.601 | 0.893 | 0.623 |

- **Tasa de satisfacción global:** 87.1 % (278 satisfechos / 41 insatisfechos).
- **Satisfacción por atractivo:** Amantaní 96.2 % · Sillustani 95.7 % · Taquile 92.9 % · Reserva Nac. 81.2 % · **Islas Uros 65.0 %** (la más baja, pese a ser la más visitada → quejas de precio).
- **Factores SHAP:** experiencia del reseñador, extensión del texto, temporada, atractivo.
- **Palabras de satisfacción:** bonito, lindo, hermoso, vistas, visitar, interesante.
- **Palabras de insatisfacción:** soles (precio), no, nada, frío, alto (altura/soroche).

Gráficos en [`figures/`](figures/). Datos limpios en [`data/`](data/).

---

## 📈 Modelo de demanda (Prophet)

Modelo complementario de series temporales sobre la serie mensual oficial de visitantes (MINCETUR, 72 meses, 2020–2025):

- **Pronóstico:** ≈ 184 700 visitantes (2026) y 217 200 (2027).
- **Estacionalidad:** pico en enero–febrero (verano austral / Candelaria), valle entre abril y julio.
- **Validación (reteniendo 2025):** MAPE ≈ 86 % → por el crecimiento atípico de 2025, el pronóstico es orientativo (bandas de confianza amplias).

Código: [`model/modelo_demanda.py`](model/modelo_demanda.py) · Figura: `figures/fig_demanda_forecast.png`.

---

## 🗂️ Estructura del repositorio

```
recomendador-turismo-titicaca/
├── index.html            # App web (PWA): recomendador + modelo + asistente multilingüe
├── manifest.json         # Manifiesto PWA (instalable / Play Store ready)
├── sw.js                 # Service worker (offline-first)
├── icons/                # Iconos de la app
├── model/
│   ├── modelo_satisfaccion.py   # Pipeline: limpieza, features, RF/XGBoost/TF-IDF, SHAP
│   ├── modelo_demanda.py        # Pronóstico de demanda (Prophet) con datos MINCETUR
│   └── requirements.txt
├── data/                 # reseñas limpias (2020-2025), métricas, resumen JSON
├── figures/              # 8 gráficos (satisfacción, SHAP, ROC, palabras, etc.)
└── README.md
```

---

## 🚀 Cómo usar

### App
Abre `index.html` en el navegador, o visita la versión en línea (GitHub Pages). En el celular, menú ▸ **"Instalar app"**.

### Reentrenar el modelo
```bash
cd model
pip install -r requirements.txt
python modelo_satisfaccion.py "ruta/al/archivo_de_reseñas.xlsx"
```

---

## 🔬 Metodología
CRISP-DM. Variable objetivo: `satisfecho = 1` si la reseña tiene 4–5 estrellas. División 70/30 estratificada. Interpretabilidad con SHAP (TreeExplainer). Análisis de texto con TF-IDF + regresión logística.

## 🔭 Trabajo futuro
- Sustituir TF-IDF por **BETO** (BERT en español).
- Soporte real en **quechua y aymara** con traducción automática (**NLLB**) + encuesta propia.
- **MLOps**: reentrenamiento continuo ante desvío de concepto.
- Pronóstico de demanda con series temporales (Prophet) sobre datos de MINCETUR.

## 📚 Fuentes
- MINCETUR — datosTurismo: https://datosturismo.mincetur.gob.pe/
- Reseñas públicas de Google Maps (2020–2025).

## 📄 Licencia
MIT — ver [`LICENSE`](LICENSE).
