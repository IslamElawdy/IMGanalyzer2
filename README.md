# KI-Bildlabor (AI Image Lab)

Eine interaktive Streamlit-App, um Computer Vision und Deep Learning Konzepte visuell zu erlernen.

## Installation

1.  Repository klonen oder herunterladen.
2.  Python Umgebung erstellen (empfohlen Python 3.9+).
3.  Abhängigkeiten installieren:
    ```bash
    pip install -r requirements.txt
    ```

## Starten der App

Führe folgenden Befehl im Hauptverzeichnis aus:

```bash
streamlit run app.py
```

## Features

1.  **Upload & Info**: Bildanalyse und Normalisierung.
2.  **Pixel Explorer**: Interaktiver Zoom und Tensor-Werte.
3.  **Kanäle & Tensor**: RGB-Split und Histogramme.
4.  **Patch Viewer**: Convolution-Logik Schritt für Schritt.
5.  **Conv Playground**: Filter (Sobel, Blur, etc.) auf das ganze Bild.
6.  **Activation & Pooling**: ReLU, MaxPool und Informationsverlust.
7.  **Rauschen & Robustheit**: Störungen und JPEG-Artefakte.
8.  **Augmentation**: Trainingsdaten-Variationen.
9.  **Mini-CNN Features**: Einblick in Feature-Maps.
10. **Klassifikation**: MobileNetV2 + Grad-CAM Explainability.
11. **Adversarial Demo**: FGSM Attacke.
12. **Autoencoder**: Bildkompression lernen.
13. **Training Dashboard**: Live-Training auf synthetischen Daten.

## Hinweise für Entwickler

-   **Neue Module**: Erstelle eine neue Datei in `modules/` und registriere sie in `app.py` und `utils/localization.py`.
-   **Sprache**: Alle Texte werden über `utils/localization.py` verwaltet.
-   **Modelle**: Große Modelle werden cached (`st.cache_resource`).

## Offline-Nutzung

Beim ersten Start werden Modelle (MobileNetV2, ImageNet-Klassen) heruntergeladen. Danach funktioniert die App komplett offline (vorausgesetzt die Dateien bleiben im Cache/Verzeichnis).
