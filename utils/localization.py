# Dictionary for localization
# Keys are identifiers, values are dictionaries with 'de' and 'en' keys

TRANSLATIONS = {
    'app_title': {
        'de': 'IMGanalyzer',
        'en': 'IMGanalyzer'
    },
    'sidebar_settings': {
        'de': 'Einstellungen',
        'en': 'Settings'
    },
    'language': {
        'de': 'Sprache / Language',
        'en': 'Sprache / Language'
    },
    'choose_module': {
        'de': 'Modul wählen',
        'en': 'Choose Module'
    },
    'module_upload': {
        'de': '1. Upload & Basisinfos',
        'en': '1. Upload & Basic Info'
    },
    'module_pixel': {
        'de': '2. Pixel & Werte (Explorer)',
        'en': '2. Pixel & Values (Explorer)'
    },
    'module_channels': {
        'de': '3. Kanäle & Tensor-3D',
        'en': '3. Channels & Tensor-3D'
    },
    'module_patch': {
        'de': '4. Patch / Receptive Field',
        'en': '4. Patch / Receptive Field'
    },
    'module_conv': {
        'de': '5. Convolution Playground',
        'en': '5. Convolution Playground'
    },
    'module_activation': {
        'de': '6. Activation & Pooling',
        'en': '6. Activation & Pooling'
    },
    'module_noise': {
        'de': '7. Rauschen & Robustheit',
        'en': '7. Noise & Robustness'
    },
    'module_augmentation': {
        'de': '8. Data Augmentation Studio',
        'en': '8. Data Augmentation Studio'
    },
    'module_cnn_features': {
        'de': '9. Mini-CNN Feature Maps',
        'en': '9. Mini-CNN Feature Maps'
    },
    'module_classification': {
        'de': '10. Klassifikation & Explainability',
        'en': '10. Classification & Explainability'
    },
    'module_adversarial': {
        'de': '11. Adversarial Demo',
        'en': '11. Adversarial Demo'
    },
    'module_autoencoder': {
        'de': '12. Autoencoder',
        'en': '12. Autoencoder'
    },
    'module_training': {
        'de': '13. Training Dashboard',
        'en': '13. Training Dashboard'
    },
    'module_detection': {
        'de': '14. Object Detection',
        'en': '14. Object Detection'
    },
    'module_style': {
        'de': '15. Style Transfer',
        'en': '15. Style Transfer'
    },
    'module_segmentation': {
        'de': '16. Segmentation',
        'en': '16. Segmentation'
    },
    'module_basics': {
        'de': '17. Bild-Mathematik & Formate',
        'en': '17. Image Basics & Formats'
    },
    'module_manipulation': {
        'de': '18. Basis-Manipulationen',
        'en': '18. Basic Manipulations'
    },
    'module_cnn_explainer': {
        'de': '19. CNN-Schichten Erklärer',
        'en': '19. CNN Layer Explainer'
    },
    'upload_image': {
        'de': 'Bild hochladen (JPG, PNG)',
        'en': 'Upload Image (JPG, PNG)'
    },
    'no_image_warning': {
        'de': 'Bitte lade zuerst ein Bild hoch oder nutze ein Beispielbild (falls verfügbar).',
        'en': 'Please upload an image first or use a sample image (if available).'
    },
    'full_resolution': {
        'de': 'Volle Auflösung nutzen (kann langsam sein)',
        'en': 'Use full resolution (may be slow)'
    },
    'grayscale': {
        'de': 'In Graustufen konvertieren',
        'en': 'Convert to Grayscale'
    },
    'normalization': {
        'de': 'Normalisierung',
        'en': 'Normalization'
    },
    'tensor_preview': {
        'de': 'Tensor Vorschau',
        'en': 'Tensor Preview'
    },
    'show_raw_values': {
        'de': 'Zeige Rohwerte (erste 5x5 Pixel)',
        'en': 'Show raw values (first 5x5 pixel)'
    },
    'load_sample': {
        'de': 'Beispielbild laden (Synthetisch)',
        'en': 'Load Sample Image (Synthetic)'
    },
    'pixel_intro': {
        'de': 'Zoome ins Bild, um einzelne Pixel und ihre Werte zu sehen.',
        'en': 'Zoom into the image to see individual pixels and their values.'
    },
    'crop_size': {
        'de': 'Ausschnitt-Größe',
        'en': 'Crop Size'
    },
    'selected_pixel': {
        'de': 'Ausgewählter Pixel',
        'en': 'Selected Pixel'
    },
    'original_uint8': {
        'de': 'Original (uint8)',
        'en': 'Original (uint8)'
    },
    'normalized_val': {
        'de': 'Normalisiert',
        'en': 'Normalized'
    },
    'tensor_value_pytorch': {
        'de': 'Tensor Wert (PyTorch)',
        'en': 'Tensor Value (PyTorch)'
    },
    'pixel_explanation': {
        'de': 'Erklärung: Im Tensor (C, H, W) sind die Farbkanäle die erste Dimension.',
        'en': 'Explanation: In the tensor representation (C, H, W), the color channels are the first dimension.'
    },
    'image_grayscale': {
        'de': 'Bild ist Graustufen (1 Kanal).',
        'en': 'Image is Grayscale (1 Channel).'
    },
    'color_stack_intro': {
        'de': 'Ein Farbbild ist ein Stapel von 3 Matrizen (Rot, Grün, Blau).',
        'en': 'A color image is a stack of 3 matrices (Red, Green, Blue).'
    },
    'separation': {
        'de': 'R, G, B Trennung',
        'en': 'R, G, B Separation'
    },
    '3d_view': {
        'de': '3D Interpretation',
        'en': '3D Interpretation'
    },
    'histograms': {
        'de': 'Histogramme',
        'en': 'Histograms'
    },
    'stack_viz_intro': {
        'de': 'Visualisiere die Stapel-Struktur für einen kleinen Ausschnitt (10x10):',
        'en': 'Visualize the stack structure for a small random patch (10x10):'
    },
    'color_dist': {
        'de': 'Farbverteilung',
        'en': 'Color Distribution'
    },
    'patch_size': {
        'de': 'Patch-Größe (Kernel)',
        'en': 'Patch Size (Kernel Size)'
    },
    'center_x': {
        'de': 'Mittelpunkt X',
        'en': 'Center X'
    },
    'visual_heatmap': {
        'de': 'Visuell (Heatmap)',
        'en': 'Visual (Heatmap)'
    },
    'matrix_values': {
        'de': 'Matrix-Werte',
        'en': 'Matrix Values'
    },
    'conv_op_title': {
        'de': 'Faltung (Convolution) - Skalarprodukt',
        'en': 'Convolution Operation (Dot Product)'
    },
    'apply_filter': {
        'de': 'Filter anwenden',
        'en': 'Apply Filter'
    },
    'calculation': {
        'de': 'Berechnung:',
        'en': 'Calculation:'
    },
    'conv_result': {
        'de': 'Faltungsergebnis',
        'en': 'Convolution Result'
    },
    'mode': {
        'de': 'Modus',
        'en': 'Mode'
    },
    'choose_filter': {
        'de': 'Filter wählen',
        'en': 'Choose Filter'
    },
    'kernel_shape': {
        'de': 'Kernel Form',
        'en': 'Kernel Shape'
    },
    'kernel_heatmap': {
        'de': 'Kernel Heatmap',
        'en': 'Kernel Heatmap'
    },
    'same_padding': {
        'de': 'Same Padding (Output-Größe = Input-Größe)',
        'en': 'Same Padding (Output size = Input size)'
    },
    'result': {
        'de': 'Ergebnis',
        'en': 'Result'
    },
    'clip_warning': {
        'de': 'Hinweis: Werte sind für die Anzeige auf [0, 1] begrenzt, aber der Tensor kann negative Werte enthalten.',
        'en': 'Note: Values are clipped to [0, 1] for display, but the tensor contains negative values or values > 1.'
    },
    'act_relu_title': {
        'de': '1. Aktivierungsfunktion: ReLU',
        'en': '1. Activation Function: ReLU'
    },
    'relu_desc': {
        'de': 'ReLU (Rectified Linear Unit) setzt alle negativen Werte auf Null: `max(0, x)`.',
        'en': 'ReLU (Rectified Linear Unit) sets all negative values to zero: `max(0, x)`.'
    },
    'relu_output': {
        'de': 'Output (nach ReLU)',
        'en': 'Output (after ReLU)'
    },
    'diff_lost': {
        'de': 'Differenz (Was verloren ging)',
        'en': 'Difference (What was lost)'
    },
    'pooling_title': {
        'de': '2. Pooling (Downsampling)',
        'en': '2. Pooling (Downsampling)'
    },
    'pooling_type': {
        'de': 'Pooling Typ',
        'en': 'Pooling Type'
    },
    'recon_upsample': {
        'de': '3. Rekonstruktion (Upsampling)',
        'en': '3. Reconstruction (Upsampling)'
    },
    'recon_desc': {
        'de': 'Versuch, die originale Auflösung wiederherzustellen.',
        'en': 'Trying to get the original resolution back.'
    },
    'info_loss_mse': {
        'de': 'Informationsverlust (MSE)',
        'en': 'Information Loss (MSE)'
    },
    'noise_type': {
        'de': 'Rausch-Typ',
        'en': 'Noise Type'
    },
    'original': {
        'de': 'Original',
        'en': 'Original'
    },
    'distorted': {
        'de': 'Gestört',
        'en': 'Distorted'
    },
    'difference': {
        'de': 'Differenz',
        'en': 'Difference'
    },
    'aug_intro': {
        'de': 'Data Augmentation erhöht die Vielfalt der Daten durch zufällige Transformationen.',
        'en': 'Data augmentation increases the diversity of your data by applying random transformations.'
    },
    'active_aug': {
        'de': 'Aktive Augmentierungen',
        'en': 'Active Augmentations'
    },
    'aug_preview': {
        'de': 'Vorschau (Batch von 8)',
        'en': 'Augmentation Preview (Batch of 8)'
    },
    'gen_new_variations': {
        'de': 'Neue Variationen generieren',
        'en': 'Generate New Variations'
    },
    'aug_note': {
        'de': 'Im Training sieht das neuronale Netz jede Epoche eine andere Version des Bildes!',
        'en': 'In training, a neural network sees a different version of the image every epoch!'
    },
    'random_seed': {
        'de': 'Zufalls-Seed (Ändern für andere Filter)',
        'en': 'Random Seed (Change to see different filters)'
    },
    'input_tensor': {
        'de': 'Input Tensor',
        'en': 'Input Tensor'
    },
    'l1_maps': {
        'de': 'Layer 1 Feature Maps (16 Kanäle)',
        'en': 'Layer 1 Feature Maps (16 Channels)'
    },
    'l1_desc': {
        'de': 'Diese Features erkennen einfache Muster wie Kanten und Farben.',
        'en': 'These features detect simple patterns like edges and colors.'
    },
    'l2_maps': {
        'de': 'Layer 2 Feature Maps (32 Kanäle, Downsampled)',
        'en': 'Layer 2 Feature Maps (32 Channels, Downsampled)'
    },
    'l2_desc': {
        'de': 'Diese Features kombinieren frühere Muster zu komplexeren Formen.',
        'en': 'These features combine earlier patterns to detect more complex shapes.'
    },
    'mobilenet_intro': {
        'de': 'Nutze MobileNetV2 (vortrainiert auf ImageNet, 1000 Klassen).',
        'en': 'Using MobileNetV2 pretrained on ImageNet (1000 classes).'
    },
    'top5_pred': {
        'de': 'Top-5 Vorhersagen',
        'en': 'Top-5 Predictions'
    },
    'explainability_title': {
        'de': 'Erklärbarkeit (Grad-CAM)',
        'en': 'Explainability (Grad-CAM)'
    },
    'gradcam_desc': {
        'de': 'Warum hat das Netzwerk so entschieden? Wir visualisieren den Fokus.',
        'en': 'Why did the network decide this? We visualize the regions it focused on.'
    },
    'explain_class': {
        'de': 'Klasse erklären',
        'en': 'Explain Class'
    },
    'gen_heatmap': {
        'de': 'Heatmap generieren',
        'en': 'Generate Heatmap'
    },
    'overlay': {
        'de': 'Überlagerung',
        'en': 'Overlay'
    },
    'fgsm_intro': {
        'de': 'Fast Gradient Sign Method (FGSM) Demo.',
        'en': 'Fast Gradient Sign Method (FGSM) Demo.'
    },
    'edu_warning': {
        'de': 'Nur zu Lehrzwecken.',
        'en': 'Educational purpose only.'
    },
    'orig_pred': {
        'de': 'Original Vorhersage',
        'en': 'Original Prediction'
    },
    'noise_x50': {
        'de': 'Rauschen (x50 zur Sichtbarkeit)',
        'en': 'Noise (x50 for visibility)'
    },
    'adversarial_pred': {
        'de': 'Adversarial (Pred: ...)',
        'en': 'Adversarial (Pred: ...)'
    },
    'attack_success': {
        'de': 'Angriff erfolgreich! Klasse geändert.',
        'en': 'Attack Successful! Class changed.'
    },
    'attack_failed': {
        'de': 'Angriff fehlgeschlagen (oder Epsilon zu klein).',
        'en': 'Attack Failed (or epsilon too small).'
    },
    'ae_intro': {
        'de': 'Trainiere einen Autoencoder auf Patches deines Bildes, um Kompression zu lernen.',
        'en': 'Train an Autoencoder on patches of your image to learn compression.'
    },
    'latent_dim': {
        'de': 'Latente Dimension',
        'en': 'Latent Dimension'
    },
    'train_retrain': {
        'de': 'Trainieren / Neu trainieren (50 Epochen)',
        'en': 'Train / Retrain (50 Epochs)'
    },
    'training_complete': {
        'de': 'Training abgeschlossen!',
        'en': 'Training Complete!'
    },
    'reconstructed': {
        'de': 'Rekonstruiert',
        'en': 'Reconstructed'
    },
    'recon_error': {
        'de': 'Rekonstruktionsfehler',
        'en': 'Reconstruction Error'
    },
    'training_intro': {
        'de': 'Echtzeit-Training Demo auf synthetischen Daten (Quadrate vs Kreise).',
        'en': 'Real-time training demo on a synthetic dataset (Squares vs Circles).'
    },
    'lr': {
        'de': 'Lernrate',
        'en': 'Learning Rate'
    },
    'epochs': {
        'de': 'Epochen',
        'en': 'Epochs'
    },
    'start_training': {
        'de': 'Training starten',
        'en': 'Start Training'
    },
    'training_progress': {
        'de': 'Trainingsfortschritt...',
        'en': 'Training Progress...'
    },
    'training_finished': {
        'de': 'Training beendet!',
        'en': 'Training Finished!'
    },
    'evaluation': {
        'de': 'Evaluation',
        'en': 'Evaluation'
    },
    'test_examples': {
        'de': 'Test Beispiele',
        'en': 'Test Examples'
    },
    'detect_intro': {
        'de': 'Objekterkennung lokalisiert Objekte im Bild (Bounding Boxes) und klassifiziert sie.',
        'en': 'Object detection localizes objects in the image (Bounding Boxes) and classifies them.'
    },
    'style_intro': {
        'de': 'Style Transfer überträgt den künstlerischen Stil eines Bildes auf ein anderes.',
        'en': 'Style Transfer transfers the artistic style of one image to another.'
    },
    'segmentation_intro': {
        'de': 'Segmentierung unterteilt das Bild in sinnvolle Bereiche.',
        'en': 'Segmentation divides the image into meaningful regions.'
    },
    'math_norm': {
        'de': 'Normalisierung: Transformiert Pixelwerte (z.B. [0, 255]) in einen Bereich, den das neuronale Netz erwartet (z.B. [0, 1] oder [-1, 1]).',
        'en': 'Normalization: Transforms pixel values (e.g. [0, 255]) into a range expected by the neural network (e.g. [0, 1] or [-1, 1]).'
    },
    'math_pixel': {
        'de': 'Ein digitales Bild ist eine Matrix aus Pixeln. Jeder Pixel hat Werte für Rot, Grün und Blau (RGB).',
        'en': 'A digital image is a matrix of pixels. Each pixel has values for Red, Green, and Blue (RGB).'
    },
    'math_tensor': {
        'de': 'Ein Tensor ist eine n-dimensionale Matrix. Für Bilder nutzen wir (Channels, Height, Width).',
        'en': 'A tensor is an n-dimensional matrix. For images, we use (Channels, Height, Width).'
    },
    'math_conv': {
        'de': 'Faltung ist die elementweise Multiplikation eines Kernels mit einem Bildausschnitt, gefolgt von der Summe.',
        'en': 'Convolution is the element-wise multiplication of a kernel with an image patch, followed by the sum.'
    },
    'math_relu': {
        'de': 'ReLU (Rectified Linear Unit) ist eine nicht-lineare Funktion: f(x) = max(0, x).',
        'en': 'ReLU (Rectified Linear Unit) is a non-linear function: f(x) = max(0, x).'
    },
    'math_pooling': {
        'de': 'Pooling reduziert die Dimensionen, indem es Zusammenfassungen bildet (z.B. Max oder Average).',
        'en': 'Pooling reduces dimensions by aggregating values (e.g. Max or Average).'
    },
    'math_noise': {
        'de': 'Rauschen simuliert Störungen. Additives Rauschen: I_noisy = I + Noise.',
        'en': 'Noise simulates disturbances. Additive noise: I_noisy = I + Noise.'
    },
    'math_aug': {
        'de': 'Affine Transformationen (Rotation, Skalierung, Translation) werden durch Matrizen dargestellt.',
        'en': 'Affine transformations (rotation, scaling, translation) are represented by matrices.'
    },
    'math_feature': {
        'de': 'Feature Maps zeigen, wo bestimmte Muster (Kanten, Texturen) im Bild erkannt wurden.',
        'en': 'Feature maps show where specific patterns (edges, textures) were detected in the image.'
    },
    'math_softmax': {
        'de': 'Softmax konvertiert Logits in Wahrscheinlichkeiten, die sich zu 1 summieren.',
        'en': 'Softmax converts logits into probabilities that sum to 1.'
    },
    'math_fgsm': {
        'de': 'Fast Gradient Sign Method (FGSM) addiert Rauschen in Richtung des Gradienten, um den Loss zu maximieren.',
        'en': 'Fast Gradient Sign Method (FGSM) adds noise in the direction of the gradient to maximize loss.'
    },
    'math_ae': {
        'de': 'Ein Autoencoder minimiert den Rekonstruktionsfehler (MSE) zwischen Input und Output.',
        'en': 'An autoencoder minimizes the reconstruction error (MSE) between input and output.'
    },
    'math_backprop': {
        'de': 'Backpropagation berechnet Gradienten (Ableitungen) des Fehlers bezüglich der Gewichte (Kettenregel).',
        'en': 'Backpropagation calculates gradients (derivatives) of the error with respect to weights (chain rule).'
    },
    'math_detection': {
        'de': 'Object Detection sagt (x, y, w, h) und die Klasse für jedes Objekt vorher.',
        'en': 'Object detection predicts (x, y, w, h) and the class for each object.'
    },
    'math_style': {
        'de': 'Style Transfer minimiert Content-Loss (Inhalt) und Style-Loss (Gram-Matrizen).',
        'en': 'Style Transfer minimizes Content Loss (content) and Style Loss (Gram matrices).'
    },
    'math_seg': {
        'de': 'Segmentierung weist jedem Pixel eine Klasse zu (dichte Vorhersage).',
        'en': 'Segmentation assigns a class to every pixel (dense prediction).'
    },
    'basics_intro': {
        'de': 'Hier lernst du die mathematischen Grundlagen von digitalen Bildern: Matrizen, Farbräume und Datentypen.',
        'en': 'Here you learn the mathematical basics of digital images: matrices, color spaces, and data types.'
    },
    'manipulation_intro': {
        'de': 'Grundlegende Bildmanipulationen: Geometrie, Filter und Mischungen.',
        'en': 'Basic image manipulations: Geometry, filters, and blending.'
    },
    'cnn_expl_intro': {
        'de': 'Visualisiere den Datenfluss durch ein CNN: Von Normalisierung bis zu Feature Maps.',
        'en': 'Visualize data flow through a CNN: From normalization to feature maps.'
    },
    'nn_step': {
        'de': 'Analyseschritt wählen',
        'en': 'Select Analysis Step'
    },
    'step_norm': {
        'de': '1. Normalisierung',
        'en': '1. Normalization'
    },
    'step_conv': {
        'de': '2. Faltung (Convolution)',
        'en': '2. Convolution (Filtering)'
    },
    'step_relu': {
        'de': '3. Aktivierung (ReLU)',
        'en': '3. Activation (ReLU)'
    },
    'step_pool': {
        'de': '4. Pooling',
        'en': '4. Pooling'
    },
    'step_flat': {
        'de': '5. Flattening',
        'en': '5. Flattening'
    },
    'step_cnn': {
        'de': '6. Mini-CNN',
        'en': '6. Mini-CNN'
    },
    'norm_expl': {
        'de': 'Neuronale Netze arbeiten am besten mit kleinen Zahlenwerten (meist zwischen 0 und 1). Die Normalisierung hilft dem Modell, schneller und stabiler zu lernen. Wir teilen dazu jeden Pixelwert durch 255.',
        'en': 'Neural networks work best with small numerical values (usually between 0 and 1). Normalization helps the model learn faster and more stably. We divide pixel values by 255.'
    },
    'flat_expl': {
        'de': 'Convolutional Layers geben 3D-Volumen aus (Höhe, Breite, Kanäle). Um eine Entscheidung zu treffen (z.B. "Ist das eine Katze?"), müssen wir diese Merkmale in einen klassischen "Dense Layer" speisen. Flattening wandelt die 3D-Struktur in einen langen 1D-Vektor um.',
        'en': 'Convolutional layers output 3D volumes (Height, Width, Channels). To make a final decision (e.g., "Is this a cat?"), we need to connect these features to a standard Dense Layer. Flattening converts the 3D structure into a 1D vector.'
    },
    'input_shape': {
        'de': 'Eingabe-Form',
        'en': 'Input Shape'
    },
    'output_shape': {
        'de': 'Ausgabe-Form',
        'en': 'Output Shape'
    },
    'select_filter': {
        'de': 'Filter wählen',
        'en': 'Select Filter'
    },
    'kernel_values': {
        'de': 'Kernel-Werte (Gewichte)',
        'en': 'Kernel Values (Weights)'
    },
    'conv_expl': {
        'de': 'Die Convolution (Faltung) ist das Herzstück eines CNN. Eine kleine Matrix (Filter/Kernel) gleitet über das Bild, um Merkmale wie Kanten, Texturen oder Muster zu erkennen. Die Ausgabe zeigt, wo diese Merkmale im Bild vorhanden sind.',
        'en': 'Convolution is the core of a CNN. A small matrix (Filter/Kernel) slides over the image to detect features like edges, textures, or patterns. The output shows where these features are present.'
    },
    'relu_expl': {
        'de': 'ReLU (Rectified Linear Unit) ist eine Aktivierungsfunktion. Sie ersetzt alle negativen Werte durch 0. Dies führt "Nichtlinearität" ein, wodurch das Netzwerk komplexe Muster statt nur linearer Zusammenhänge lernen kann.',
        'en': 'ReLU (Rectified Linear Unit) is an activation function. It replaces all negative values with 0. This introduces "non-linearity", allowing the network to learn complex patterns instead of just linear combinations.'
    },
    'pool_expl': {
        'de': 'Pooling reduziert die räumlichen Dimensionen (Breite, Höhe) der Feature Maps. Das spart Rechenleistung und macht das Netzwerk robuster gegenüber kleinen Verschiebungen im Bild. Max Pooling nimmt den höchsten Wert in einem Fenster.',
        'en': 'Pooling reduces the spatial dimensions (Width, Height) of the feature maps. This reduces computation and makes the network robust to small shifts in the image. Max Pooling takes the largest value in a window.'
    },
    'original_map': {
        'de': 'Vorher',
        'en': 'Before'
    },
    'processed_map': {
        'de': 'Nachher',
        'en': 'After'
    },
    'cnn_expl': {
        'de': 'Ein echtes CNN lernt seine Filter während des Trainings selbst. Hier simulieren wir die erste Schicht mit 6 spezifischen Filtern (wie Kantenerkennung). Jeder Filter erzeugt eine "Feature Map", die verschiedene Aspekte des Bildes hervorhebt.',
        'en': 'A real CNN learns its own filters during training. Here, we simulate the first layer with 6 specific filters (like Edge Detectors). Each filter produces a "Feature Map" highlighting different aspects of the image.'
    },
    'fmap': {
        'de': 'Feature Map',
        'en': 'Feature Map'
    },
    'mean': {
        'de': 'Mittelwert',
        'en': 'Mean'
    },
    'original_gray': {
        'de': 'Original (Graustufen)',
        'en': 'Original (Grayscale)'
    },
    'fmap_out': {
        'de': 'Feature Map (Ausgabe)',
        'en': 'Feature Map (Output)'
    },
    'min_val': {
        'de': 'Min Wert',
        'en': 'Min Value'
    },
    'example_patch': {
        'de': 'Beispiel: Kleiner 4x4 Ausschnitt',
        'en': 'Example: Small 4x4 Patch'
    },
    'pixel_view_mode': {
        'de': 'Ansichtsmodus',
        'en': 'View Mode'
    },
    'selected_pixel': {
        'de': 'Ausgewählter Pixel',
        'en': 'Selected Pixel'
    }
}

def get_text(key, lang='de'):
    """Retrieve text for a given key and language. fallback to key if not found."""
    if key in TRANSLATIONS:
        return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get('de', key))
    return key
