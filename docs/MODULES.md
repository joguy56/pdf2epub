# Documentation Détaillée des Modules

## Table des Matières
1. [pipeline.py - Orchestrateur Principal](#pipelinepy---orchestrateur-principal)
2. [config.py - Gestion Configuration](#configpy---gestion-configuration)
3. [pdf_processor.py - Conversion PDF](#pdf_processorpy---conversion-pdf)
4. [ocr/ - Moteurs OCR](#ocr---moteurs-ocr)
5. [chapter_detector.py - Détection Structure](#chapter_detectorpy---détection-structure)
6. [text_processor.py - Post-traitement Texte](#text_processorpy---post-traitement-texte)
7. [epub_generator.py - Génération EPUB](#epub_generatorpy---génération-epub)
8. [ai_proofreader.py - Correction IA](#ai_proofreaderpy---correction-ia)
9. [cli.py - Interface Ligne de Commande](#clipy---interface-ligne-de-commande)

---

## pipeline.py - Orchestrateur Principal

### Responsabilité
Coordonne l'ensemble du processus de conversion PDF → EPUB en orchestrant tous les composants.

### Classe Principale : `ConversionPipeline`

```python
class ConversionPipeline:
    """Orchestrateur du pipeline de conversion PDF vers EPUB"""
    
    def __init__(
        self,
        config: Config,
        pdf_processor: PDFProcessor,
        ocr_engine: BaseOCREngine,
        chapter_detector: ChapterDetector,
        text_processor: TextProcessor,
        epub_generator: EPUBGenerator,
        ai_proofreader: Optional[AIProofreader] = None
    ):
        """
        Initialise le pipeline avec injection de dépendances.
        
        Args:
            config: Configuration globale
            pdf_processor: Convertisseur PDF → images
            ocr_engine: Moteur de reconnaissance OCR
            chapter_detector: Détecteur de structure documentaire
            text_processor: Processeur de nettoyage texte
            epub_generator: Générateur de fichiers EPUB
            ai_proofreader: Correcteur IA optionnel
        """
```

### Méthodes Publiques

#### `run(pdf_path: str, output_path: Optional[str] = None) -> str`

**Description** : Point d'entrée principal du pipeline. Exécute toutes les étapes séquentiellement.

**Paramètres** :
- `pdf_path` : Chemin vers fichier PDF source
- `output_path` : Chemin EPUB de sortie (optionnel, auto-généré si absent)

**Retour** : Chemin du fichier EPUB généré

**Flux d'exécution** :
```python
def run(self, pdf_path: str, output_path: Optional[str] = None) -> str:
    # 1. Validation entrée
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # 2. Conversion PDF → images
    logger.info("Converting PDF to images...")
    images = self.pdf_processor.process(pdf_path)
    
    # 3. OCR sur images
    logger.info(f"Running OCR with {self.config.ocr_engine}...")
    raw_text = self.ocr_engine.recognize_batch(images)
    
    # 4. Détection structure
    if self.config.chapter_detection:
        logger.info("Detecting document structure...")
        structured_text = self.chapter_detector.detect(raw_text)
    else:
        structured_text = raw_text
    
    # 5. Post-traitement texte
    logger.info("Post-processing text...")
    clean_text = self.text_processor.process(structured_text)
    
    # 6. Correction IA (optionnel)
    if self.ai_proofreader:
        logger.info(f"AI proofreading with {self.config.ai_provider}...")
        clean_text = self.ai_proofreader.correct(clean_text)
    
    # 7. Génération EPUB
    logger.info("Generating EPUB...")
    if not output_path:
        output_path = pdf_path.replace('.pdf', '.epub')
    epub_path = self.epub_generator.generate(
        text=clean_text,
        output_path=output_path,
        metadata=self._extract_metadata(pdf_path)
    )
    
    logger.info(f"✓ Conversion complete: {epub_path}")
    return epub_path
```

### Méthodes Privées

#### `_extract_metadata(pdf_path: str) -> Dict[str, str]`

Extrait métadonnées du PDF (titre, auteur) ou utilise config.

**Retour** : Dictionnaire avec clés `title`, `author`, `language`

### Exemple d'Utilisation

```python
from pdf2epub.config import Config
from pdf2epub.pipeline import ConversionPipeline
from pdf2epub.pdf_processor import PDFProcessor
from pdf2epub.ocr import create_ocr_engine
from pdf2epub.chapter_detector import ChapterDetector
from pdf2epub.text_processor import TextProcessor
from pdf2epub.epub_generator import EPUBGenerator

# Charger configuration
config = Config.from_yaml('config.yaml')

# Créer composants
pdf_proc = PDFProcessor(config)
ocr = create_ocr_engine(config.ocr_engine, config)
chapter_det = ChapterDetector(config)
text_proc = TextProcessor(config)
epub_gen = EPUBGenerator(config)

# Créer et exécuter pipeline
pipeline = ConversionPipeline(
    config=config,
    pdf_processor=pdf_proc,
    ocr_engine=ocr,
    chapter_detector=chapter_det,
    text_processor=text_proc,
    epub_generator=epub_gen
)

# Conversion
epub_path = pipeline.run('livre.pdf')
print(f"EPUB créé : {epub_path}")
```

### Points d'Extension

1. **Ajouter étape pré-OCR** : Insérer entre PDF processing et OCR
2. **Ajouter post-processing custom** : Hériter `TextProcessor` et override méthodes
3. **Logging personnalisé** : Injecter custom logger dans constructeur
4. **Gestion erreurs** : Wrapper `run()` avec try/except et retry logic

---

## pdf_processor.py - Conversion PDF

### Responsabilité
Convertit fichiers PDF en images préprocessées optimisées pour l'OCR.

### Classe Principale : `PDFProcessor`

```python
class PDFProcessor:
    """Convertit PDF en images avec preprocessing adaptatif"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration avec paramètres DPI, marges, thresholds
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
```

### Méthodes Publiques

#### `process(pdf_path: str) -> List[np.ndarray]`

**Description** : Convertit toutes les pages PDF en images préprocessées.

**Étapes détaillées** :

1. **Conversion PDF → PIL Images**
   ```python
   images_pil = pdf2image.convert_from_path(
       pdf_path,
       dpi=self.config.pdf_dpi  # Défaut: 200 DPI
   )
   ```

2. **Preprocessing par page** :
   - Conversion PIL → numpy array
   - Conversion RGB → Grayscale
   - Cropping avec marges configurables
   - GaussianBlur pour réduire bruit
   - **Adaptive thresholding** (voir détails ci-dessous)
   - Page dewarping optionnel

3. **Sauvegarde images intermédiaires**
   ```python
   # tmp/page_001_binary.jpg
   # tmp/page_001_dewarped.jpg
   ```

**Retour** : Liste d'images numpy (uint8, binaires ou grayscale)

#### `_adaptive_threshold(image: np.ndarray, page_num: int) -> Tuple[np.ndarray, int, float]`

**Description** : Teste plusieurs valeurs de seuil et sélectionne la meilleure.

**Algorithme** :
```python
def _adaptive_threshold(self, image, page_num):
    # 1. Liste seuils à tester
    thresholds = [
        ('otsu', None),        # Otsu automatique
        ('manual', 120),
        ('manual', 140),
        ('manual', 160),
        ('manual', 180),
        ('manual', 200)
    ]
    
    best_score = 0
    best_threshold = None
    best_image = None
    
    for thresh_type, value in thresholds:
        # 2. Appliquer seuil
        if thresh_type == 'otsu':
            _, binary = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                image, value, 255,
                cv2.THRESH_BINARY
            )
        
        # 3. Tester avec page-dewarp si activé
        if self.config.use_dewarp:
            dewarped = self._dewarp_image(binary, page_num)
        else:
            dewarped = binary
        
        # 4. Évaluer qualité OCR
        quality_score = self._assess_ocr_quality(dewarped)
        
        # 5. Garder meilleur
        if quality_score > best_score:
            best_score = quality_score
            best_threshold = value or 'otsu'
            best_image = dewarped
    
    return best_image, best_threshold, best_score
```

**Métrique de qualité** (score 0-100) :
```python
def _assess_ocr_quality(self, image: np.ndarray) -> float:
    # OCR rapide sur échantillon
    sample_text = pytesseract.image_to_data(
        image, lang=self.config.ocr_lang, output_type=Output.DICT
    )
    
    # Composantes du score
    confidence = np.mean([c for c in sample_text['conf'] if c > 0])
    
    text_full = ' '.join(sample_text['text'])
    special_ratio = len([c for c in text_full if not c.isalnum()]) / max(len(text_full), 1)
    
    # Pattern suspects: UPPERCASE 6+ chars (ex: "LIFTULIIURCI")
    suspicious_patterns = len(re.findall(r'\b[A-Z]{6,}\b', text_full))
    
    word_count = len([w for w in sample_text['text'] if w.strip()])
    
    # Score composite
    score = (
        confidence * 0.5 +
        (1 - special_ratio) * 100 * 0.3 +
        (word_count / 50) * 100 * 0.2
    )
    
    # Pénalités
    score -= suspicious_patterns * 30
    if confidence < 70:
        score -= 10
    
    return max(0, min(100, score))
```

#### `_dewarp_image(image: np.ndarray, page_num: int) -> np.ndarray`

**Description** : Redresse pages courbées via outil externe `page-dewarp`.

**Commande exécutée** :
```bash
page-dewarp \
    -f 1.2 \              # Focal length
    -x 30 -y 50 \         # Marges
    -nb 1 \               # No binary (entrée déjà binaire)
    tmp/page_001.jpg \
    tmp/page_001_dewarped.jpg
```

**Gestion qualité** :
```python
# Si qualité dewarped < 75 ET qualité non-dewarped > dewarped
# → Skip dewarp, utiliser image binaire directe
if dewarped_quality < 75 and non_dewarped_quality > dewarped_quality:
    return original_binary
return dewarped_image
```

### Configuration Associée

```yaml
# config.yaml - Section pdf_processing
pdf_processing:
  dpi: 200
  x_margin: 30
  y_margin: 50
  gaussian_blur_kernel: [5, 5]
  use_dewarp: true
  dewarp_focal_length: 1.2
  adaptive_threshold: true
  threshold_values: [120, 140, 160, 180, 200]
  skip_dewarp_if_quality_below: 75
```

### Exemple d'Utilisation

```python
from pdf2epub.config import Config
from pdf2epub.pdf_processor import PDFProcessor

config = Config()
config.pdf_dpi = 300  # Haute résolution
config.use_dewarp = True
config.adaptive_threshold = True

processor = PDFProcessor(config)
images = processor.process('livre_scanne.pdf')

print(f"✓ {len(images)} pages converties")
# Images prêtes pour OCR
```

### Fichiers Intermédiaires Générés

```
tmp/
├── page_001.jpg              # Image grayscale brute
├── page_001_cropped.jpg      # Après cropping
├── page_001_blurred.jpg      # Après GaussianBlur
├── page_001_binary.jpg       # Après thresholding
├── page_001_dewarped.jpg     # Après page-dewarp
├── ...
```

**Note** : Fichiers `tmp/` nettoyés automatiquement à chaque nouvelle conversion.

---

## ocr/ - Moteurs OCR

### Responsabilité
Abstraction unifiée pour multiples moteurs de reconnaissance OCR (Tesseract, EasyOCR).

### Architecture

```
ocr/
├── __init__.py          # Exports + factory function
├── base.py              # BaseOCREngine (interface abstraite)
├── tesseract.py         # TesseractEngine (implémentation)
└── easyocr.py           # EasyOCREngine (implémentation)
```

### Classe Abstraite : `BaseOCREngine`

```python
# ocr/base.py
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class BaseOCREngine(ABC):
    """Interface commune pour tous les moteurs OCR"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def recognize(self, image: np.ndarray, lang: str) -> Dict:
        """
        Reconnaît texte dans une image.
        
        Args:
            image: Image numpy array (grayscale ou RGB)
            lang: Code langue ('fra', 'eng', etc.)
        
        Returns:
            Dict avec structure standardisée:
            {
                'text': str,              # Texte complet reconnu
                'confidence': float,      # Confiance moyenne 0-100
                'blocks': List[Dict],     # Liste blocs de texte
                'page_num': int,          # Numéro page
                'page_height': int,       # Hauteur page en pixels
                'page_width': int         # Largeur page en pixels
            }
            
            Chaque bloc dans 'blocks':
            {
                'block_num': int,         # ID unique bloc
                'text': str,              # Texte du bloc
                'confidence': float,      # Confiance bloc
                'left': int,              # Position X
                'top': int,               # Position Y
                'width': int,             # Largeur
                'height': int,            # Hauteur
                'is_junk': bool          # True si détecté comme junk
            }
        """
        pass
    
    def recognize_batch(self, images: List[np.ndarray]) -> str:
        """
        Reconnaît texte sur multiple images (pages).
        
        Args:
            images: Liste d'images numpy
        
        Returns:
            Texte complet de toutes les pages concaténé
        """
        all_text = []
        for i, image in enumerate(images):
            self.logger.info(f"OCR page {i+1}/{len(images)}")
            result = self.recognize(image, self.config.ocr_lang)
            all_text.append(result['text'])
        
        return '\n\n'.join(all_text)
```

### Implémentation Tesseract : `TesseractEngine`

```python
# ocr/tesseract.py
import pytesseract
from pytesseract import Output

class TesseractEngine(BaseOCREngine):
    """Moteur OCR basé sur Tesseract"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Configuration Tesseract
        if config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = config.tesseract_path
        
        # Vérifier installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found: {e}")
    
    def recognize(self, image: np.ndarray, lang: str) -> Dict:
        # OCR avec données détaillées
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=Output.DICT,
            config='--psm 1'  # Automatic page segmentation
        )
        
        # Grouper par blocs
        blocks = self._parse_blocks(data)
        
        # Filtrer junk
        filtered_blocks = [b for b in blocks if not self._is_junk(b)]
        
        # Texte complet
        text = '\n\n'.join(b['text'] for b in filtered_blocks)
        
        # Confiance moyenne
        confidences = [b['confidence'] for b in filtered_blocks]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'text': text,
            'confidence': avg_confidence,
            'blocks': filtered_blocks,
            'page_num': data.get('page_num', [1])[0],
            'page_height': data.get('height', [0])[0],
            'page_width': data.get('width', [0])[0]
        }
    
    def _parse_blocks(self, data: Dict) -> List[Dict]:
        """Groupe données Tesseract par block_num"""
        blocks_dict = {}
        
        for i in range(len(data['text'])):
            block_num = data['block_num'][i]
            text = data['text'][i].strip()
            
            if not text:
                continue
            
            if block_num not in blocks_dict:
                blocks_dict[block_num] = {
                    'block_num': block_num,
                    'text': [],
                    'confidences': [],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                }
            
            blocks_dict[block_num]['text'].append(text)
            blocks_dict[block_num]['confidences'].append(data['conf'][i])
        
        # Convertir en liste
        blocks = []
        for block in blocks_dict.values():
            blocks.append({
                'block_num': block['block_num'],
                'text': ' '.join(block['text']),
                'confidence': np.mean(block['confidences']),
                'left': block['left'],
                'top': block['top'],
                'width': block['width'],
                'height': block['height'],
                'is_junk': False  # Sera défini par _is_junk()
            })
        
        return blocks
    
    def _is_junk(self, block: Dict) -> bool:
        """Détecte si bloc est du junk (numéro page, header, etc.)"""
        text = block['text'].strip()
        
        # Numéros de page isolés
        if re.match(r'^ +[0-9]+ *$', text):
            return True
        
        # Blocs faible confiance
        if block['confidence'] < self.config.junk_confidence_threshold:
            words = text.split()
            low_conf_ratio = sum(1 for w in words if len(w) < 3) / max(len(words), 1)
            if low_conf_ratio > 0.6:
                return True
        
        # Ratio caractères spéciaux élevé
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' -—')
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # Correspondance avec titre livre (éviter headers répétitifs)
        if hasattr(self.config, 'book_title'):
            similarity = difflib.SequenceMatcher(None, text.lower(), 
                                                 self.config.book_title.lower()).ratio()
            if similarity > 0.9:
                return True
        
        return False
```

### Implémentation EasyOCR : `EasyOCREngine`

```python
# ocr/easyocr.py
import easyocr

class EasyOCREngine(BaseOCREngine):
    """Moteur OCR basé sur EasyOCR (deep learning)"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        # Initialiser reader (télécharge modèles si nécessaire)
        self.reader = easyocr.Reader(
            [config.ocr_lang],
            gpu=config.use_gpu
        )
    
    def recognize(self, image: np.ndarray, lang: str) -> Dict:
        # EasyOCR reconnaissance
        results = self.reader.readtext(
            image,
            detail=1,  # Retourne bbox, texte, confidence
            paragraph=True  # Groupe en paragraphes
        )
        
        # Convertir au format standardisé
        blocks = []
        full_text = []
        
        for i, (bbox, text, conf) in enumerate(results):
            # bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            left = int(min(p[0] for p in bbox))
            top = int(min(p[1] for p in bbox))
            width = int(max(p[0] for p in bbox)) - left
            height = int(max(p[1] for p in bbox)) - top
            
            block = {
                'block_num': i,
                'text': text,
                'confidence': conf * 100,  # EasyOCR retourne 0-1
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'is_junk': False
            }
            
            blocks.append(block)
            full_text.append(text)
        
        return {
            'text': '\n'.join(full_text),
            'confidence': np.mean([b['confidence'] for b in blocks]),
            'blocks': blocks,
            'page_num': 1,
            'page_height': image.shape[0],
            'page_width': image.shape[1]
        }
```

### Factory Function

```python
# ocr/__init__.py
from .base import BaseOCREngine
from .tesseract import TesseractEngine
from .easyocr import EasyOCREngine

def create_ocr_engine(engine_name: str, config: Config) -> BaseOCREngine:
    """
    Factory pour créer instance OCR engine.
    
    Args:
        engine_name: 'tesseract' ou 'easyocr'
        config: Configuration
    
    Returns:
        Instance de BaseOCREngine
    
    Raises:
        ValueError: Si engine_name inconnu
    """
    engines = {
        'tesseract': TesseractEngine,
        'easyocr': EasyOCREngine,
    }
    
    engine_class = engines.get(engine_name.lower())
    if not engine_class:
        raise ValueError(
            f"Unknown OCR engine: {engine_name}. "
            f"Available: {', '.join(engines.keys())}"
        )
    
    return engine_class(config)
```

### Exemple d'Utilisation

```python
from pdf2epub.ocr import create_ocr_engine
from pdf2epub.config import Config
import cv2

config = Config()
config.ocr_engine = 'tesseract'
config.ocr_lang = 'fra'

# Créer engine
ocr = create_ocr_engine(config.ocr_engine, config)

# Charger image
image = cv2.imread('page_001.jpg', cv2.IMREAD_GRAYSCALE)

# OCR
result = ocr.recognize(image, 'fra')
print(f"Texte: {result['text']}")
print(f"Confiance: {result['confidence']:.2f}")
print(f"Blocs: {len(result['blocks'])}")
```

### Ajouter un Nouveau Moteur OCR

1. **Créer fichier** : `ocr/nouveau_moteur.py`
2. **Hériter** : `class NouveauMoteur(BaseOCREngine)`
3. **Implémenter** : `recognize()` avec retour format standardisé
4. **Enregistrer** : Ajouter dans factory `ocr/__init__.py`

```python
# ocr/google_vision.py
from google.cloud import vision

class GoogleVisionEngine(BaseOCREngine):
    def __init__(self, config):
        super().__init__(config)
        self.client = vision.ImageAnnotatorClient()
    
    def recognize(self, image, lang):
        # Appel API Google Vision
        response = self.client.text_detection(image=image)
        
        # Convertir au format standardisé
        return {
            'text': response.full_text_annotation.text,
            'confidence': 95.0,  # Google ne donne pas de confidence
            'blocks': self._parse_google_blocks(response),
            ...
        }
```

---

## chapter_detector.py - Détection Structure

### Responsabilité
Analyse le texte OCR brut et détecte la structure documentaire (chapitres, sections, notes).

### Classe Principale : `ChapterDetector`

```python
class ChapterDetector:
    """Détecte structure hiérarchique du document"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
```

### Méthodes Publiques

#### `detect(ocr_results: List[Dict]) -> str`

**Description** : Analyse résultats OCR et insère marqueurs de structure.

**Paramètres** :
- `ocr_results` : Liste de dictionnaires (un par page) issus de `BaseOCREngine.recognize()`

**Retour** : Texte avec marqueurs spéciaux :
- `@@@Titre Chapitre@@@` : Début de chapitre
- `$$$Titre Section$$$` : Section interne
- `~~~Notes de bas de page~~~` : Notes

**Algorithme détaillé** :

```python
def detect(self, ocr_results: List[Dict]) -> str:
    structured_text = []
    
    for page in ocr_results:
        page_height = page['page_height']
        page_num = page['page_num']
        
        # Seuil Y pour haut de page (défaut: 25%)
        chapter_threshold = page_height * self.config.chapter_threshold_pct / 100
        
        # Tracking état page
        text_on_top = False  # True si texte déjà rencontré en haut
        
        for block in page['blocks']:
            if block['is_junk']:
                continue
            
            text = block['text'].strip()
            y_position = block['top']
            
            # 1. DÉTECTION CHAPITRE
            if self._is_chapter(block, y_position, chapter_threshold, text_on_top):
                # Nettoyer titre (enlever "Chapitre X" si présent)
                title = self._clean_chapter_title(text)
                structured_text.append(f"\n@@@{title}@@@\n")
                text_on_top = True
                continue
            
            # 2. DÉTECTION SECTION
            if self._is_section(block, page_height):
                structured_text.append(f"\n$$${text}$$$\n")
                # Section n'active PAS text_on_top (bug corrigé!)
                continue
            
            # 3. DÉTECTION NOTES BAS DE PAGE
            if self._is_footnote(block, page_height):
                if not any('~~~' in t for t in structured_text[-3:]):
                    structured_text.append("\n~~~Notes~~~\n")
                structured_text.append(text)
                continue
            
            # 4. TEXTE NORMAL
            structured_text.append(text)
            
            # Marquer que texte apparaît en haut
            if y_position < chapter_threshold:
                text_on_top = True
        
        # Reset entre pages
        text_on_top = False
    
    return '\n\n'.join(structured_text)
```

### Méthodes Privées

#### `_is_chapter(block, y_pos, threshold, text_on_top) -> bool`

**Logique de détection chapitre** :

```python
def _is_chapter(self, block, y_pos, threshold, text_on_top):
    text = block['text'].strip()
    
    # Condition 1: Position en haut de page
    if y_pos > threshold:
        return False
    
    # Condition 2: Pas encore de texte en haut (début de page)
    if text_on_top:
        return False
    
    # Condition 3: Texte court (titre, pas paragraphe)
    if len(text) > 100:
        return False
    
    # Condition 4: Keyword "Chapitre" si option activée
    if self.config.detect_only_with_keyword:
        if not re.search(r'\bChapitre\b', text, re.IGNORECASE):
            return False
    
    # Condition 5: Haute confiance OCR
    if block['confidence'] < 70:
        return False
    
    return True
```

**Paramètres configurables** :
- `chapter_threshold_pct` : Pourcentage hauteur page (défaut: 25%)
- `detect_only_with_keyword` : Exiger mot "Chapitre" (défaut: False)

#### `_is_section(block, page_height) -> bool`

**Détection sections** :

```python
def _is_section(self, block, page_height):
    text = block['text'].strip()
    y_pos = block['top']
    
    # Au milieu de la page (entre 30% et 70%)
    if not (0.3 * page_height < y_pos < 0.7 * page_height):
        return False
    
    # Texte court et centré
    if len(text) > 50:
        return False
    
    # Approximation "centré" : position X entre 30% et 70% largeur
    page_width = block.get('page_width', 1000)
    x_center = block['left'] + block['width'] / 2
    if not (0.3 * page_width < x_center < 0.7 * page_width):
        return False
    
    return True
```

#### `_is_footnote(block, page_height) -> bool`

**Détection notes de bas de page** :

```python
def _is_footnote(self, block, page_height):
    y_pos = block['top']
    
    # En bas de page (> 85% hauteur)
    if y_pos < 0.85 * page_height:
        return False
    
    # Petite taille de texte (heuristique)
    text_height = block['height']
    if text_height > 30:  # Pixels
        return False
    
    return True
```

#### `_clean_chapter_title(text: str) -> str`

**Nettoyage titre chapitre** :

```python
def _clean_chapter_title(self, text):
    # Enlever "Chapitre 1", "Chapitre I", etc.
    text = re.sub(r'^Chapitre\s+[IVXLCDM0-9]+\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
    
    # Enlever numéros en début
    text = re.sub(r'^[0-9]+\.\s*', '', text)
    
    return text.strip()
```

### Configuration Associée

```yaml
# config.yaml - Section chapter_detection
chapter_detection:
  enabled: true
  threshold_pct: 25                    # % hauteur page pour "haut"
  detect_only_with_keyword: false      # Exiger "Chapitre"
  min_confidence: 70                   # Confiance OCR minimale
  max_title_length: 100                # Chars max pour titre
  section_y_range: [0.3, 0.7]         # Zone Y pour sections
  footnote_y_threshold: 0.85           # Seuil Y pour notes
```

### Exemple d'Utilisation

```python
from pdf2epub.chapter_detector import ChapterDetector
from pdf2epub.config import Config

config = Config()
config.chapter_detection = True
config.chapter_threshold_pct = 30  # Plus permissif

detector = ChapterDetector(config)

# OCR results (format BaseOCREngine)
ocr_results = [
    {
        'page_num': 1,
        'page_height': 2000,
        'blocks': [
            {'text': 'Chapitre 1: Le Début', 'top': 150, 'confidence': 95, 'is_junk': False},
            {'text': 'Il était une fois...', 'top': 300, 'confidence': 92, 'is_junk': False},
        ]
    }
]

structured = detector.detect(ocr_results)
print(structured)
# Output:
# @@@Le Début@@@
# 
# Il était une fois...
```

### Bug Corrigé : text_on_top

**Ancien comportement (buggé)** :
```python
if self._is_section(block):
    text_on_top = True  # ❌ Section déclenchait text_on_top
```

**Nouveau comportement (corrigé)** :
```python
if self._is_section(block):
    # ✅ Section ne modifie PAS text_on_top
    continue

# Seul le contenu réel active text_on_top
if y_position < chapter_threshold:
    text_on_top = True
```

**Impact** : Élimine faux positifs "Chapitre 2" détectés après sections.


---

## text_processor.py - Post-traitement Texte

### Responsabilité
Nettoie et formate le texte OCR brut (correction césures, accents, dialogues, etc.).

### Classe Principale : `TextProcessor`

```python
class TextProcessor:
    """Post-traite texte OCR pour améliorer lisibilité"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Caractères accentués français
ls ÀÂÄÇÉÈÊËÏÎÔÙÛÜ'
```

### Méthodes Publiques

#### `process(text: str) -> str`

**Description** : Applique toutes les corrections de texte séquentiellement.

**Pipeline de traitement** :
1. Correction césures (mots coupés entre lignes)
2. Correction accents mal reconnus
3. Formatage dialogues
4. Gestion lettrines
5. Nettoyage espaces
6. Filtres personnalisés

```python
def process(self, text: str) -> str:
    self.logger.info("Post-processing text...")
    
    # 1. Césures
    text = self._fix_hyphenation(text)
    
    # 2. Accents
    text = self._fix_accents(text)
    
    # 3. Dialogues
    text = self._format_dialogues(text)
    
    # 4. Lettrines
    text = self._fix_lettrine(text)
    
    # 5. Espaces
    text = self._clean_whitespace(text)
    
    # 6. Filtres custom
    for pattern in self.config.custom_filters:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    return text
```

### Méthodes Privées

#### `_fix_hyphenation(text: str) -> str`

**Problème** : OCR détecte mots coupés avec tiret en fin de ligne  
**Exemple** : `"extraordi-\nnaire"` → `"extraordinaire"`

```python
def _fix_hyphenation(self, text):
    # Pattern: mot + trait d'union + newline + mot
    # Supporte tiret normal (-) et cadratin (—)
    pattern = f'([a-zA-Z{self.acc}]+) *[\u002D\u2014] *\n+ *([a-zA-Z{self.acc}]+)'
    text = re.sub(pattern, r'\1\2', text)
    
    return text
```

**Avant** :
```
La prin-
cesse était trs heu-
reuse.
```

**Après** :
```
La princesse était très heureuse.
```

#### `_fix_accents(text: str) -> str`

**Problème** : OCR confond accents encodés  
**Exemple** : `"À"` (mal encodé) → `"À"` (correct)

```python
def _fix_accents(self, text):
    # Table substitutions caractères mal reconnus
    replacements = {
        'À': 'À', 'Â': 'Â', 'Ä': 'Ä',
        'Ç': 'Ç',
        'É': 'É', 'È': 'È', 'Ê': 'Ê', 'Ë': 'Ë',
        'Ï': 'Ï', 'Î': '',
        'Ô': 'Ô',
        'Ù': 'Ù', 'Û': 'Û', 'Ü': 'Ü',
        # Minuscules
        'à': 'à', '': 'â', 'ä': 'ä',
        'ç': 'ç',
        'é': 'é', 'è': 'è', 'ê': 'ê', 'ë': 'ë',
        'ï': 'ï', 'î': 'î',
        'ô': 'ô',
        'ù': 'ù', '': 'û', 'ü': 'ü'
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text
```

#### `_format_dialogues(text: str) -> str`

**Problème** : Dialogues non indentés  
**Exemple** : Lignes commençant par `—` ou `-`

```python
def _format_dialogues(self, text):
    lines = text.split('\n')
    formatted = []
    
    for line in lines:
        # Ligne de dialogue si débute par — ou -
        if re.match(r'^[\u2014\u002D]\s', line):
            # Ajouter indentation 4 espaces
            formatted.append('    ' + line)
        else:
            formatted.append(line)
    
    return '\n'.join(formatted)
```

**Avant** :
```
Il dit :
 Bonjour !
 Comment allez-vous ?
```

**Après** :
```
Il dit :
    — Bonjour !
    — Comment allez-vous ?
```

#### `_fix_lettrine(text: str) -> str`

**Problème** : Lettrines (capitales ornées) dtectées avec espace  
**Exemple** : `"L a princesse"` → `"La princesse"`

```python
def _fix_lettrine(self, text):
    # Pattern: début de ligne, capitale seule, espace, minuscule
    pattern = r'^([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜ]) ([a-zàâäçéèêëïîôùûü])'
    text = re.sub(pattern, r'\1\2', text, flags=re.MULTILINE)
    
    return text
```

**Avant** :
```
L a princesse arriva au château.
```

**Après** :
```
La princesse arriva au château.
```

#### `_clean_whitespace(text: str) -> str`

**Problème** : Espaces multiples et lignes vides excessives

```python
def _clean_whitespace(self, text):
    # Espaces multiples → espace unique
    text = re.sub(r' +', ' ', text)
    
    # Lignes vides multiples → une seule
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Espaces en début/fin de lignes
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text
```

### Configuration Associée

```yaml
# config.yaml - Section text_processing
text_processing:
  fix_hyphenation: true
  fix_accents: true
  format_dialogues: true
  fix_lettrine: true
  custom_filters:
    - '^Page [0-9]+$'          # Enlever "Page 1", "Page 2"
    - '^\d+$'                  # Enlever lignes avec juste numéros
    - 'www\.[^\s]+'            # Enlever URLs
```

### Exemple d'Utilisation

```python
from pdf2epub.text_processor import TextProcessor
from pdf2epub.config import Config

config = Config()
config.custom_filters = ['^Page [0-9]+$', '^\d+$']

processor = TextProcessor(config)

# Texte brut OCR
raw_text = """
L a princesse était extraordi-
naire.

 Bonjour ! dit-elle.

Page 42
"""

# Post-traitement
clean_text = processor.process(raw_text)
print(clean_text)
# Output:
# La princesse était extraordinaire.
# 
#     — Bonjour ! dit-elle.
```

### Tests de Régression

Le fichier `tests/test_regression.py` valide que post-traitement ne change pas output sur PDFs de référence :

```python
def test_text_processing_consistency():
    """Vérifie que post-traitement est stable"""
    ocr_text = load_reference_text('ext_cigales_ocr.txt')
    
    processor = TextProcessor(Config())
    processed = processor.process(ocr_text)
    
    expected = load_reference_text('ext_cigales_processed.txt')
    assert processed == expected
```

---

## epub_generator.py - Génération EPUB

### Responsabilité
Crée fichiers EPUB valides à partir de texte structuré avec métadonnes.

### Classe Principale : `EPUBGenerator`

```python
class EPUBGenerator:
    """Génère fichiers EPUB conformes EPUB 3.0"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
```

### Méthodes Publiques

#### `generate(text: str, output_path: str, metadata: Dict) -> str`

**Description** : Crée fichier EPUB complet.

**Paramètres** :
- `text` : Texte avec marqueurs `@@@`, `$$$`, `~~~`
- `output_path` : Chemin fichier .epub sortie
- `metadata` : Dict avec `title`, `author`, `language`, `cover_image_path`

**Retour** : Chemin fichier EPUB créé

**Étapes** :
```python
def generate(self, text, output_path, metadata):
    # 1. Créer objet EPUB
    book = epub.EpubBook()
    
    # 2. Métadonnées
    book.set_identifier(metadata.get('identifier', 'id123456'))
    book.set_title(metadata['title'])
    book.set_language(metadata.get('language', 'fr'))
    book.add_author(metadata['author'])
    
    # 3. Parser texte en chapitres
    chapters = self._parse_chapters(text)
    
    # 4. Créer fichiers XHTML par chapitre
    epub_chapters = []
    for i, chapter in enumerate(chapters):
        epub_ch = self._create_chapter(chapter, i)
        book.add_item(epub_ch)
        epub_chapters.append(epub_ch)
    
    # 5. Couverture
    if metadata.get('cover_image_path'):
        book.set_cover('cover.jpg', open(metadata['cover_image_path'], 'rb').read())
    
    # 6. CSS
    style = self._create_css()
    book.add_item(style)
    
    # 7. Table des matières
    book.toc = epub_chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    
    # 8. Spine (ordre lecture)
    book.spine = ['nav'] + epub_chapters
    
    # 9. Écrire fichier
    epub.write_epub(output_path, book)
    
    self.logger.info(f"✓ EPUB created: {output_path}")
    return output_path
```

### Méthodes Privées

#### `_parse_chapters(text: str) -> List[Dict]`

**Description** : Parse marqueurs en structure chapitres

```python
def _parse_chapters(self, text):
    chapters = []
    current_chapter = None
    
    for line in text.split('\n'):
        # Chapitre: @@@Titre@@@
        if line.startswith('@@@') and line.endswith('@@@'):
            if current_chapter:
                chapters.append(current_chapter)
            
            title = line.strip('@').strip()
            current_chapter = {
                'title': title,
                'content': [],
                'sections': []
            }
        
        # Section: $$$Titre$$$
        elif line.startswith('$$$') and line.endswith('$$$'):
            title = line.strip('$').strip()
            current_chapter['sections'].append({
                'title': title,
                'content': []
            })
        
        # Notes: ~~~
        elif line.startswith('~~~'):
            current_chapter['has_footnotes'] = True
        
        # Contenu normal
        else:
            if current_chapter:
                if current_chapter.get('has_footnotes'):
                    # Ajouter à section footnotes
                    if 'footnotes' not in current_chapter:
                        current_chapter['footnotes'] = []
                    current_chapter['footnotes'].append(line)
                else:
                    current_chapter['content'].append(line)
    
    # Dernier chapitre
    if current_chapter:
        chapters.append(current_chapter)
    
    return chapters
```

#### `_create_chapter(chapter: Dict, index: int) -> epub.EpubHtml`

**Description** : Convertit chapitre en XHTML

```python
def _create_chapter(self, chapter, index):
    # Créer fichier XHTML
    epub_chapter = epub.EpubHtml(
        title=chapter['title'],
        file_name=f'chap_{index:03d}.xhtml',
        lang='fr'
    )
    
    # HTML content
    html = f'<h1>{chapter["title"]}</h1>\n'
    
    # Paragraphes
    for para in chapter['content']:
        if para.strip():
            html += f'<p>{para}</p>\n'
    
    # Sections
    for section in chapter.get('sections', []):
        html += f'<h2 class="center">{section["title"]}</h2>\n'
        for para in section.get('content', []):
            html += f'<p>{para}</p>\n'
    
    # Notes de bas de page
    if chapter.get('footnotes'):
        html += '<div class="footnote">\n'
        for note in chapter['footnotes']:
            html += f'<p class="footnote-text">{note}</p>\n'
        html += '</div>\n'
    
    epub_chapter.content = html
    return epub_chapter
```

#### `_create_css() -> epub.EpubItem`

**Description** : CSS pour styling EPUB

```python
def _create_css(self):
    css = '''
    body {
        font-family: Georgia, serif;
        line-height: 1.6;
        margin: 2em;
    }
    
    h1 {
        text-align: center;
        margin-top: 2em;
        margin-bottom: 1em;
        font-size: 2em;
    }
    
    h2.center {
        text-align: center;
        margin-top: 1.5em;
        margin-bottom: 1em;
        font-size: 1.5em;
    }
    
    p {
        text-align: justify;
        text-indent: 1.5em;
        margin: 0;
    }
    
    .footnote {
        font-size: 0.9em;
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid #ccc;
    }
    
    .footnote-text {
        text-indent: 0;
    }
    '''
    
    style = epub.EpubItem(
        uid='style_default',
        file_name='style/default.css',
        media_type='text/css',
        content=css
    )
    
    return style
```

### Configuration Associée

```yaml
# config.yaml - Section epub_generation
epub_generation:
  include_cover: true
  font_family: 'Georgia, serif'
  line_height: 1.6
  text_align: 'justify'
  chapter_numbering: false    # Auto-numéroter chapitres
```

### Exemple d'Utilisation

```python
from pdf2epub.epub_generator import EPUBGenerator
from pdf2epub.config import Config

config = Config()
generator = EPUBGenerator(config)

# Texte structuré
text = """
@@@Le Début@@@

Il était une fois une princesse.

$$$Partie 1$$$

Elle vivait dans un château.

~~~Notes~~~
1. Chteau construit en 1245
"""

# Métadonnées
metadata = {
    'title': 'Mon Livre',
    'author': 'Auteur Inconnu',
    'language': 'fr',
    'cover_image_path': 'cover.jpg'
}

# Générer EPUB
epub_path = generator.generate(text, 'livre.epub', metadata)
print(f"✓ {epub_path}")
```

### Validation EPUB

Utiliser `epubcheck` pour valider :

```bash
# Installer epubcheck
wget https://github.com/w3c/epubcheck/releases/download/v5.0.0/epubcheck-5.0.0.zip
unzip epubcheck-5.0.0.zip

# Valider EPUB
java -jar epubcheck-5.0.0/epubcheck.jar livre.epub
```

