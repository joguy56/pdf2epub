# Architecture du Code Refactoré pdf2epub

## Table des Matières
1. [Vue d'ensemble](#vue-densemble)
2. [Contexte du Refactoring](#contexte-du-refactoring)
3. [Objectifs de l'Architecture](#objectifs-de-larchitecture)
4. [Structure des Dossiers](#structure-des-dossiers)
5. [Flux de Données](#flux-de-données)
6. [Principes de Conception](#principes-de-conception)
7. [Dépendances](#dépendances)

---

## Vue d'ensemble

Le projet `pdf2epub` a été refactorisé depuis un script monolithique (`pdf2epub.py`) vers une architecture modulaire et extensible. Cette refonte vise à améliorer la maintenabilité, la testabilité et l'évolutivité du code tout en préservant les fonctionnalités existantes.

### Comparaison Ancien vs Nouveau

| Aspect | Code Original | Code Refactoré |
|--------|---------------|----------------|
| **Structure** | Script monolithique ~1000 lignes | Modules séparés avec responsabilités claires |
| **OCR** | Logique Tesseract câblée en dur | Abstraction avec support multi-moteurs (Tesseract, EasyOCR) |
| **Configuration** | Arguments CLI seulement | CLI + fichiers YAML + valeurs par défaut |
| **Testabilité** | Difficile (fonctions couplées) | Facile (injection de dépendances, mocks) |
| **Extensibilité** | Modifications dispersées | Points d'extension clairs (interfaces, factories) |
| **IA Proofreading** | Absent | Support Gemini, OpenAI, Claude |

---

## Contexte du Refactoring

### Problèmes Identifiés dans le Code Original

1. **Monolithe difficile à maintenir**
   - Toutes les fonctionnalités dans un seul fichier
   - Couplage fort entre les étapes du pipeline
   - Difficile d'isoler et tester une fonctionnalité

2. **Manque d'abstraction**
   - Logique OCR spécifique à Tesseract dispersée
   - Impossible d'ajouter facilement un nouveau moteur OCR
   - Duplication de code pour les traitements similaires

3. **Configuration rigide**
   - Tous les paramètres en ligne de commande
   - Pas de profils de configuration réutilisables
   - Difficile de gérer différents cas d'usage

4. **Évolutions limitées**
   - Ajout de fonctionnalités nécessite modifications invasives
   - Pas de support pour l'amélioration IA du texte
   - Architecture ne permet pas l'extension facile

### Décisions de Refactoring

- **Modularisation** : Séparation en composants avec responsabilités uniques
- **Abstraction OCR** : Interface commune pour tous les moteurs OCR
- **Configuration YAML** : Fichiers de config réutilisables et versionnés
- **Pipeline orchestré** : Classe centrale coordonnant les étapes
- **Tests automatisés** : Suite de tests de régression et unitaires

---

## Objectifs de l'Architecture

### 1. **Séparation des Préoccupations (SoC)**
Chaque module a une responsabilité unique et bien définie :
- `pdf_processor.py` : Conversion PDF → images
- `ocr/` : Reconnaissance de texte
- `chapter_detector.py` : Détection structure documentaire
- `text_processor.py` : Nettoyage et formatage texte
- `epub_generator.py` : Génération fichier EPUB

### 2. **Principe Ouvert/Fermé (OCP)**
Les modules sont ouverts à l'extension mais fermés à la modification :
- Nouveau moteur OCR → hériter de `BaseOCREngine`
- Nouveau format de sortie → implémenter interface génération
- Nouveau preprocessing → ajouter dans `pdf_processor.py` sans casser l'existant

### 3. **Injection de Dépendances (DI)**
Les composants reçoivent leurs dépendances plutôt que les créer :
```python
pipeline = ConversionPipeline(
    pdf_processor=PDFProcessor(config),
    ocr_engine=TesseractEngine(config),
    chapter_detector=ChapterDetector(config)
)
```

### 4. **Testabilité**
Architecture conçue pour faciliter les tests :
- Interfaces mockables
- Logique métier isolée des I/O
- Tests de régression automatisés

### 5. **Configuration Flexible**
Support multiple niveaux de configuration :
1. Valeurs par défaut dans le code
2. Fichiers YAML pour profils réutilisables
3. Arguments CLI pour overrides ponctuels

---

## Structure des Dossiers

```
pdf2epub/
├── src/
│   └── pdf2epub/              # Package principal
│       ├── __init__.py         # Exports publics du package
│       ├── pipeline.py         # Orchestrateur principal du pipeline
│       ├── config.py           # Gestion configuration YAML + CLI
│       ├── cli.py              # Interface ligne de commande
│       ├── pdf_processor.py    # Conversion PDF → images
│       ├── chapter_detector.py # Détection chapitres/sections
│       ├── text_processor.py   # Nettoyage et formatage texte
│       ├── epub_generator.py   # Génération fichier EPUB
│       ├── ai_proofreader.py   # Correction IA (Gemini/OpenAI/Claude)
│       └── ocr/                # Module OCR avec abstractions
│           ├── __init__.py
│           ├── base.py         # BaseOCREngine (interface abstraite)
│           ├── tesseract.py    # Implémentation Tesseract
│           └── easyocr.py      # Implémentation EasyOCR
├── tests/                      # Tests automatisés
│   ├── test_regression.py      # Tests de non-régression
│   └── test_config.yaml        # Configuration pour tests
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Ce fichier
│   ├── MODULES.md              # Documentation détaillée modules
│   └── MIGRATION.md            # Guide migration ancien → nouveau
├── configs/                    # Configurations YAML exemples
│   ├── default.yaml
│   ├── fast.yaml               # OCR rapide, qualité moyenne
│   └── high_quality.yaml       # OCR lent, qualité maximale
├── pyproject.toml              # Métadonnées projet (Poetry)
├── setup.py                    # Installation package
├── README.md                   # Documentation utilisateur
└── QUICKSTART.md               # Guide démarrage rapide
```

### Responsabilités par Dossier

#### `src/pdf2epub/` (Code source principal)
- **Contenu** : Tous les modules Python du projet
- **Organisation** : Un fichier par responsabilité majeure
- **Principe** : Chaque module est importable et testable indépendamment

#### `src/pdf2epub/ocr/` (Abstraction OCR)
- **Contenu** : Interface OCR et implémentations concrètes
- **Organisation** : 
  - `base.py` définit le contrat (`BaseOCREngine`)
  - Chaque moteur OCR dans son propre fichier
- **Extensibilité** : Ajouter un nouveau moteur = créer nouveau fichier héritant de `BaseOCREngine`

#### `tests/` (Tests automatisés)
- **Contenu** : Tests de régression, tests unitaires
- **Organisation** : `test_*.py` découverts automatiquement par pytest
- **Validation** : Garantit que les modifications ne cassent pas les fonctionnalités existantes

#### `docs/` (Documentation technique)
- **Contenu** : Documentation architecture, modules, migration
- **Public cible** : Développeurs contribuant au projet
- **Format** : Markdown pour faciliter versioning et lecture

#### `configs/` (Profils de configuration)
- **Contenu** : Fichiers YAML pré-configurés pour cas d'usage courants
- **Usage** : `pdf2epub -c configs/fast.yaml -i input.pdf`
- **Personnalisation** : Copier et modifier selon besoins

---

## Flux de Données

### Pipeline Complet

```
┌─────────────────────────────────────────────────────────────────┐
│                    ConversionPipeline                           │
│                    (pipeline.py)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 1. CHARGEMENT CONFIGURATION                             │
    │    - Fichier YAML (optionnel)                           │
    │    - Arguments CLI (prioritaires)                       │
    │    - Valeurs par défaut                                 │
    │    → Config unifiée                                     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. CONVERSION PDF → IMAGES                              │
    │    (PDFProcessor - pdf_processor.py)                    │
    │                                                          │
    │    PDF Pages                                            │
    │      ↓                                                   │
    │    pdf2image (200 DPI) → Images PIL                     │
    │      ↓                                                   │
    │    Cropping (marges x/y) → Images rognées               │
    │      ↓                                                   │
    │    GaussianBlur → Images lissées                        │
    │      ↓                                                   │
    │    Adaptive Thresholding:                               │
    │      - Test Otsu + seuils manuels [120,140,160,180,200] │
    │      - Évaluation qualité OCR par seuil                 │
    │      - Sélection meilleur seuil                         │
    │      ↓                                                   │
    │    Images binaires (noir/blanc)                         │
    │      ↓                                                   │
    │    Page Dewarp (optionnel):                             │
    │      - Appel externe page-dewarp                        │
    │      - Paramètres: -f 1.2 -x 30 -y 50 -nb 1            │
    │      - Skip si qualité < 75                             │
    │      ↓                                                   │
    │    → JPEG images dans tmp/                              │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. RECONNAISSANCE OCR                                    │
    │    (TesseractEngine ou EasyOCREngine - ocr/)            │
    │                                                          │
    │    Images JPEG                                          │
    │      ↓                                                   │
    │    OCR Engine (Tesseract par défaut)                    │
    │      - Mode: --psm 1 (auto segmentation)               │
    │      - Langue: fra ou eng                               │
    │      - Output: données hiérarchiques                    │
    │        (page → block → paragraph → line → word)         │
    │      ↓                                                   │
    │    Analyse par bloc:                                    │
    │      - Position Y (haut/milieu/bas page)               │
    │      - Confiance mots (0-100)                          │
    │      - Ratio caractères spéciaux                        │
    │      - Patterns suspects (UPPERCASE 6+ chars)           │
    │      ↓                                                   │
    │    Filtrage junk:                                       │
    │      - Numéros de page (^ +[0-9]+ *$)                  │
    │      - Blocs faible confiance (<80)                    │
    │      - En-têtes répétitifs (titre livre)               │
    │      ↓                                                   │
    │    → Texte brut avec marqueurs spéciaux                 │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. DÉTECTION STRUCTURE                                   │
    │    (ChapterDetector - chapter_detector.py)              │
    │                                                          │
    │    Analyse position texte:                              │
    │      - Texte en haut de page (Y < 25% hauteur)         │
    │      - Texte court (< 50 chars)                        │
    │      - Après page vierge/quasi-vierge                  │
    │      ↓                                                   │
    │    Détection chapitres:                                 │
    │      - Keyword "Chapitre" (optionnel)                  │
    │      - Position + contexte                              │
    │      - Insertion marqueur @@@Titre@@@                   │
    │      ↓                                                   │
    │    Détection sections:                                  │
    │      - Texte centré milieu page                        │
    │      - Court et isolé                                   │
    │      - Insertion marqueur $$$Titre$$$                   │
    │      ↓                                                   │
    │    Détection notes de bas de page:                      │
    │      - Texte en bas de page                             │
    │      - Petite taille police                             │
    │      - Insertion marqueur ~~~Notes~~~                   │
    │      ↓                                                   │
    │    → Texte structuré avec marqueurs                     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 5. POST-TRAITEMENT TEXTE                                 │
    │    (TextProcessor - text_processor.py)                  │
    │                                                          │
    │    Corrections regex séquentielles:                     │
    │      1. Césures (mots coupés ligne)                    │
    │         Pattern: ([a-z]+) *[-—] *\n+ *([a-z]+)         │
    │         → Concaténation: mot1mot2                       │
    │      2. Accents mal reconnus                            │
    │         À → À, È → È, etc.                              │
    │      3. Dialogues                                       │
    │         Lignes débutant par — ou -                      │
    │         → Indentation 4 espaces                         │
    │      4. Lettrines (capitales ornées)                   │
    │         ^([A-Z]) ([a-z])                                │
    │         → Suppression espace: A → Ab                    │
    │      5. Espaces multiples → espace unique               │
    │      6. Lignes vides multiples → ligne vide unique      │
    │      7. Filtres custom (regex utilisateur)             │
    │      ↓                                                   │
    │    → Texte nettoyé et formaté                           │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 6. CORRECTION IA (OPTIONNEL)                             │
    │    (AIProofreader - ai_proofreader.py)                  │
    │                                                          │
    │    Si --ai-provider spécifié:                           │
    │      - Découpage texte en chunks (~4000 tokens)         │
    │      - Appel API IA (Gemini/OpenAI/Claude)              │
    │      - Prompt: correction erreurs OCR seulement         │
    │      - Recombinaison chunks                             │
    │      ↓                                                   │
    │    → Texte corrigé par IA                               │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 7. GÉNÉRATION EPUB                                       │
    │    (EPUBGenerator - epub_generator.py)                  │
    │                                                          │
    │    Parsing marqueurs:                                   │
    │      - @@@Chapitre@@@ → <h1> + nouveau fichier XHTML   │
    │      - $$$Section$$$ → <h2 class="center">             │
    │      - ~~~Notes~~~ → <div class="footnote">            │
    │      - Paragraphes → <p>                                │
    │      ↓                                                   │
    │    Création structure EPUB:                             │
    │      - content.opf (métadonnées, manifest, spine)      │
    │      - toc.ncx (table des matières NCX)                │
    │      - nav.xhtml (table des matières HTML5)            │
    │      - chapX.xhtml (contenu par chapitre)              │
    │      - styles.css (mise en forme)                       │
    │      - cover.jpg (image couverture, page 1 du PDF)     │
    │      ↓                                                   │
    │    Assemblage ZIP:                                      │
    │      - mimetype (non compressé)                         │
    │      - META-INF/container.xml                           │
    │      - OEBPS/* (tous les fichiers EPUB)                │
    │      ↓                                                   │
    │    → Fichier .epub final                                │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                    📗 EPUB GÉNÉRÉ 📗
```

### Points Clés du Flux

1. **Isolation des étapes** : Chaque étape peut être testée indépendamment
2. **Données intermédiaires** : Sauvegardées dans `tmp/` pour debug
3. **Marqueurs spéciaux** : Système de délimiteurs pour préserver structure
4. **Qualité adaptive** : Tests multiples thresholds + évaluation qualité
5. **Extensibilité** : Facile d'insérer nouvelles étapes (ex: correction IA)

---

## Principes de Conception

### 1. Pattern Strategy (OCR Engines)

**Problème** : Support de multiples moteurs OCR (Tesseract, EasyOCR) avec interface unifiée

**Solution** : Classe abstraite `BaseOCREngine` définissant le contrat

```python
# ocr/base.py
class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray, lang: str) -> Dict:
        """Reconnaît texte dans image
        
        Returns:
            Dict avec structure:
            {
                'text': str,           # Texte complet
                'confidence': float,   # Confiance moyenne
                'blocks': List[Dict]   # Blocs de texte avec positions
            }
        """
        pass
```

**Implémentations concrètes** :
- `TesseractEngine` : Utilise `pytesseract.image_to_data()` → structure hiérarchique
- `EasyOCREngine` : Utilise `easyocr.Reader.readtext()` → liste rectangles

**Avantages** :
- Pipeline agnostique du moteur OCR utilisé
- Ajout nouveau moteur = hériter de `BaseOCREngine`
- Tests unitaires avec mock OCR engine

### 2. Pattern Template Method (Pipeline)

**Problème** : Séquence d'étapes fixe mais avec variations possibles

**Solution** : Méthode `run()` définit squelette, sous-méthodes personnalisables

```python
# pipeline.py
class ConversionPipeline:
    def run(self, pdf_path: str) -> str:
        """Template method - squelette fixe"""
        self._load_config()
        images = self._convert_pdf(pdf_path)
        text = self._recognize_text(images)
        text = self._detect_structure(text)
        text = self._post_process(text)
        if self.config.ai_enabled:
            text = self._ai_correct(text)  # Étape optionnelle
        epub_path = self._generate_epub(text)
        return epub_path
```

**Avantages** :
- Ordre des étapes garanti
- Points d'extension clairs (hooks)
- Facilite traçage et logging

### 3. Pattern Dependency Injection

**Problème** : Couplage fort rend tests difficiles

**Solution** : Composants injectés via constructeur

```python
# Ancien code (couplage fort)
class Pipeline:
    def __init__(self):
        self.ocr = TesseractEngine()  # Dépendance hard-codée
        
# Nouveau code (injection)
class ConversionPipeline:
    def __init__(self, ocr_engine: BaseOCREngine):
        self.ocr = ocr_engine  # Dépendance injectée
        
# Usage production
pipeline = ConversionPipeline(ocr_engine=TesseractEngine(config))

# Usage tests
pipeline = ConversionPipeline(ocr_engine=MockOCREngine())
```

**Avantages** :
- Tests avec mocks sans modifications code
- Flexibilité configuration runtime
- Respect principe inversion de dépendances

### 4. Pattern Configuration Object

**Problème** : Trop de paramètres passés entre fonctions

**Solution** : Objet configuration centralisé

```python
# config.py
@dataclass
class Config:
    # PDF processing
    pdf_dpi: int = 200
    x_margin: int = 30
    y_margin: int = 50
    
    # OCR
    ocr_engine: str = 'tesseract'
    ocr_lang: str = 'fra'
    tesseract_path: Optional[str] = None
    
    # Chapter detection
    chapter_detection: bool = True
    chapter_threshold_pct: int = 25
    detect_only_with_keyword: bool = False
    
    # Text processing
    custom_filters: List[str] = field(default_factory=list)
    
    # AI correction
    ai_provider: Optional[str] = None
    ai_api_key: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Charge config depuis YAML"""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def merge_cli_args(self, args: Namespace) -> None:
        """Override avec args CLI (priorité max)"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
```

**Hiérarchie de configuration** :
1. Valeurs par défaut (dataclass)
2. Fichier YAML (optionnel)
3. Arguments CLI (prioritaires)

**Avantages** :
- Une seule source de vérité
- Validation centralisée
- Sérialisation facile (YAML, JSON)

### 5. Pattern Factory (OCR Engine Selection)

**Problème** : Création instance OCR selon paramètre runtime

**Solution** : Factory method basée sur string

```python
# ocr/__init__.py
def create_ocr_engine(engine_name: str, config: Config) -> BaseOCREngine:
    """Factory pour créer instance OCR engine"""
    engines = {
        'tesseract': TesseractEngine,
        'easyocr': EasyOCREngine,
    }
    
    engine_class = engines.get(engine_name.lower())
    if not engine_class:
        raise ValueError(f"Unknown OCR engine: {engine_name}")
    
    return engine_class(config)

# Usage
ocr = create_ocr_engine(config.ocr_engine, config)
```

**Avantages** :
- Point unique de création instances
- Facile d'ajouter nouveaux moteurs
- Validation noms moteurs centralisée

### 6. Gestion des Erreurs

**Principe** : Exceptions spécifiques par type d'erreur

```python
# exceptions.py (à créer)
class PDF2EPUBError(Exception):
    """Exception de base"""
    pass

class PDFConversionError(PDF2EPUBError):
    """Erreur conversion PDF → images"""
    pass

class OCRError(PDF2EPUBError):
    """Erreur reconnaissance OCR"""
    pass

class EPUBGenerationError(PDF2EPUBError):
    """Erreur génération EPUB"""
    pass
```

**Usage** :
```python
try:
    images = pdf_processor.convert(pdf_path)
except PDFConversionError as e:
    logger.error(f"PDF conversion failed: {e}")
    # Gestion spécifique erreur PDF
```

### 7. Logging Structuré

**Principe** : Logs avec contexte par module

```python
# Chaque module a son logger
import logging
logger = logging.getLogger(__name__)

# Usage dans code
logger.info(f"Processing page {page_num}/{total_pages}")
logger.debug(f"OCR confidence: {confidence:.2f}")
logger.warning(f"Low quality detected: {quality_score}")
logger.error(f"Failed to process image: {e}", exc_info=True)
```

**Configuration** :
```python
# cli.py
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Dépendances

### Dépendances de Production

```toml
[tool.poetry.dependencies]
python = "^3.10"
opencv-python = "^4.8.0"         # Traitement images
pytesseract = "^0.3.10"          # OCR Tesseract
easyocr = "^1.7.0"               # OCR EasyOCR (optionnel)
Pillow = "^10.0.0"               # Manipulation images
pdf2image = "^1.16.3"            # Conversion PDF
ebooklib = "^0.18"               # Génération EPUB
PyYAML = "^6.0"                  # Configuration YAML
colorlog = "^6.7.0"              # Logs colorés
google-generativeai = "^0.3.0"   # Gemini AI (optionnel)
openai = "^1.0.0"                # OpenAI API (optionnel)
anthropic = "^0.7.0"             # Claude API (optionnel)
```

### Dépendances de Développement

```toml
[tool.poetry.dev-dependencies]
pytest = "^7.4.0"                # Tests unitaires
pytest-cov = "^4.1.0"            # Couverture tests
black = "^23.7.0"                # Formatage code
mypy = "^1.5.0"                  # Type checking
flake8 = "^6.1.0"                # Linting
```

### Dépendances Système

- **Tesseract OCR** : `apt install tesseract-ocr tesseract-ocr-fra`
- **page-dewarp** : Outil externe pour redressement pages ([lien GitHub](https://github.com/mzucker/page_dewarp))
- **Poppler** : Pour pdf2image (`apt install poppler-utils`)

### Installation Complète

```bash
# Dépendances système (Ubuntu/Debian)
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-fra poppler-utils

# Dépendances Python
cd /chemin/vers/pdf2epub
poetry install

# Ou avec pip
pip install -e .
```

