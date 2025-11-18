# Guide de Migration : Code Monolithique → Architecture Refactorisée

## Vue d'Ensemble

Ce guide explique comment migrer du script `pdf2epub.py` monolithique vers l'architecture modulaire refactorisée.

**Avantages de la migration** :
- ✅ Code testable et maintenable
- ✅ Support multi-moteurs OCR (Tesseract, EasyOCR)
- ✅ Configuration YAML réutilisable
- ✅ Correction IA optionnelle (Gemini, OpenAI, Claude)
- ✅ Extensibilité facilitée

**Compatibilité** : L'architecture refactorisée produit **exactement les mêmes résultats** que le code original (validé par tests de régression).

---

## Comparaison Utilisation

### Ancien Code (Monolithe)

```bash
# Conversion complète
python pdf2epub.py \
    -i livre.pdf \
    -a "Victor Hugo" \
    -t "Les Misérables" \
    -l fra \
    -x 30 -y 50

# OCR seulement
python pdf2epub.py -i livre.pdf --recognize-only

# EPUB seulement
python pdf2epub.py -i livre.pdf --generate-epub-only
```

### Nouveau Code (Refactorisé)

```bash
# Conversion complète
pdf2epub convert livre.pdf \
    --author "Victor Hugo" \
    --title "Les Misérables" \
    --language fra

# Avec fichier de configuration
pdf2epub convert livre.pdf --config configs/high_quality.yaml

# Avec correction IA
pdf2epub convert livre.pdf \
    --ai-provider gemini \
    --ai-api-key YOUR_KEY

# OCR seulement
pdf2epub ocr livre.pdf --output livre_ocr.txt

# EPUB seulement
pdf2epub epub livre_ocr.txt \
    --title "Les Misérables" \
    --author "Victor Hugo" \
    --output livre.epub
```

---

## Migration Pas-à-Pas

### Étape 1 : Installation

```bash
# Cloner branche refactorisée
cd /chemin/vers/pdf2epub
git checkout feature/refactored-architecture

# Installer dépendances (Poetry recommandé)
poetry install

# Ou avec pip
pip install -e .

# Vérifier installation
pdf2epub --version
```

### Étape 2 : Configuration YAML

Créer fichier de configuration réutilisable au lieu d'arguments CLI répétitifs.

**Ancien** (arguments CLI répétés) :
```bash
python pdf2epub.py -i book1.pdf -a "Author" -l fra -x 30 -y 50
python pdf2epub.py -i book2.pdf -a "Author" -l fra -x 30 -y 50
python pdf2epub.py -i book3.pdf -a "Author" -l fra -x 30 -y 50
```

**Nouveau** (config YAML) :
```yaml
# my_config.yaml
pdf_processing:
  dpi: 200
  x_margin: 30
  y_margin: 50
  use_dewarp: true

ocr:
  engine: tesseract
  language: fra
  tesseract_path: /usr/bin/tesseract

chapter_detection:
  enabled: true
  threshold_pct: 25
  detect_only_with_keyword: false

text_processing:
  fix_hyphenation: true
  fix_accents: true
  format_dialogues: true
  custom_filters:
    - '^Page [0-9]+$'

epub_generation:
  include_cover: true
  font_family: 'Georgia, serif'
```

Utilisation :
```bash
pdf2epub convert book1.pdf --config my_config.yaml --author "Author"
pdf2epub convert book2.pdf --config my_config.yaml --author "Author"
pdf2epub convert book3.pdf --config my_config.yaml --author "Author"
```

### Étape 3 : Migration Scripts Existants

#### Cas 1 : Script Simple

**Ancien** :
```python
import subprocess

# Appel externe au script
subprocess.run([
    'python', 'pdf2epub.py',
    '-i', 'livre.pdf',
    '-a', 'Victor Hugo',
    '-t', 'Les Misérables',
    '-l', 'fra'
])
```

**Nouveau** :
```python
from pdf2epub import convert_pdf_to_epub

# API Python directe
convert_pdf_to_epub(
    pdf_path='livre.pdf',
    author='Victor Hugo',
    title='Les Misérables',
    language='fra'
)
```

#### Cas 2 : Traitement par Lots

**Ancien** :
```python
import os
import subprocess

for pdf_file in os.listdir('books/'):
    if pdf_file.endswith('.pdf'):
        subprocess.run([
            'python', 'pdf2epub.py',
            '-i', f'books/{pdf_file}',
            '-l', 'fra'
        ])
```

**Nouveau** :
```python
from pathlib import Path
from pdf2epub import ConversionPipeline
from pdf2epub.config import Config

# Charger config une fois
config = Config.from_yaml('my_config.yaml')

# Créer pipeline
pipeline = ConversionPipeline.from_config(config)

# Traiter tous les PDFs
for pdf_file in Path('books/').glob('*.pdf'):
    try:
        epub_path = pipeline.run(str(pdf_file))
        print(f"✓ {pdf_file.name} → {epub_path}")
    except Exception as e:
        print(f"✗ {pdf_file.name}: {e}")
```

#### Cas 3 : Intégration Avancée

**Ancien** (impossible avec script monolithique) :
```python
# Pas de moyen de personnaliser le pipeline
```

**Nouveau** (injection de composants custom) :
```python
from pdf2epub import ConversionPipeline
from pdf2epub.config import Config
from pdf2epub.pdf_processor import PDFProcessor
from pdf2epub.ocr import TesseractEngine
from pdf2epub.text_processor import TextProcessor

# Créer TextProcessor personnalisé
class CustomTextProcessor(TextProcessor):
    def process(self, text):
        # Appeler traitement de base
        text = super().process(text)
        
        # Ajouter logique custom
        text = self._remove_publisher_watermark(text)
        return text
    
    def _remove_publisher_watermark(self, text):
        return text.replace("Copyright Editeur X", "")

# Assembler pipeline avec composant custom
config = Config.from_yaml('config.yaml')
pipeline = ConversionPipeline(
    config=config,
    pdf_processor=PDFProcessor(config),
    ocr_engine=TesseractEngine(config),
    text_processor=CustomTextProcessor(config),  # Custom!
    # ... autres composants
)

epub_path = pipeline.run('livre.pdf')
```

### Étape 4 : Tester avec Vos PDFs

```bash
# 1. Conversion avec ancien code
python pdf2epub.py -i test.pdf -a "Author" -t "Title" -l fra
# → Génère test.epub et test_tesseract.txt

# 2. Conversion avec nouveau code
pdf2epub convert test.pdf --author "Author" --title "Title" --language fra
# → Génère test.epub et test_ocr.txt

# 3. Comparer résultats
diff test_tesseract.txt test_ocr.txt
# → Doit être identique (ou différences minimes)

# 4. Valider EPUBs
java -jar epubcheck.jar test.epub
```

### Étape 5 : Migration Complète

Une fois validation OK :

```bash
# 1. Sauvegarder ancien code
cp pdf2epub.py pdf2epub_old.py

# 2. Mettre à jour scripts appelants
# Remplacer appels subprocess par API Python

# 3. Commit changements
git add .
git commit -m "Migrate to refactored architecture"

# 4. (Optionnel) Garder ancien script pour transition
# Le nouveau code coexiste avec l'ancien
```

---

## Correspondance Arguments CLI

| Ancien Argument | Nouveau Argument | Note |
|----------------|------------------|------|
| `-i / --input` | `pdf_path` (positional) | Premier argument |
| `-a / --author` | `--author` | Identique |
| `-t / --title` | `--title` | Identique |
| `-l / --lang` | `--language` | Renommé |
| `-x / --x-margin` | `--x-margin` | Identique |
| `-y / --y-margin` | `--y-margin` | Identique |
| `--tesseract-dir` | `--tesseract-path` | Renommé |
| `--no-chap-detection` | `--no-chapter-detection` | Renommé |
| `--detect-only-on-chap-header` | `--detect-only-with-keyword` | Renommé |
| `-f / --filter` | `--custom-filter` (multiple) | Renommé |
| `--recognize-only` | Commande `ocr` séparée | Refactorisé |
| `--generate-epub-only` | Commande `epub` séparée | Refactorisé |
| N/A | `--config` | **Nouveau** : Config YAML |
| N/A | `--ai-provider` | **Nouveau** : Correction IA |
| N/A | `--ocr-engine` | **Nouveau** : Choix moteur OCR |

---

## Nouvelles Fonctionnalités

### 1. Support Multi-Moteurs OCR

**EasyOCR** (alternative à Tesseract) :
```yaml
# config.yaml
ocr:
  engine: easyocr  # Au lieu de tesseract
  language: fr
  use_gpu: true    # Accélération GPU
```

```bash
pdf2epub convert livre.pdf --ocr-engine easyocr
```

### 2. Correction IA

**Gemini** :
```bash
pdf2epub convert livre.pdf \
    --ai-provider gemini \
    --ai-api-key YOUR_GEMINI_KEY
```

**OpenAI** :
```bash
pdf2epub convert livre.pdf \
    --ai-provider openai \
    --ai-api-key YOUR_OPENAI_KEY
```

**Claude** :
```bash
pdf2epub convert livre.pdf \
    --ai-provider claude \
    --ai-api-key YOUR_ANTHROPIC_KEY
```

### 3. Profils de Configuration

```bash
# Profil rapide (OCR rapide, qualité moyenne)
pdf2epub convert livre.pdf --config configs/fast.yaml

# Profil haute qualité (OCR lent, qualité max)
pdf2epub convert livre.pdf --config configs/high_quality.yaml

# Profil debug (logs verbeux, images intermédiaires)
pdf2epub convert livre.pdf --config configs/debug.yaml
```

### 4. API Programmatique

**Import Python** :
```python
from pdf2epub import (
    ConversionPipeline,
    Config,
    PDFProcessor,
    TesseractEngine,
    ChapterDetector,
    TextProcessor,
    EPUBGenerator
)

# Utilisation directe en Python
config = Config()
pipeline = ConversionPipeline.from_config(config)
epub_path = pipeline.run('livre.pdf')
```

---

## Dépannage

### Problème : "Module 'pdf2epub' not found"

**Solution** :
```bash
# Installer package en mode développement
cd /chemin/vers/pdf2epub
pip install -e .

# Vérifier
python -c "import pdf2epub; print(pdf2epub.__version__)"
```

### Problème : "Tesseract not found"

**Solution** :
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-fra

# Spécifier chemin dans config
# config.yaml
ocr:
  tesseract_path: /usr/bin/tesseract
```

### Problème : Résultats différents entre ancien/nouveau code

**Causes possibles** :
1. **Thresholds différents** : Vérifier paramètres `-x`, `-y`, thresholding
2. **Version Tesseract** : Assurer même version OCR
3. **Filtres custom** : Vérifier patterns regex identiques

**Debug** :
```bash
# Comparer fichiers intermédiaires
diff old_tesseract.txt new_ocr.txt

# Mode debug
pdf2epub convert livre.pdf --debug --config debug.yaml
```

### Problème : Performance dégradée

**Optimisations** :
```yaml
# config.yaml
pdf_processing:
  adaptive_threshold: false  # Désactiver tests multiples seuils
  use_dewarp: false          # Désactiver page-dewarp

ocr:
  engine: tesseract          # Tesseract plus rapide qu'EasyOCR
```

---

## Checklist Migration Complète

- [ ] Tests unitaires passent : `pytest tests/`
- [ ] Tests régression OK : `pytest tests/test_regression.py`
- [ ] PDFs de référence identiques : `diff old_output.txt new_output.txt`
- [ ] EPUBs valides : `epubcheck output.epub`
- [ ] Scripts batch mis à jour
- [ ] CI/CD adapté (si applicable)
- [ ] Documentation utilisateur mise à jour
- [ ] Ancien code archivé : `git mv pdf2epub.py legacy/`

---

## Support et Ressources

- **Documentation complète** : `docs/ARCHITECTURE.md`, `docs/MODULES.md`
- **Guide démarrage rapide** : `QUICKSTART.md`
- **Exemples configurations** : `configs/*.yaml`
- **Tests** : `tests/test_*.py`
- **GitHub** : [Issues](https://github.com/joguy56/pdf2epub/issues) pour bugs/questions

