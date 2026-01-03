# 🚀 Quick Start Guide

## Installation

### Prerequisites

1. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
   
   # macOS
   brew install tesseract tesseract-lang
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **Poppler** (for PDF processing)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install poppler-utils
   
   # macOS
   brew install poppler
   ```

### Install pdf2epub

#### Option A: With Poetry (Recommended)

```bash
cd pdf2epub-refactored

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the tool
poetry run pdf2epub --help
```

#### Option B: With pip

```bash
cd pdf2epub-refactored

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
pip install -e .

# Run the tool
pdf2epub --help
```

## Première Conversion

### Mode Recommandé: Utilisation Directe

La façon la plus simple est d'utiliser PYTHONPATH:

```bash
cd /home/jguyot/pdf2epub/main/pdf2epub-refactored

# Mode interactif (pose des questions)
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf -a "Auteur" -t "Titre"

# Mode batch (sans questions)
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf -a "Auteur" -t "Titre" --batch

# L'EPUB sera créé dans le même répertoire que le PDF
```

### Avec Clé API Gemini

```bash
# Pour la correction IA, ajoutez votre clé API
PYTHONPATH=src GEMINI_API_KEY=$(cat ~/gemini.key) python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  -a "Auteur" \
  -t "Titre" \
  --ai-proofread \
  --batch
```

## Cas d'Usage Courants

### Conversion Complète avec IA

```bash
cd /home/jguyot/pdf2epub/main/pdf2epub-refactored

PYTHONPATH=src GEMINI_API_KEY=$(cat ~/gemini.key) python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  -a "Auteur" \
  -t "Titre" \
  --ai-proofread \
  --batch \
  --max-workers 4
```

### Régénérer l'EPUB depuis le texte OCR existant

```bash
# Si vous avez déjà livre_tesseract.txt ou livre_tesseract_ai.txt
PYTHONPATH=src python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  --generate-epub-only
```

### Nettoyer les fichiers temporaires

```bash
# Supprime *_tesseract.txt, *_tesseract_ai.txt et tmp/
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf --clean
```

### Mode Debug

```bash
# Logs détaillés
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf -d
```

## Configuration

### Clé API Gemini

Pour utiliser la correction IA, stockez votre clé dans un fichier:

```bash
# Créer le fichier de clé (une seule fois)
echo "votre-cle-api-gemini" > ~/gemini.key
chmod 600 ~/gemini.key

# Utiliser dans les commandes
GEMINI_API_KEY=$(cat ~/gemini.key)
```

### Options Principales

**Couverture:**
- `--cover-mode 1`: Image uniquement (pas d'OCR)
- `--cover-mode 2`: OCR + Image (défaut, recommandé)
- `--cover-mode 3`: Page normale (pas de cover)

**Langue:**
- `-l fra`: Français
- `-l eng`: Anglais

**Performance:**
- `--max-workers 4`: 4 threads parallèles (ajustez selon CPU)
- `--batch`: Mode automatique sans questions

**IA:**
- `--ai-proofread`: Active la correction Gemini

**Nettoyage:**
- `--clean`: Supprime les fichiers temporaires

## Dépannage

### "Tesseract not found"

```bash
# Vérifier l'installation
tesseract --version

# Si non trouvé, installer
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### Manque de mémoire

```bash
# Réduire le nombre de workers
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf --max-workers 2
```

### Chapitres mal détectés

```bash
# Désactiver la détection auto
PYTHONPATH=src python3 src/pdf2epub/cli.py -i ../livre.pdf --no-chap-detection
```

### L'IA corrige trop ou pas assez

```bash
# Vérifier les fichiers intermédiaires:
# - livre_tesseract.txt (après post-processing)
# - livre_tesseract_ai.txt (après IA)

# Compter les mots de chaque étape
wc -w livre_tesseract*.txt

# Régénérer l'EPUB sans IA
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --generate-epub-only
# (utilise le fichier _tesseract.txt au lieu de _tesseract_ai.txt)
```

## Next Steps

- Read the full [README.md](../README.md) for detailed documentation
- Check the [Architecture section](README.md#-architecture) to understand the codebase
- Run the test suite: `poetry run pytest`
- Contribute: See [Contributing section](README.md#-contributing)

## Getting Help

- Check existing [GitHub Issues](https://github.com/jguyot/pdf2epub/issues)
- Read the troubleshooting section in README.md
- Enable debug mode: `pdf2epub -i book.pdf -d`
