# 📚 pdf2epub - Convert Scanned PDFs to EPUB Ebooks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, modern Python tool for converting scanned PDF books into high-quality EPUB ebooks using OCR technology. Complete rewrite of the original prototype with production-ready code, comprehensive error handling, and advanced features.

## ✨ Features

### Core Functionality
- **📄 PDF to EPUB conversion** with OCR text recognition
- **🔍 Dual OCR engines**: Tesseract (default) or EasyOCR
- **📖 Smart chapter detection** based on page layout analysis
- **✏️ Advanced text post-processing** (hyphenation, dialogs, special characters)
- **🎨 Automatic cover image extraction**
- **🌍 Multi-language support** (French, English, and more)

### Advanced Features
- **🤖 AI-powered proofreading** (Gemini, OpenAI, Claude)
- **⚡ Parallel processing** for faster conversions
- **💾 Resume on error** with automatic checkpoints
- **📊 Quality metrics** and confidence scoring
- **🔧 Highly configurable** via YAML configuration files

### Developer-Friendly
- **🏗️ Modern architecture** with clean separation of concerns
- **✅ Comprehensive error handling** with detailed logging
- **🧪 Unit and integration tests**
- **📝 Type hints** throughout (Python 3.10+)
- **🎨 Code quality tools** (black, ruff, mypy)

## 🚀 Quick Start

### Installation

```bash
cd pdf2epub-refactored

# Installer les dépendances Python
pip install -r requirements.txt

# Vérifier que Tesseract est installé
tesseract --version
# Si absent: sudo apt-get install tesseract-ocr tesseract-ocr-fra

# Rendre le script helper exécutable
chmod +x pdf2epub.sh
```

### Utilisation avec le Script Helper (Recommandé)

Le script `pdf2epub.sh` simplifie l'utilisation en gérant automatiquement PYTHONPATH et la clé Gemini:

**Mode Wizard - Interface Interactive Complète:**

```bash
# Lance l'interface interactive qui demande TOUTES les informations
./pdf2epub.sh --wizard

# L'outil demande:
# - Fichier PDF à convertir
# - Titre du livre
# - Auteur
# - Langue
# - Gestion de la couverture
# - Détection chapitres
# - Correction IA
# - Traitement parallèle
```

**Mode Direct (si vous connaissez les paramètres):**

```bash
# Créer votre clé Gemini (une seule fois)
echo "votre-clé-api-gemini" > ~/gemini.key
chmod 600 ~/gemini.key

# Utiliser le script
./pdf2epub.sh -i ../livre.pdf -a "Auteur" -t "Titre"

# Avec IA
./pdf2epub.sh -i ../livre.pdf -a "Auteur" --ai-proofread --batch

# Toutes les options sont supportées
./pdf2epub.sh --help
```

### Utilisation Basique

**Mode Interactif (Recommandé pour débuter):**

```bash
cd /home/jguyot/pdf2epub/main/pdf2epub-refactored

# L'outil pose des questions essentielles
PYTHONPATH=src python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  -a "Nom Auteur" \
  -t "Titre du Livre"
```

**Mode Batch (Automatique, sans questions):**

```bash
# Pour automatiser ou avec valeurs par défaut
PYTHONPATH=src python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  -a "Nom Auteur" \
  -t "Titre du Livre" \
  --batch
```

**Avec Correction IA (Gemini):**

```bash
# Stocker la clé API (une fois)
echo "votre-clé-gemini" > ~/gemini.key
chmod 600 ~/gemini.key

# Utiliser avec l'IA
PYTHONPATH=src GEMINI_API_KEY=$(cat ~/gemini.key) python3 src/pdf2epub/cli.py \
  -i ../livre.pdf \
  -a "Auteur" \
  -t "Titre" \
  --ai-proofread \
  --batch
```

**Exemples Courants:**

```bash
# Régénérer l'EPUB depuis le texte OCR existant
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --generate-epub-only

# Nettoyer les files temporaires
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --clean

# Mode debug avec logs détaillés
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf -d
```

## 📖 Documentation

### Fichiers Générés

Lors de la conversion, l'outil crée plusieurs files:

```
livre.pdf                    # Votre PDF source
livre.epub                   # EPUB généré ✅
livre_tesseract.txt          # Texte après post-processing (Stage 3)
livre_tesseract_ai.txt       # Texte après correction IA (Stage 4, si --ai-proofread)
tmp/                         # Images temporaires (nettoyé automatiquement)
  cover.jpg                  # Image de couverture extraite
  page_*.jpg                 # Pages converties en images
```

**Ordre de priorité pour la génération EPUB:**
1. Si `livre_tesseract_ai.txt` existe → utilise ce fichier
2. Sinon, si `livre_tesseract.txt` existe → utilise ce fichier
3. Sinon, lance le pipeline complet

**Commandes de nettoyage:**
```bash
# Supprimer tous les files temporaires
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --clean

# Ou manuellement
rm livre_tesseract*.txt
rm -rf tmp/
```

### Configuration

Create a default configuration file:

```bash
pdf2epub --create-config
```

This creates `pdf2epub.yaml` with all available options. Edit it to customize:

```yaml
# OCR Configuration
ocr:
  engine: tesseract  # or easyocr
  language: fra      # fra, eng, etc.
  confidence_threshold: 80

# Chapter Detection
chapter_detection:
  enabled: true
  threshold_percentage: 25  # % from top of page

# AI Proofreading (optional)
ai_proofreading:
  enabled: false
  provider: gemini  # gemini, openai, or claude
  model: gemini-2.5-flash
  free_tier: true   # Gemini free tier (15 RPM) - see GEMINI_FREE_TIER.md
  delay_between_chunks: 5  # Delay between chunks (5s for free tier)
  chunk_size: 22000  # Auto-adjusted based on detected limits
  api_key: null  # or set via environment variable

# Performance
performance:
  parallel_processing: true
  enable_resume: true
  checkpoint_interval: 10
```

Configuration files are loaded from (in order):
1. `--config` argument
2. `./pdf2epub.yaml`
3. `~/.pdf2epub.yaml`
4. `/etc/pdf2epub/config.yaml`

### Options Principales

```
Requis:
  -i, --input PATH          Fichier PDF source

Métadonnées:
  -a, --author TEXT         Auteur du livre
  -t, --title TEXT          Titre (défaut: nom du fichier)
  -l, --language CODE       Code langue (fra, eng, etc.)

Couverture:
  --cover-mode {1,2,3}      1=Image seule, 2=OCR+Image (défaut), 3=Page normale
  --no-cover                Pas d'image de couverture

Stades de traitement:
  -r, --recognize-only      Partir des images (skip PDF→images)
  -g, --generate-epub-only  Partir du texte OCR existant (skip tout sauf EPUB)

Détection chapitres:
  --no-chap-detection       Désactiver détection automatique

Correction IA:
  --ai-proofread            Activer correction Gemini

Performance:
  --max-workers N           Nombre de threads parallèles (défaut: auto)
  --batch                   Mode automatique (pas de questions)

Maintenance:
  --clean                   Supprimer files temporaires
  -d, --debug               Logs détaillés
```

**Exemples:**

```bash
# Conversion basique
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf -a "Auteur"

# Avec IA et 4 workers
PYTHONPATH=src GEMINI_API_KEY=$(cat ~/gemini.key) python3 src/pdf2epub/cli.py \
  -i livre.pdf -a "Auteur" --ai-proofread --max-workers 4 --batch

# Régénérer EPUB sans refaire OCR
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --generate-epub-only

# Nettoyer
PYTHONPATH=src python3 src/pdf2epub/cli.py -i livre.pdf --clean
```

### Cover Page Handling

The first page of your PDF can be handled in three ways:

| Mode | Flag | Description | Use Case |
|------|------|-------------|----------|
| **1** | `--cover-mode 1` | Image only (no OCR) | Pure cover image, no text to extract |
| **2** | `--cover-mode 2` | OCR + Image (default) | Cover with title/author to extract |
| **3** | `--cover-mode 3` | Normal page | First page is regular content |

**Interactive prompt (if no flag):**
```
📖 GESTION DE LA PAGE DE COUVERTURE
Comment voulez-vous traiter la première page du PDF ?
  1. Image de couverture uniquement (pas d'OCR)
  2. OCR + image de couverture (extrait le texte ET utilise comme cover)
  3. Traiter comme page normale (OCR mais pas de cover dans EPUB)
Votre choix [1/2/3] (défaut: 2):
```

**Examples:**
```bash
# Cover with text extraction (recommended)
pdf2epub -i book.pdf --cover-mode 2

# Cover as image only (skip OCR)
pdf2epub -i book.pdf --cover-mode 1

# No cover image at all
pdf2epub -i book.pdf --cover-mode 3
# or
pdf2epub -i book.pdf --no-cover
```

### AI Proofreading Setup

pdf2epub supports three AI providers for automatic text correction:

#### Google Gemini (Recommended - Lowest Cost)

```bash
# Install SDK
pip install google-generativeai

# Set API key
export GEMINI_API_KEY="your-key"

# Or in config file
ai_proofreading:
  enabled: true
  provider: gemini
  api_key: your-key
```

**Cost**: ~$0.20-0.35 per book (300 pages)

#### OpenAI

```bash
pip install openai
export OPENAI_API_KEY="your-key"

# In command
pdf2epub -i book.pdf --ai-proofread --ai-provider openai
```

**Cost**: ~$0.45-0.70 per book

#### Anthropic Claude

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"

pdf2epub -i book.pdf --ai-proofread --ai-provider claude
```

**Cost**: ~$0.75-1.25 per book

## 🏗️ Architecture

### Project Structure

```
pdf2epub-refactored/
├── src/pdf2epub/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── pipeline.py         # Main conversion pipeline
│   ├── pdf_processor.py    # PDF → images conversion
│   ├── ocr/
│   │   ├── base.py         # OCR interface
│   │   ├── tesseract.py    # Tesseract implementation
│   │   └── easyocr.py      # EasyOCR implementation
│   ├── chapter_detector.py # Chapter detection logic
│   ├── text_processor.py   # Text post-processing
│   ├── ai_proofreader.py   # AI proofreading
│   └── epub_generator.py   # EPUB generation
├── tests/
│   ├── test_text_processor.py
│   └── fixtures/
├── pyproject.toml          # Poetry configuration
└── README.md
```

### Processing Pipeline

```
┌─────────────┐
│  PDF Input  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  PDF → Images       │  (pdf_processor.py)
│  - Convert pages    │
│  - Preprocessing    │
│  - Page dewarping   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  OCR Processing     │  (ocr/*.py)
│  - Text extraction  │
│  - Block analysis   │
│  - Confidence check │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Chapter Detection  │  (chapter_detector.py)
│  - Layout analysis  │
│  - Junk filtering   │
│  - Structure markup │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Text Processing    │  (text_processor.py)
│  - Hyphenation fix  │
│  - Dialog format    │
│  - Special chars    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  AI Proofreading    │  (ai_proofreader.py)
│  (optional)         │
│  - Spelling fixes   │
│  - Grammar check    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  EPUB Generation    │  (epub_generator.py)
│  - Chapter assembly │
│  - Metadata         │
│  - CSS styling      │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ EPUB Output │
└─────────────┘
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=pdf2epub --cov-report=html

# Run specific test
poetry run pytest tests/test_text_processor.py -v

# Run with debug output
poetry run pytest -s -v
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install with dev dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run code formatters
poetry run black src/
poetry run ruff check src/ --fix

# Type checking
poetry run mypy src/
```

### Code Quality Standards

- **Formatting**: `black` (line length: 100)
- **Linting**: `ruff` (see pyproject.toml for rules)
- **Type hints**: Required for all functions
- **Docstrings**: Google style
- **Tests**: Aim for >80% coverage

## 📊 Performance

Typical performance on a modern laptop (8-core CPU):

| Document | Pages | Sequential | Parallel | Speedup |
|----------|-------|-----------|----------|---------|
| Small    | 50    | 2.5 min   | 1.5 min  | 1.7x    |
| Medium   | 150   | 8 min     | 4 min    | 2.0x    |
| Large    | 300   | 18 min    | 9 min    | 2.0x    |

*Times include full pipeline (PDF → images → OCR → processing → EPUB)*

## 🐛 Troubleshooting

**📖 Guide complet : [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

### Problèmes Courants

**Installation Tesseract**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fra  # Ubuntu/Debian
brew install tesseract tesseract-lang                  # macOS
```

**Validation des Chapitres**

OCR can misread numbers: `17. Titre` → `47. Titre`

Automatic validation detects these errors. See [CHAPTER_VALIDATION.md](docs/CHAPTER_VALIDATION.md).

**Plan Gratuit Gemini**

- 15 req/min, 1500 req/day
- Délai automatique : 5s entre chunks
- Temps : ~1min 30s pour 380k chars

If quota reached → automatic checkpoint. See [GEMINI_FREE_TIER.md](docs/GEMINI_FREE_TIER.md).

**Détection Automatique des Limites IA**

The system detects your token limit and automatically adjusts chunks.

See [AI_AUTO_DETECTION.md](docs/AI_AUTO_DETECTION.md) for technical details.

---
## � Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Démarrage rapide (5 minutes)
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Guide de dépannage complet
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Historique des versions
- **[GEMINI_FREE_TIER.md](docs/GEMINI_FREE_TIER.md)** - Gemini free tier configuration
- **[AI_AUTO_DETECTION.md](docs/AI_AUTO_DETECTION.md)** - Détection automatique des limites IA
- **[CHAPTER_VALIDATION.md](docs/CHAPTER_VALIDATION.md)** - Validation des chapitres OCR

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`poetry run pytest && poetry run black . && poetry run ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Alternative OCR engine
- [ebooklib](https://github.com/aerkalov/ebooklib) - EPUB generation
- [page-dewarp](https://github.com/mzucker/page_dewarp) - Image dewarping

## 📧 Support

For bug reports and feature requests, please open an issue on GitHub.

---

**Made with ❤️ for book lovers and digital archivists**
