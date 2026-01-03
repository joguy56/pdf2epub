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
cd pdf2epub

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the tool
poetry run pdf2epub --help
```

#### Option B: With pip

```bash
cd pdf2epub

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .

# Run
pdf2epub --help
```

## Basic Usage

### Simple Conversion (OCR only)

```bash
# Basic conversion with Tesseract OCR
pdf2epub -i mybook.pdf -a "Author Name" -t "Book Title" -l fra

# The EPUB will be created in the same directory as the PDF
# Output: mybook.epub
```

**Parameters:**
- `-i`: Input PDF file
- `-a`: Author name
- `-t`: Book title
- `-l`: Language code (fra, eng, deu, etc.)

### Full Conversion with AI

```bash
# With AI proofreading (Gemini)
pdf2epub -i mybook.pdf -a "Author" -t "Title" -l fra --ai-proofread

# The tool will:
# 1. Extract text with OCR
# 2. Detect chapters automatically
# 3. Validate chapter numbering
# 4. Correct errors with AI
# 5. Generate EPUB
```

### Advanced Options

```bash
# Full control
pdf2epub \
  -i input.pdf \
  -a "Jules Verne" \
  -t "Twenty Thousand Leagues Under the Sea" \
  -l fra \
  --ai-proofread \
  --ocr tesseract \
  --max-workers 4 \
  --verbose

# Clean temporary files afterward
pdf2epub -i book.pdf -a "Author" -t "Title" -l fra --clean
```

## Configuration

### AI Proofreading Setup

To use AI correction, store your key in a file:

```bash
# Create key file (once)
echo "YOUR_GEMINI_API_KEY" > ~/gemini.key
chmod 600 ~/gemini.key

# Use in commands
pdf2epub -i book.pdf -a "Author" -t "Title" --ai-proofread
```

### Custom Configuration

Create `pdf2epub.yaml` in your working directory:

```yaml
pdf:
  dpi: 300
  
ocr:
  engine: tesseract
  language: fra
  confidence_threshold: 80

ai_proofreading:
  enabled: true
  provider: gemini
  model: gemini-2.0-flash-exp
  free_tier: true
  delay_between_chunks: 5
  
chapter_detection:
  use_ai: true
  validate_numbering: true
```

**Key options:**
- `--max-workers 4`: 4 parallel threads (adjust for CPU)
- `--verbose`: Show detailed logs
- `--ai-proofread`: Enable AI correction (requires API key)
- `--ocr tesseract`: OCR engine (tesseract or easyocr)
- `--clean`: Delete temporary files

## Troubleshooting

### Tesseract not found
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### Low OCR quality
Reduce confidence threshold in `pdf2epub.yaml`:
```yaml
ocr:
  confidence_threshold: 60  # Lower from 80
```

### Quota errors (Gemini free tier)
See [GEMINI_FREE_TIER.md](GEMINI_FREE_TIER.md) for details:
```yaml
ai_proofreading:
  delay_between_chunks: 10  # Increase delay
```

### Chapter validation errors
```bash
# Check intermediate files:
ls -lh *_tesseract.txt

# Manually validate:
python3 validate_chapters.py book_tesseract.txt
```

### Skip AI correction
```bash
# Use OCR only (skips AI correction)
# (uses _tesseract.txt instead of _tesseract_ai.txt)
pdf2epub -i book.pdf -a "Author" -t "Title" -l fra
```

## Performance Tips

### Free Tier (Gemini)
- Processing time: ~1min 30s for typical book (17 chunks)
- Daily limit: ~88 books/day (1500 requests/day ÷ 17)
- See [GEMINI_FREE_TIER.md](GEMINI_FREE_TIER.md)

### Paid Plan (Gemini)
- Much faster (no 5s delays)
- Cost: ~$0.01-$0.05 per book
- 1000+ requests/minute

### Optimization
```yaml
ai_proofreading:
  chunk_size: 30000      # Fewer chunks (13 instead of 17)
  delay_between_chunks: 1  # Faster (if using paid plan)
```

## Next Steps

- **Full documentation**: See [docs/README.md](README.md)
- **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Free tier guide**: See [GEMINI_FREE_TIER.md](GEMINI_FREE_TIER.md)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Example Workflow

```bash
# 1. Convert PDF to EPUB
pdf2epub -i mybook.pdf -a "John Doe" -t "My Book" -l eng --ai-proofread

# 2. Check output
ls -lh mybook.epub

# 3. Validate EPUB (optional)
epubcheck mybook.epub

# 4. Read with your favorite reader
# - Calibre
# - Apple Books
# - Google Play Books
# - etc.
```

