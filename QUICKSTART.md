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

## First Conversion

```bash
# Test with a sample PDF (use one from the workspace)
poetry run pdf2epub \
  -i ../Retrouvailles.pdf \
  -a "Anne-Marie Desplat-Duc" \
  -t "Retrouvailles" \
  -l fra

# The EPUB will be created in the same directory as the PDF
```

## Common Usage Patterns

### Resume After Error

```bash
# If conversion was interrupted, just run the same command again
# The tool will resume from the last checkpoint
poetry run pdf2epub -i book.pdf -a "Author" -t "Title"
```

### Skip PDF Conversion (Reuse Images)

```bash
# If you already converted the PDF to images
poetry run pdf2epub -i book.pdf --recognize-only
```

### Generate EPUB from Existing OCR Text

```bash
# If you already have book_tesseract.txt
poetry run pdf2epub -i book.pdf --generate-epub-only
```

### Enable AI Proofreading

```bash
# Set API key
export GEMINI_API_KEY="your-api-key"

# Run with proofreading
poetry run pdf2epub -i book.pdf --ai-proofread
```

### Debug Mode

```bash
# See detailed logs and keep temporary files
poetry run pdf2epub -i book.pdf -d --keep-temp
```

## Configuration

### Create Config File

```bash
poetry run pdf2epub --create-config
```

This creates `pdf2epub.yaml` in the current directory. Edit it to customize:

```yaml
ocr:
  engine: tesseract
  language: fra

ai_proofreading:
  enabled: true
  provider: gemini
  
performance:
  parallel_processing: true
  max_workers: 4
```

### Environment Variables

Set these to avoid putting API keys in config files:

```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Troubleshooting

### "Tesseract not found"

```bash
# Check if tesseract is installed
tesseract --version

# If not found, install it (see Prerequisites above)

# If installed but not found, set the path
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
# Or
poetry run pdf2epub -i book.pdf --tesseract-dir /path/to/tessdata
```

### "Out of memory"

```bash
# Reduce parallel workers
poetry run pdf2epub -i large-book.pdf --max-workers 2
```

### Chapters Not Detected

```bash
# Adjust threshold (higher = more strict)
poetry run pdf2epub -i book.pdf --chap-detect-thres-pct 30

# Or use keyword-based detection
poetry run pdf2epub -i book.pdf --detect-only-on-chap-header

# Or disable completely
poetry run pdf2epub -i book.pdf --no-chap-detection
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check the [Architecture section](README.md#-architecture) to understand the codebase
- Run the test suite: `poetry run pytest`
- Contribute: See [Contributing section](README.md#-contributing)

## Getting Help

- Check existing [GitHub Issues](https://github.com/jguyot/pdf2epub/issues)
- Read the troubleshooting section in README.md
- Enable debug mode: `pdf2epub -i book.pdf -d`
