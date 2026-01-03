# Changelog

## [2.0.0] - 2026-01-03

### ✨ New Features

#### 🤖 AI Proofreading with Automatic Detection
- **Multi-provider support**: Gemini, OpenAI, Claude
- **Automatic limit detection**: The system detects your API's output token limit and automatically adjusts chunk sizes
- **Intelligent fallback**: If detection fails, uses conservative safe values (8000 tokens for Gemini)
- **Free tier optimized**: Automatic configuration to respect Gemini free tier limits (15 RPM, 1500/day)
- **Quota management**: Detection and automatic checkpoint saving when quota reached

#### 📖 Chapter Validation
- **OCR error detection**: Automatically identifies misrecognized chapter numbers
- **Integrated validation**: Runs validation before AI processing to avoid wasting API calls
- **Correction suggestions**: Shows errors with line numbers and suggestions

#### ⚡ Performance
- **Parallel processing**: Multiprocessing for faster OCR
- **Automatic checkpoints**: Resume on error without losing progress
- **Optimized memory**: Support for large books (500+ pages)

### 🔧 Improvements

#### Configuration
- **YAML files**: Hierarchical configuration with default values
- **Environment variables**: Support for API keys and sensitive parameters
- **Processing modes**: `--pdf-only`, `--ocr-only`, `--generate-epub-only`, `--ai-proofread`
- **Wizard mode**: Interactive assistant for beginners

#### Code
- **Modern architecture**: Clear separation of responsibilities (PDF → OCR → Text → AI → EPUB)
- **Type hints**: Python 3.10+ with complete annotations
- **Error handling**: Typed exceptions and clear messages
- **Logging**: Colored logs with detail levels

### 🐛 Bug Fixes

- **AI truncation**: Resolved chapter loss problem (automatic limit detection)
- **Page order**: Fixed parallel processing that could mix pages
- **Chapter numbering**: OCR validation to avoid recognition errors
- **Memory management**: Optimization to avoid OOM on large files
- **Encoding**: Full UTF-8 support for special characters and accents

### 📚 Documentation

- **Complete README**: User guide with examples
- **QUICKSTART**: Quick start in 5 minutes
- **GEMINI_FREE_TIER**: Detailed free tier guide
- **AI_AUTO_DETECTION**: Technical explanation of automatic detection
- **CHAPTER_VALIDATION**: Chapter validation guide

### 🧪 Tests

- **Unit tests**: Coverage of critical functions
- **Integration tests**: Full pipeline validation
- **Regression tests**: Automatic results verification

### ⚙️ Gemini Free Tier Configuration

By default, the system is optimized for the free tier:
- `free_tier: true` - Activates adapted delays
- `delay_between_chunks: 5` - 5 seconds between chunks (12 req/min < 15 RPM)
- `chunk_size: 22000` - Safe size for 8k token limit

**Processing time**: ~1min 30s for a 380k character book (17 chunks)  
**Daily capacity**: ~88 books/day maximum

---

## [1.0.0] - Original Prototype

### Initial Features
- PDF → EPUB conversion with Tesseract OCR
- Basic chapter detection
- Text post-processing (hyphenation, dialogs)
- Command-line interface

### Limitations
- Monolithic code difficult to maintain
- No robust error handling
- Hardcoded configuration
- No automated tests
