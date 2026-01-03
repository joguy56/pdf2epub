# Troubleshooting Guide

## 🔍 Table of Contents

1. [Installation Issues](#installation-issues)
2. [OCR Errors](#ocr-errors)
3. [Chapter Validation](#chapter-validation)
4. [AI Proofreading and Quota](#ai-proofreading-and-quota)
5. [Performance and Memory](#performance-and-memory)

---

## Installation Issues

### Tesseract not found
```
Error: tesseract not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Verify installation
tesseract --version
```

### TESSDATA_PREFIX variable
```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

### page-dewarp missing

The tool works without it, but image quality may be lower.  
Installation: https://github.com/mzucker/page_dewarp

---

## OCR Errors

### Low OCR confidence

If too much text is rejected:
```yaml
ocr:
  confidence_threshold: 60  # Reduce from 80 to 60
```

### Incorrect language

```bash
# Check available languages
tesseract --list-langs

# Install a language
sudo apt-get install tesseract-ocr-eng  # English
sudo apt-get install tesseract-ocr-deu  # German
```

---

## Chapter Validation

### ⚠️ OCR Numbering Errors

OCR (Tesseract) can misread chapter numbers:
- `17. TITLE` → `47. TITLE` (1 not recognized)
- `20. OTHER` → `0. OTHER` (2 missing)

### Automatic Validation

Validation runs automatically in the pipeline:
```
 Runs after OCR, before AI proofreading
 Stops if errors found
```

**Example of detected error:**
```
 ERROR: Numbering gap detected
   Chapter 16 → Chapter 47 (line 1234)
   Expected chapter: 17
   💡 Suggested fix: Replace "47" with "17"
```

### Manual Validation

```bash
python3 validate_chapters.py ../book_tesseract.txt
```

### Fixing Errors

1. **Open the file**: `vim book_tesseract.txt`
2. **Go to line**: `:1234` (number shown in error)
3. **Correct the number**: `47.` → `17.`
4. **Save**: `:wq`
5. **Re-run validation**: `python3 validate_chapters.py ../book_tesseract.txt`

### Detailed Example

**Original file (with errors):**
```
15. The Discovery
[text...]

47. The Trap        ← ERROR: Should be 17
[text...]

18. The Escape      ← ERROR: Should be 18 but follows 47
[text...]
```

**After correction:**
```
15. The Discovery
[text...]

17. The Trap        ✅ FIXED
[text...]

18. The Escape      ✅ OK
[text...]
```

### Ignoring Known Issues

If you are certain the numbering is correct (non-sequential chapters):
```yaml
chapter_detection:
  validate_numbering: false  # Skip validation
```

---

## AI Proofreading and Quota

### 🚨 Quota Limit Reached

```
 QUOTA LIMIT REACHED!
   Gemini free tier: 15 requests/minute, 1500 requests/day
   Chunks processed: 12/17
```

**Solutions:**
1. **Wait 24 hours** for quota to reset
2. **Upgrade to paid plan** (pay-as-you-go)
3. **Use checkpoints** (see below)

### Checkpoint Files

When quota is reached, a checkpoint is automatically saved:
```
gemini_checkpoint_12_of_17.txt
```

This contains the text processed so far.

**To resume** (manual for now):
1. Wait 24h for quota to reset
2. Restart full processing (code will retry failed chunks)

### Rate Limiting (429 Too Many Requests)

```
google.api_core.exceptions.ResourceExhausted: 429
```

**Solution:**
```yaml
ai_proofreading:
  delay_between_chunks: 10  # Increase from 5 to 10 seconds
```

### Reducing Chunk Count

Process fewer chunks by increasing size:
```yaml
ai_proofreading:
  chunk_size: 30000  # Instead of 22000 (reduces from 17 to ~13 chunks)
```

 **Risk**: Output truncation if limit < 8000 tokens.

### Invalid API Key

```
google.auth.exceptions.DefaultCredentialsError
```

**Solution:**
```bash
# Check your key
cat ~/gemini.key

# Test manually
python3 -c "
import google.generativeai as genai
genai.configure(api_key=open('~/gemini.key').read().strip())
model = genai.GenerativeModel('gemini-2.0-flash-exp')
print(model.generate_content('Hello').text)
"
```

---

## Performance and Memory

### Processing is very slow

**Possible causes:**
1. **Free tier delays**: 5s between chunks is normal  
   → See [GEMINI_FREE_TIER.md](GEMINI_FREE_TIER.md)

2. **Large PDF**: Many pages = more OCR time  
   → Normal behavior

3. **Low CPU**: OCR is CPU-intensive  
   → Use better hardware or reduce DPI

### Memory errors

```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Reduce DPI**:
   ```yaml
   pdf:
     dpi: 200  # Instead of 300
   ```

2. **Process in batches**: Split PDF into smaller files

3. **Increase system memory**: Close other applications

### Disk space issues

```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check disk space
df -h

# Clean temporary files
rm -rf /tmp/pdf2epub_*
```

---

## Common Workflow Issues

### EPUB file is empty or corrupted

**Checks:**
1. Verify OCR output exists: `ls -lh *_tesseract.txt`
2. Check file size > 0
3. Validate text encoding (UTF-8)
4. Re-run with `--verbose` flag

### Chapter titles not detected

```yaml
chapter_detection:
  use_ai: true              # Enable AI detection
  validate_numbering: true  # Enable validation
```

**Check pattern:**
```python
# In chapter_detector.py
pattern = r'^\s*(\d+)\.\s+(.+)$'  # Matches "17. Title"
```

### AI proofreading skips chapters

**Check logs:**
```
  Chunk 5 failed, retrying...
 Chunk 5 failed after 3 retries
```

**Solutions:**
1. Increase retries:
   ```yaml
   ai_proofreading:
     max_retries: 5  # Instead of 3
   ```

2. Increase timeout:
   ```yaml
   ai_proofreading:
     timeout: 120  # Instead of 60 seconds
   ```

---

## Debug Mode

Enable verbose logging:
```bash
./pdf2epub.sh -i book.pdf -a "Author" -t "Title" --verbose
```

Or in Python:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Getting Help

If problems persist:
1. Check [GitHub Issues](https://github.com/joguy56/pdf2epub/issues)
2. Create new issue with:
   - Command used
   - Full error message
   - OS and Python version
   - `pdf2epub.yaml` configuration

---

## Quick Fixes Summary

| Problem | Quick Fix |
|---------|-----------|
| Tesseract not found | `sudo apt-get install tesseract-ocr` |
| Low OCR quality | Lower `confidence_threshold` to 60 |
| Quota reached | Wait 24h or increase `delay_between_chunks` |
| Slow processing | Normal with free tier (5s delays) |
| Chapter errors | Run `validate_chapters.py` and fix manually |
| Memory errors | Reduce DPI to 200 |
| Invalid API key | Check `~/gemini.key` file exists |
