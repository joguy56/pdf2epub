# Chapter Validation Tool

## Purpose

Automatically detects OCR numbering errors in chapter markers **before** EPUB generation and AI proofreading. This prevents wasted API calls and ensures a clean EPUB with proper chapter navigation.

## Common OCR Errors Detected

1. **Missing leading digit**: `17` → `7`, `20` → `0`
2. **Digit misreads**: `1` not recognized in `17` → appears as `47`
3. **Sequence gaps**: Chapter 16 → Chapter 24 (should be 17)
4. **Fake chapters**: `@@@ Chapitre 1 @@@` without proper title
5. **Duplicate numbers**: Same chapter number appearing multiple times
6. **Out-of-order chapters**: Chapter 47 appearing after Chapter 21

## Integration

### Automatic Validation (in Pipeline)

The validation runs automatically after OCR and text processing:

```bash
./pdf2epub.sh -i book.pdf -a "Author" -t "Title" -l fra
```

If errors are found:
- **Errors** trigger a warning and ask for confirmation
- **Warnings** are logged but don't stop the pipeline
- The text file location is shown for manual correction

### Manual Validation (Standalone Tool)

Use the standalone tool to validate files before processing:

```bash
python3 validate_chapters.py path/to/book_tesseract.txt
```

**Options:**
- `--strict`: Treat warnings as errors (fail on warnings)

**Exit codes:**
- `0`: Validation passed (or warnings only in non-strict mode)
- `1`: Validation failed (errors found)

## Example Output

### ✅ Clean File (No Errors)

```
🔍 Validating chapters in: book_tesseract.txt

INFO - 📋 Found 45 chapter markers
INFO - ✅ Chapter validation passed - no issues found

✅ Validation PASSED - no issues
```

### ❌ File with OCR Errors

```
🔍 Validating chapters in: book_tesseract.txt

❌ CHAPTER VALIDATION ERRORS (5 found)
================================================================================

📍 Line 1186: 47. DRAME AUX TOILETTES
   Type: sequence_gap
   Expected: 17, Found: 47
   💡 Gap of 30 chapters: 16 → 47. Likely OCR error: check if 47 should be 17

📍 Line 1357: 0. SHEILA-DES-POULES
   Type: out_of_order
   Expected: 20, Found: 0
   💡 Chapter 0 appears after 19 - likely OCR misread

📍 Line 2611: Chapitre 1
   Type: suspicious_format
   💡 Chapter marker 'Chapitre 1' looks like OCR error - should have title
```

## How to Fix Detected Errors

1. **Open the text file** shown in the error message
2. **Go to the line number** indicated (e.g., Line 1186)
3. **Check the context** around that line in the PDF/book
4. **Correct the chapter number** in the marker:
   ```
   @@@ 47. DRAME AUX TOILETTES @@@  →  @@@ 17. DRAME AUX TOILETTES @@@
   ```
5. **Re-run validation** to confirm fixes
6. **Continue processing** with corrected file

## Typical Workflow

```bash
# 1. Run OCR on PDF
./pdf2epub.sh -i book.pdf -a "Author" -t "Title" -l fra --recognize-only

# 2. Validate chapters (automatic in pipeline, but can be done manually)
python3 validate_chapters.py ../book_tesseract.txt

# 3. If errors found, fix them in book_tesseract.txt
#    (Edit the file manually based on validation output)

# 4. Re-validate
python3 validate_chapters.py ../book_tesseract.txt

# 5. Continue with EPUB generation
./pdf2epub.sh -i book.pdf --generate-epub-only
```

## Why This Matters

### Without Validation
1. OCR creates `book_tesseract.txt` with chapter errors
2. AI proofreading processes 8 chunks (20 minutes, costs API calls)
3. EPUB generated with broken chapter navigation
4. User discovers errors in final EPUB
5. Must delete AI output, fix chapters, re-run AI (another 20 minutes + API cost)

### With Validation
1. OCR creates `book_tesseract.txt` with chapter errors
2. **Validation catches errors immediately** (2 seconds)
3. User fixes chapters in text file (5 minutes)
4. AI proofreading runs once with correct chapters (20 minutes)
5. EPUB generated perfectly the first time

**Time saved**: ~20 minutes  
**API calls saved**: 8 chunks worth of Gemini API requests  
**Quality**: Clean EPUB with proper chapter navigation

## Technical Details

The validator checks:

1. **Sequential numbering**: Chapters should increment by 1 (1, 2, 3, ...)
2. **No duplicates**: Each chapter number appears exactly once
3. **No large gaps**: Gaps > 1 indicate OCR errors
4. **Proper order**: Chapters appear in ascending order
5. **Valid format**: 
   - Numbered: `@@@ N. TITLE @@@` (N = number)
   - Sections: `$$$ TITLE $$$` (centered text)
6. **No fake chapters**: Markers like `@@@ Chapitre 1 @@@` without titles

## Performance Impact

- **Validation time**: ~1-2 seconds for 300-page book
- **Memory**: Minimal (loads text file, ~1-2 MB)
- **When to skip**: Only if book has no chapter markers at all

## Configuration

Validation settings can be adjusted in `chapter_validator.py`:

```python
# Adjust thresholds
similarity_threshold = 0.9  # For duplicate detection
max_gap_size = 5            # Warning threshold for gaps
min_chapter_title_length = 3  # Minimum title length
```

## Batch Validation

Validate multiple files at once:

```bash
for file in *_tesseract.txt; do
    echo "Validating $file..."
    python3 validate_chapters.py "$file"
done
```

## Integration with CI/CD

Use in automated pipelines:

```bash
#!/bin/bash
# convert_and_validate.sh

# Run OCR
./pdf2epub.sh -i "$1" --recognize-only

# Validate chapters (fail on errors)
if ! python3 validate_chapters.py "../$(basename "$1" .pdf)_tesseract.txt"; then
    echo "❌ Chapter validation failed - manual review required"
    exit 1
fi

# Continue with EPUB generation
./pdf2epub.sh -i "$1" --generate-epub-only --ai-proofread
```
