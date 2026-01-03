#!/usr/bin/env python3
"""Verify AI proofreading didn't truncate text."""

import sys
from pathlib import Path


def verify_ai_output(base_name: str) -> bool:
    """
    Verify AI proofreading output is complete.
    
    Args:
        base_name: Base filename (e.g., "RobinHoodT9")
        
    Returns:
        True if validation passed, False otherwise
    """
    ocr_file = Path(f"{base_name}_tesseract.txt")
    ai_file = Path(f"{base_name}_tesseract_ai.txt")
    
    if not ocr_file.exists():
        print(f"❌ ERROR: OCR file not found: {ocr_file}")
        return False
    
    if not ai_file.exists():
        print(f"❌ ERROR: AI file not found: {ai_file}")
        return False
    
    # Read files
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    with open(ai_file, 'r', encoding='utf-8') as f:
        ai_text = f.read()
    
    # Count chapters
    ocr_chapters = ocr_text.count('\n@@@')
    ai_chapters = ai_text.count('\n@@@')
    
    # Count words
    ocr_words = len(ocr_text.split())
    ai_words = len(ai_text.split())
    word_loss_pct = (1 - ai_words / ocr_words) * 100 if ocr_words > 0 else 0
    
    # Count chars
    ocr_chars = len(ocr_text)
    ai_chars = len(ai_text)
    char_loss_pct = (1 - ai_chars / ocr_chars) * 100 if ocr_chars > 0 else 0
    
    # Report
    print("=" * 80)
    print("AI PROOFREADING VERIFICATION")
    print("=" * 80)
    print(f"\n📄 OCR File: {ocr_file.name}")
    print(f"   - Chapters: {ocr_chapters}")
    print(f"   - Words: {ocr_words:,}")
    print(f"   - Characters: {ocr_chars:,}")
    
    print(f"\n🤖 AI File: {ai_file.name}")
    print(f"   - Chapters: {ai_chapters}")
    print(f"   - Words: {ai_words:,}")
    print(f"   - Characters: {ai_chars:,}")
    
    print(f"\n📊 Changes:")
    print(f"   - Chapters: {ai_chapters - ocr_chapters:+d}")
    print(f"   - Words: {ai_words - ocr_words:+,} ({word_loss_pct:+.1f}%)")
    print(f"   - Characters: {ai_chars - ocr_chars:+,} ({char_loss_pct:+.1f}%)")
    
    # Validation
    errors = []
    warnings = []
    
    if ai_chapters < ocr_chapters:
        errors.append(
            f"❌ CRITICAL: {ocr_chapters - ai_chapters} chapters MISSING! "
            f"({ocr_chapters} → {ai_chapters})"
        )
    
    if word_loss_pct > 15:
        errors.append(
            f"❌ CRITICAL: {word_loss_pct:.1f}% word loss (>15% threshold)"
        )
    elif word_loss_pct > 5:
        warnings.append(
            f"⚠️  WARNING: {word_loss_pct:.1f}% word loss (>5% threshold)"
        )
    
    if char_loss_pct > 20:
        errors.append(
            f"❌ CRITICAL: {char_loss_pct:.1f}% character loss (>20% threshold)"
        )
    
    # Find missing chapters
    if ai_chapters < ocr_chapters:
        ocr_chap_list = [
            line.strip() for line in ocr_text.split('\n')
            if line.strip().startswith('@@@')
        ]
        ai_chap_list = [
            line.strip() for line in ai_text.split('\n')
            if line.strip().startswith('@@@')
        ]
        
        missing = set(ocr_chap_list) - set(ai_chap_list)
        if missing:
            errors.append(f"\n📍 Missing chapters:")
            for chap in sorted(missing)[:10]:  # Show first 10
                errors.append(f"   - {chap}")
            if len(missing) > 10:
                errors.append(f"   ... and {len(missing) - 10} more")
    
    # Print results
    print("\n" + "=" * 80)
    if errors:
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        for error in errors:
            print(error)
        print("\n💡 LIKELY CAUSE: AI token limit truncation")
        print("   → Gemini 2.5 Flash has 8192 token output limit")
        print("   → Your chunk_size was too large (>25000 chars)")
        print("\n🔧 SOLUTION:")
        print("   1. Delete corrupted files: rm *_tesseract_ai.txt *.epub")
        print("   2. Edit config: Set chunk_size: 25000 in pdf2epub.yaml")
        print("   3. Re-run: ./pdf2epub.sh -i file.pdf --generate-epub-only --ai-proofread")
        print("\n📖 See: AI_TOKEN_LIMITS.md for details")
        return False
    elif warnings:
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
        print("=" * 80)
        for warning in warnings:
            print(warning)
        print("\n💡 Minor text reduction is normal for AI corrections")
        print("   (removing OCR junk, fixing formatting)")
        return True
    else:
        print("✅ VALIDATION PASSED")
        print("=" * 80)
        print("\n✨ AI proofreading completed successfully")
        print("   - All chapters preserved")
        print("   - Text reduction within normal range (<5%)")
        return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 verify_ai.py <base_name>")
        print("Example: python3 verify_ai.py RobinHoodT9")
        print("   (will check RobinHoodT9_tesseract.txt vs RobinHoodT9_tesseract_ai.txt)")
        return 1
    
    base_name = sys.argv[1]
    
    # Remove common suffixes
    base_name = base_name.replace('_tesseract.txt', '')
    base_name = base_name.replace('_tesseract_ai.txt', '')
    base_name = base_name.replace('.epub', '')
    base_name = base_name.replace('.pdf', '')
    
    success = verify_ai_output(base_name)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
