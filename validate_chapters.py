#!/usr/bin/env python3
"""Standalone tool to validate chapter numbering in OCR text files."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf2epub.chapter_validator import ChapterValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate chapter numbering in OCR text files to detect OCR errors"
    )
    parser.add_argument(
        "file",
        help="Path to the _tesseract.txt file to validate"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (fail on warnings)"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return 1
    
    print(f"🔍 Validating chapters in: {file_path.name}\n")
    
    validator = ChapterValidator()
    errors, warnings = validator.validate_file(str(file_path))
    
    validator.print_report()
    
    # Return exit code
    if errors:
        print("\n❌ Validation FAILED - errors found")
        return 1
    elif warnings and args.strict:
        print("\n⚠️  Validation FAILED - warnings found (strict mode)")
        return 1
    elif warnings:
        print("\n⚠️  Validation passed with warnings")
        return 0
    else:
        print("\n✅ Validation PASSED - no issues")
        return 0


if __name__ == "__main__":
    sys.exit(main())
