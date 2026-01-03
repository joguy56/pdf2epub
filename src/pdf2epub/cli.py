"""Command-line interface for pdf2epub."""

import argparse
import logging
import sys
from pathlib import Path

import colorlog

from pdf2epub import Pdf2EpubConfig, ConversionPipeline, __version__
from pdf2epub.config import create_default_config_file, interactive_config


def setup_logging(debug: bool = False) -> None:
    """Set up colored logging."""
    logger = logging.getLogger()
    logger.handlers = []
    
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.addHandler(handler)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert scanned PDF books to EPUB ebooks using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  pdf2epub -i book.pdf -a "Author Name" -t "Book Title"
  
  # French book with custom margins
  pdf2epub -i livre.pdf -a "Auteur" -t "Titre" -l fra -x 40 -y 60
  
  # Skip PDF conversion (resume from images)
  pdf2epub -i book.pdf --recognize-only
  
  # Skip OCR (generate EPUB from existing text)
  pdf2epub -i book.pdf --generate-epub-only
  
  # With AI proofreading
  pdf2epub -i book.pdf --ai-proofread
  
  # Generate default config file
  pdf2epub --create-config

For more information: https://github.com/jguyot/pdf2epub
        """
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"pdf2epub {__version__}"
    )
    
    # Config management
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file at ./pdf2epub.yaml"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: looks in standard locations)"
    )
    
    # Input/Output
    parser.add_argument(
        "-i", "--input",
        type=Path,
        help="Input PDF file (required unless --create-config)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: same as input file)"
    )
    
    # Book metadata
    parser.add_argument(
        "-a", "--author",
        help="Book author"
    )
    
    parser.add_argument(
        "-t", "--title",
        help="Book title (default: filename)"
    )
    
    # OCR options
    parser.add_argument(
        "-O", "--ocr-engine",
        choices=["tesseract", "easyocr"],
        help="OCR engine to use"
    )
    
    parser.add_argument(
        "-l", "--language",
        help="Language code (fra, eng, etc.)"
    )
    
    parser.add_argument(
        "--tesseract-dir",
        help="Tesseract data directory"
    )
    
    # Processing stages
    parser.add_argument(
        "-r", "--recognize-only",
        action="store_true",
        help="Skip PDF conversion, start from images"
    )
    
    parser.add_argument(
        "-g", "--generate-epub-only",
        action="store_true",
        help="Skip OCR, generate EPUB from existing text file"
    )
    
    # Chapter detection
    parser.add_argument(
        "--no-chap-detection",
        action="store_true",
        help="Disable automatic chapter detection"
    )
    
    parser.add_argument(
        "--chap-detect-thres-pct",
        type=int,
        help="Chapter detection threshold (percentage of page height)"
    )
    
    parser.add_argument(
        "--detect-only-on-chap-header",
        action="store_true",
        help="Only detect chapters with 'Chapitre' keyword"
    )
    
    # Image processing
    parser.add_argument(
        "-x", "--x-margin",
        type=int,
        help="X-axis cropping margin for page-dewarp"
    )
    
    parser.add_argument(
        "-y", "--y-margin",
        type=int,
        help="Y-axis cropping margin for page-dewarp"
    )
    
    parser.add_argument(
        "--no-cover",
        action="store_true",
        help="Don't include cover image in EPUB"
    )
    
    parser.add_argument(
        "--cover-mode",
        type=int,
        choices=[1, 2, 3],
        help="Cover page handling: 1=image only, 2=OCR+image (default), 3=normal page"
    )
    
    # Text processing
    parser.add_argument(
        "-f", "--filter",
        action="append",
        help="Regex pattern to filter out unwanted lines (can specify multiple times)"
    )
    
    # AI proofreading
    parser.add_argument(
        "--ai-proofread",
        action="store_true",
        help="Enable AI-powered text correction"
    )
    
    parser.add_argument(
        "--ai-provider",
        choices=["gemini", "openai", "claude"],
        help="AI provider for proofreading"
    )
    
    # Performance
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker threads"
    )
    
    # Debug
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after conversion"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: skip all interactive prompts (use defaults or CLI args)"
    )
    
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Interactive wizard mode: asks ALL questions with nice UI (overrides --batch)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean temporary files (*_tesseract.txt, *_tesseract_ai.txt, tmp/)"
    )
    
    return parser


def apply_cli_overrides(config: Pdf2EpubConfig, args: argparse.Namespace) -> Pdf2EpubConfig:
    """Apply command-line argument overrides to configuration."""
    # OCR
    if args.ocr_engine:
        config.ocr.engine = args.ocr_engine
    if args.language:
        config.ocr.language = args.language
    if args.tesseract_dir:
        config.ocr.tesseract_dir = args.tesseract_dir
    
    # Chapter detection
    if args.no_chap_detection:
        config.chapter_detection.enabled = False
    if args.chap_detect_thres_pct:
        config.chapter_detection.threshold_percentage = args.chap_detect_thres_pct
    if args.detect_only_on_chap_header:
        config.chapter_detection.detect_only_on_header = True
    
    # Image processing & cover handling
    if args.x_margin:
        config.image_processing.x_margin = args.x_margin
    if args.y_margin:
        config.image_processing.y_margin = args.y_margin
    
    # Cover mode handling
    if args.cover_mode:
        if args.cover_mode == 1:
            config.output.include_cover = True
            config._cover_ocr_mode = "skip"
        elif args.cover_mode == 2:
            config.output.include_cover = True
            config._cover_ocr_mode = "include"
        elif args.cover_mode == 3:
            config.output.include_cover = False
            config._cover_ocr_mode = "normal"
    elif args.no_cover:
        config.output.include_cover = False
        config._cover_ocr_mode = "normal"
    
    # Text processing
    if args.filter:
        config.text_processing.custom_filters = args.filter
    
    # AI proofreading
    if args.ai_proofread:
        config.ai_proofreading.enabled = True
    if args.ai_provider:
        config.ai_proofreading.provider = args.ai_provider
    
    # Performance
    if args.no_parallel:
        config.performance.parallel_processing = False
    if args.max_workers:
        config.performance.max_workers = args.max_workers
    
    # Output
    if args.output_dir:
        config.output.output_directory = str(args.output_dir)
    if args.keep_temp:
        config.output.keep_intermediate_files = True
    
    # Debug
    if args.debug:
        config.debug = True
        config.log_level = "DEBUG"
    
    return config


def clean_temporary_files(pdf_path: Path, config: Pdf2EpubConfig) -> None:
    """Clean temporary files for a given PDF."""
    import shutil
    import logging
    from pdf2epub.utils import get_file_base_name
    
    logger = logging.getLogger(__name__)
    base_name = get_file_base_name(pdf_path)
    parent_dir = pdf_path.parent
    
    files_to_remove = [
        parent_dir / f"{base_name}_tesseract.txt",
        parent_dir / f"{base_name}_tesseract_ai.txt",
        parent_dir / f"{base_name}_easyocr.txt",
    ]
    
    # Remove text files
    removed_count = 0
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"🗑️  Removed: {file_path.name}")
            removed_count += 1
    
    # Remove tmp directory
    tmp_dir = Path(config.output.temp_directory)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.info(f"🗑️  Removed directory: {tmp_dir}")
        removed_count += 1
    
    if removed_count == 0:
        logger.info("✨ No temporary files to clean")
    else:
        logger.info(f"✅ Cleaned {removed_count} temporary file(s)/directory")


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --create-config
    if args.create_config:
        output_path = Path("./pdf2epub.yaml")
        create_default_config_file(output_path)
        print(f"✅ Created default configuration file: {output_path}")
        print("Edit this file to customize settings, then run pdf2epub with your PDF.")
        return 0
    
    # Validate required arguments (sauf si --wizard qui demandera le fichier)
    if not args.wizard and not args.input:
        parser.error("argument -i/--input is required (unless using --create-config or --wizard)")
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            config = Pdf2EpubConfig.from_yaml(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = Pdf2EpubConfig.load_default()
        
        # Apply CLI overrides
        config = apply_cli_overrides(config, args)
        
        # Store batch mode flag in config for pipeline access
        config._batch_mode = args.batch
        
        # Interactive configuration for essential options
        # Skip if user is in batch/automation mode (MAIS PAS si wizard)
        if args.wizard or not (args.recognize_only or args.generate_epub_only or args.batch):
            config = interactive_config(config, args)
        
        # Vérifier que args.input existe maintenant (après wizard)
        if not args.input:
            print(f"❌ Error: No input file specified", file=sys.stderr)
            return 1
        
        args.input = Path(args.input)
        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        
        # Handle clean mode
        if args.clean:
            logger.info("🧹 Cleaning temporary files...")
            clean_temporary_files(args.input, config)
            return 0
        
        # Create pipeline
        pipeline = ConversionPipeline(config)
        
        # Run conversion
        logger.info(f"🚀 Starting conversion: {args.input}")
        epub_path = pipeline.convert(
            pdf_path=args.input,
            title=args.title,
            author=args.author,
            skip_pdf_conversion=args.recognize_only or args.generate_epub_only,
            skip_ocr=args.generate_epub_only  # Only skip OCR if generating EPUB only
        )
        
        print(f"\n✅ Success! EPUB created: {epub_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception(f"❌ Conversion failed: {e}")
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
