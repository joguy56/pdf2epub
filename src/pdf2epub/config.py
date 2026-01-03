"""Configuration management for pdf2epub."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .file_picker import pick_pdf_interactive


class OCRConfig(BaseSettings):
    """OCR engine configuration."""
    
    engine: str = Field(default="tesseract", description="OCR engine: tesseract or easyocr")
    language: str = Field(default="fra", description="Language code (fra, eng, etc.)")
    tesseract_dir: Optional[str] = Field(default=None, description="Tesseract data directory")
    confidence_threshold: int = Field(default=80, description="Minimum confidence for text acceptance")
    
    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        """Validate OCR engine choice."""
        if v not in ["tesseract", "easyocr"]:
            raise ValueError(f"Invalid OCR engine: {v}. Must be 'tesseract' or 'easyocr'")
        return v


@dataclass
class ChapterDetectionConfig:
    """Configuration for chapter detection."""
    
    enabled: bool = True
    threshold_percentage: int = 25  # Percentage from top where chapters can start
    max_position_percentage: int = 75  # Maximum percentage from top where chapters can be detected
    min_position_tolerance: float = 0.95  # Tolerance factor for minimum position (0.95 = 95% of threshold)
    detect_only_on_header: bool = False  # Only detect chapters with "Chapitre" keyword


class ImageProcessingConfig(BaseSettings):
    """Image processing configuration."""
    
    dpi: int = Field(default=200, description="DPI for PDF to image conversion")
    x_margin: int = Field(default=30, description="X-axis cropping margin for page-dewarp")
    y_margin: int = Field(default=50, description="Y-axis cropping margin for page-dewarp")
    gaussian_blur_kernel: tuple[int, int] = Field(
        default=(5, 5),
        description="Gaussian blur kernel size"
    )
    use_page_dewarp: bool = Field(default=True, description="Use external page-dewarp tool")


class TextProcessingConfig(BaseSettings):
    """Text post-processing configuration."""
    
    fix_hyphenation: bool = Field(default=True, description="Fix word breaks across lines")
    fix_dialogs: bool = Field(default=True, description="Indent dialog lines")
    fix_lettrine: bool = Field(default=True, description="Fix dropped capital letters")
    remove_blank_lines: bool = Field(default=True, description="Remove excessive blank lines")
    custom_filters: list[str] = Field(default_factory=list, description="Custom regex filters")


class AIProofreadingConfig(BaseSettings):
    """AI proofreading configuration."""
    
    enabled: bool = Field(default=False, description="Enable AI proofreading")
    provider: str = Field(default="gemini", description="AI provider: gemini, openai, claude")
    model: str = Field(default="gemini-1.5-flash", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key (or use env var)")
    chunk_size: int = Field(default=22000, description="Text chunk size in characters (safe for Gemini 8k limit, auto-adjusted if higher limits detected)")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    free_tier: bool = Field(default=True, description="Use free tier limits (15 RPM, slower processing)")
    delay_between_chunks: int = Field(default=5, description="Delay in seconds between chunks (default 5s for free tier 15 RPM)")
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate AI provider choice."""
        if v not in ["gemini", "openai", "claude"]:
            raise ValueError(f"Invalid AI provider: {v}. Must be 'gemini', 'openai', or 'claude'")
        return v
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        
        # Try environment variables
        env_vars = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY"
        }
        env_var = env_vars.get(self.provider)
        return os.getenv(env_var) if env_var else None


class PerformanceConfig(BaseSettings):
    """Performance and processing configuration."""
    
    parallel_processing: bool = Field(default=True, description="Enable parallel page processing")
    max_workers: Optional[int] = Field(default=None, description="Max worker threads (None = auto)")
    enable_resume: bool = Field(default=True, description="Enable resume on error")
    checkpoint_interval: int = Field(default=10, description="Save checkpoint every N pages")


class OutputConfig(BaseSettings):
    """Output configuration."""
    
    include_cover: bool = Field(default=True, description="Include cover image in EPUB")
    output_directory: Optional[str] = Field(default=None, description="Custom output directory")
    keep_intermediate_files: bool = Field(default=False, description="Keep temporary files")
    temp_directory: str = Field(default="./tmp", description="Temporary files directory")


class Pdf2EpubConfig(BaseSettings):
    """Main application configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="PDF2EPUB_",
        env_nested_delimiter="__",
        case_sensitive=False
    )
    
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chapter_detection: ChapterDetectionConfig = Field(default_factory=ChapterDetectionConfig)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    ai_proofreading: AIProofreadingConfig = Field(default_factory=AIProofreadingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Pdf2EpubConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Configuration object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def load_default(cls) -> "Pdf2EpubConfig":
        """
        Load default configuration with user overrides.
        
        Looks for config files in:
        1. Current directory: ./pdf2epub.yaml
        2. User home: ~/.pdf2epub.yaml
        3. System: /etc/pdf2epub/config.yaml
        
        Returns:
            Configuration object with defaults and overrides
        """
        config_paths = [
            Path("./pdf2epub.yaml"),
            Path.home() / ".pdf2epub.yaml",
            Path("/etc/pdf2epub/config.yaml"),
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    return cls.from_yaml(config_path)
                except Exception as e:
                    print(f"Warning: Failed to load {config_path}: {e}")
        
        # Return default configuration
        return cls()
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def interactive_config_full(config: "Pdf2EpubConfig", pdf_path: Optional[str] = None) -> tuple["Pdf2EpubConfig", dict[str, Any]]:
    """
    Mode interactif complet avec interface utilisateur améliorée.
    Demande TOUTES les informations nécessaires pour la conversion.
    
    Args:
        config: Configuration actuelle
        pdf_path: Chemin du fichier PDF (optionnel, demandé si non fourni)
    
    Returns:
        Tuple (config mise à jour, métadonnées du livre incluant pdf_path)
    """
    # Welcome header
    print("\n" + "═" * 75)
    print(f"{'📚 INTERACTIVE CONFIGURATION - pdf2epub':^75}")
    print("═" * 75 + "\n")
    
    metadata = {}
    
    # 0. PDF FILE (if not provided)
    if not pdf_path:
        print("📂 PDF FILE TO CONVERT")
        print("─" * 75)
        pdf_path = pick_pdf_interactive()
        metadata['pdf_path'] = pdf_path
        print()
    else:
        metadata['pdf_path'] = pdf_path
    
    pdf_name = Path(pdf_path).stem
    print(f"📄 File: {pdf_name}.pdf\n")
    
    # 1. BOOK TITLE
    print("📖 BOOK TITLE")
    print("─" * 75)
    title = input(f"Title [{pdf_name}]: ").strip() or pdf_name
    metadata['title'] = title
    print()
    
    # 2. AUTHOR
    print("✍️  AUTHOR")
    print("─" * 75)
    author = input("Author: ").strip()
    metadata['author'] = author
    print()
    
    # 3. LANGUAGE
    print("🌍 DOCUMENT LANGUAGE")
    print("─" * 75)
    print("Available languages:")
    print("  fra - Français")
    print("  eng - English")
    print("  deu - Deutsch")
    print("  spa - Español")
    print("  ita - Italiano")
    print("  (or other ISO 639-3 code)")
    lang = input(f"\nLanguage [{config.ocr.language}]: ").strip() or config.ocr.language
    config.ocr.language = lang
    metadata['language'] = lang
    print()
    
    # 4. COVER PAGE
    print("📕 FIRST PAGE (COVER)")
    print("─" * 75)
    print("How to handle the first page?")
    print("  1. Extract cover image + OCR (recommended)")
    print("  2. Cover image only (no OCR on page 1)")
    print("  3. Normal page (no cover in EPUB)")
    while True:
        cover = input("\nChoice [1/2/3] (default: 1): ").strip() or "1"
        if cover in ["1", "2", "3"]:
            if cover == "1":
                config.output.include_cover = True
                config._cover_ocr_mode = "include"
            elif cover == "2":
                config.output.include_cover = True
                config._cover_ocr_mode = "skip"
            else:
                config.output.include_cover = False
                config._cover_ocr_mode = "normal"
            break
        print("❌ Invalid choice. Use 1, 2 or 3.")
    print()
    
    # 5. CHAPTER DETECTION
    print("📑 AUTOMATIC CHAPTER DETECTION")
    print("─" * 75)
    print("Do you want to automatically detect chapters?")
    print("  → Detects titles at the top of pages")
    detect = input("\nEnable? [Y/n]: ").strip().lower()
    config.chapter_detection.enabled = detect not in ['n', 'no']
    
    if config.chapter_detection.enabled:
        print("\nDetection mode:")
        print("  → Standard: any text at the top of page")
        print("  → Strict: only if 'Chapter' keyword is present")
        strict = input("\nStrict mode? [y/N]: ").strip().lower()
        config.chapter_detection.detect_only_on_header = strict in ['y', 'yes']
    print()
    
    # 6. AI CORRECTION
    print("🤖 AI-POWERED PROOFREADING")
    print("─" * 75)
    print("Enable AI correction of OCR errors (Gemini)?")
    print("  ⚠️  Requires Gemini API key")
    print("  ⚠️  Increases processing time (~2-3 min per chunk)")
    use_ai = input("\nEnable? [y/N]: ").strip().lower()
    
    if use_ai in ['y', 'yes']:
        config.ai_proofreading.enabled = True
        
        # Check API key
        api_key = config.ai_proofreading.get_api_key()
        
        if not api_key:
            # Search automatically in common locations
            key_locations = [
                Path.home() / "gemini.key",
                Path.home() / ".gemini.key",
                Path("./gemini.key"),
            ]
            
            found_key = None
            for key_path in key_locations:
                if key_path.exists():
                    try:
                        with open(key_path) as f:
                            found_key = f.read().strip()
                        if found_key:
                            print(f"\n  ✅ API key found: {key_path}")
                            config.ai_proofreading.api_key = found_key
                            break
                    except Exception as e:
                        print(f"  ⚠️  Error reading {key_path}: {e}")
            
            if not found_key:
                print("\n  ⚠️  No Gemini API key found automatically")
                print("  Checked locations:")
                for loc in key_locations:
                    print(f"    - {loc}")
                
                # Ask for key file path
                key_file = input("\n  API key file path (or Enter to disable AI): ").strip()
                
                if key_file:
                    key_file_path = Path(key_file).expanduser()
                    if key_file_path.exists():
                        try:
                            with open(key_file_path) as f:
                                found_key = f.read().strip()
                            if found_key:
                                config.ai_proofreading.api_key = found_key
                                print(f"  ✅ API key loaded from: {key_file_path}")
                            else:
                                print("  ❌ Empty file")
                                config.ai_proofreading.enabled = False
                        except Exception as e:
                            print(f"  ❌ Read error: {e}")
                            config.ai_proofreading.enabled = False
                    else:
                        print(f"  ❌ File not found: {key_file_path}")
                        config.ai_proofreading.enabled = False
                else:
                    print("  ℹ️  AI proofreading disabled")
                    config.ai_proofreading.enabled = False
    print()
    
    # 7. PARALLEL PROCESSING
    cpu_count = os.cpu_count() or 4
    print("⚡ PARALLEL PROCESSING")
    print("─" * 75)
    print(f"Enable parallel processing? ({cpu_count} CPU cores available)")
    print("  → Faster but more resource-intensive")
    print("  → Workers automatically limited to 3 for page-dewarp to avoid timeouts")
    parallel = input("\nEnable? [Y/n]: ").strip().lower()
    config.performance.parallel_processing = parallel not in ['n', 'no']
    print()
    
    # Summary
    print("═" * 75)
    print("📋 CONFIGURATION SUMMARY")
    print("═" * 75)
    print(f"  File:         {Path(pdf_path).name}")
    print(f"  Title:        {metadata.get('title', 'N/A')}")
    print(f"  Author:       {metadata.get('author', 'N/A') or '(not specified)'}")
    print(f"  Language:     {config.ocr.language}")
    print(f"  Cover:        {'Yes' if config.output.include_cover else 'No'} (mode: {getattr(config, '_cover_ocr_mode', 'N/A')})")
    print(f"  Chapters:     {'Auto-detect' if config.chapter_detection.enabled else 'Disabled'}")
    if config.chapter_detection.enabled and config.chapter_detection.detect_only_on_header:
        print(f"                (strict mode)")
    print(f"  AI:           {'Yes' if config.ai_proofreading.enabled else 'No'}")
    print(f"  Parallel:     {'Yes (max 3 workers for dewarp)' if config.performance.parallel_processing else 'No'}")
    print("═" * 75 + "\n")
    
    confirm = input("✅ Start conversion with these settings? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("\n❌ Conversion cancelled.\n")
        exit(0)
    
    print()
    return config, metadata


def interactive_config(config: "Pdf2EpubConfig", args: Any) -> "Pdf2EpubConfig":
    """
    Mode interactif basique (conservé pour compatibilité).
    Demande seulement les options manquantes.
    
    Args:
        config: Configuration actuelle
        args: Arguments de ligne de commande (Namespace)
    
    Returns:
        Configuration mise à jour
    """
    # Si --wizard, utiliser le mode complet
    if hasattr(args, 'wizard') and args.wizard:
        pdf_path = args.input if hasattr(args, 'input') and args.input else None
        config, metadata = interactive_config_full(config, pdf_path)
        
        # Appliquer les métadonnées aux args
        if 'pdf_path' in metadata and not args.input:
            args.input = metadata['pdf_path']
        if 'title' in metadata and not args.title:
            args.title = metadata.get('title')
        if 'author' in metadata and not args.author:
            args.author = metadata.get('author')
        if 'language' in metadata and not args.language:
            args.language = metadata.get('language')
        
        return config
    
    # Sinon, mode basique (ancien comportement)
    print("\n" + "="*70)
    print("   QUICK CONFIGURATION - pdf2epub")
    print("="*70 + "\n")
    
    # Juste les questions pour les flags manquants
    if not (hasattr(args, 'cover_mode') and args.cover_mode) and not (hasattr(args, 'no_cover') and args.no_cover):
        while True:
            print("📖 Cover page handling:")
            print("  1. Image only (no OCR)")
            print("  2. OCR + image (recommended)")
            print("  3. Normal page")
            choice = input("\nChoice [1/2/3] (default: 2): ").strip() or "2"
            if choice in ["1", "2", "3"]:
                if choice == "1":
                    config.output.include_cover = True
                    config._cover_ocr_mode = "skip"
                elif choice == "2":
                    config.output.include_cover = True
                    config._cover_ocr_mode = "include"
                else:
                    config.output.include_cover = False
                    config._cover_ocr_mode = "normal"
                break
            print("Invalid choice.")
    
    if not hasattr(args, 'language') or args.language is None:
        print("\n🌍 Language: fra (French), eng (English), etc.")
        lang = input(f"Language [{config.ocr.language}]: ").strip() or config.ocr.language
        config.ocr.language = lang
    
    if not hasattr(args, 'ai_proofread') or not args.ai_proofread:
        use_ai = input("\n🤖 Enable AI proofreading? [y/N]: ").strip().lower()
        if use_ai in ['y', 'yes', 'o', 'oui']:
            config.ai_proofreading.enabled = True
    
    if not hasattr(args, 'no_chap_detection') or not args.no_chap_detection:
        detect = input("\n📑 Enable chapter detection? [Y/n]: ").strip().lower()
        config.chapter_detection.enabled = detect not in ['n', 'no', 'non']
    
    if not hasattr(args, 'no_parallel') or not args.no_parallel:
        parallel = input("\n⚡ Enable parallel processing? [Y/n]: ").strip().lower()
        config.performance.parallel_processing = parallel not in ['n', 'no', 'non']
    
    print("\n" + "="*70 + "\n")
    return config


def create_default_config_file(output_path: str | Path) -> None:
    """
    Create a default configuration file with comments.
    
    Args:
        output_path: Path to save the configuration file
    """
    config_content = """# pdf2epub Configuration File
# See documentation at: https://github.com/jguyot/pdf2epub

# OCR Configuration
ocr:
  engine: tesseract  # tesseract or easyocr
  language: fra      # Language code: fra (French), eng (English)
  tesseract_dir: null  # Path to tesseract data directory (or use TESSDATA_PREFIX env var)
  confidence_threshold: 80  # Minimum OCR confidence score (0-100)

# Chapter Detection
chapter_detection:
  enabled: true  # Automatically detect chapters
  threshold_percentage: 25  # % of page height from top for chapter detection
  detect_only_on_header: false  # Only detect chapters with "Chapitre" keyword
  min_chapter_text_length: 1
  max_chapter_text_length: 100

# Image Processing
image_processing:
  dpi: 200  # DPI for PDF conversion
  x_margin: 30  # X-axis margin for page-dewarp
  y_margin: 50  # Y-axis margin for page-dewarp
  gaussian_blur_kernel: [5, 5]
  use_page_dewarp: true  # Use external page-dewarp tool

# Text Processing
text_processing:
  fix_hyphenation: true  # Fix word breaks across lines
  fix_dialogs: true      # Indent dialog lines
  fix_lettrine: true     # Fix dropped capital letters
  remove_blank_lines: true
  custom_filters: []  # List of custom regex patterns to remove

# AI Proofreading (EXPERIMENTAL)
ai_proofreading:
  enabled: false  # Enable AI-powered text correction
  provider: gemini  # gemini, openai, or claude
  model: gemini-1.5-flash  # Model to use
  api_key: null  # API key (or use environment variable)
  chunk_size: 22000  # Safe for Gemini 8k limit, will auto-increase if higher limits detected
  max_retries: 3
  timeout: 60

# Performance
performance:
  parallel_processing: true  # Process pages in parallel
  max_workers: null  # Max threads (null = auto)
  enable_resume: true  # Resume on error
  checkpoint_interval: 10  # Save progress every N pages

# Output
output:
  include_cover: true  # Include cover image
  output_directory: null  # Custom output directory
  keep_intermediate_files: false  # Keep temp files
  temp_directory: ./tmp

# Logging
debug: false
log_level: INFO  # DEBUG, INFO, WARNING, ERROR
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(config_content)
