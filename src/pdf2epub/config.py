"""Configuration management for pdf2epub."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    chunk_size: int = Field(default=50000, description="Text chunk size in characters")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    
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
  chunk_size: 50000  # Characters per request
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
