"""pdf2epub package."""

__version__ = "2.0.0"
__author__ = "jguyot"

from pdf2epub.config import Pdf2EpubConfig
from pdf2epub.pipeline import ConversionPipeline

__all__ = ["Pdf2EpubConfig", "ConversionPipeline", "__version__"]
