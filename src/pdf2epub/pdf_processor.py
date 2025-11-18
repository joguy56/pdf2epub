"""PDF processing module for converting PDFs to images."""

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from pdf2epub.config import ImageProcessingConfig, OutputConfig
from pdf2epub.utils import ensure_directory

logger = logging.getLogger(__name__)


@dataclass
class PageImage:
    """Represents a processed page image."""
    
    page_number: int
    file_path: Path
    width: int
    height: int
    is_cover: bool = False


class PDFProcessorError(Exception):
    """Base exception for PDF processing errors."""
    pass


class PDFProcessor:
    """Handles PDF to image conversion with preprocessing."""
    
    def __init__(
        self,
        image_config: ImageProcessingConfig,
        output_config: OutputConfig
    ):
        """
        Initialize PDF processor.
        
        Args:
            image_config: Image processing configuration
            output_config: Output configuration
        """
        self.image_config = image_config
        self.output_config = output_config
        self.temp_dir = Path(output_config.temp_directory)
        
    def check_dependencies(self) -> dict[str, bool]:
        """
        Check if required external dependencies are available.
        
        Returns:
            Dictionary of dependency name to availability status
        """
        dependencies = {}
        
        # Check page-dewarp
        if self.image_config.use_page_dewarp:
            try:
                result = subprocess.run(
                    ["page-dewarp", "--help"],
                    capture_output=True,
                    timeout=5
                )
                dependencies["page-dewarp"] = result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                dependencies["page-dewarp"] = False
                logger.warning(
                    "page-dewarp not found. Install it for better image quality: "
                    "https://github.com/mzucker/page_dewarp"
                )
        
        return dependencies
    
    def get_pdf_info(self, pdf_path: Path) -> dict[str, any]:
        """
        Get PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
            
        Raises:
            PDFProcessorError: If PDF cannot be read
        """
        try:
            info = pdfinfo_from_path(pdf_path)
            logger.info(f"PDF info: {info['Pages']} pages")
            return info
        except Exception as e:
            raise PDFProcessorError(f"Failed to read PDF info: {e}") from e
    
    def convert_pdf_to_images(
        self,
        pdf_path: Path,
        include_cover: bool = True
    ) -> list[PageImage]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            include_cover: Whether to save the first page as cover
            
        Returns:
            List of PageImage objects
            
        Raises:
            PDFProcessorError: If conversion fails
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Create temp directory
        ensure_directory(self.temp_dir)
        
        # Get PDF info
        try:
            info = self.get_pdf_info(pdf_path)
            max_pages = info["Pages"]
        except Exception as e:
            raise PDFProcessorError(f"Failed to get PDF info: {e}") from e
        
        page_images: list[PageImage] = []
        
        # Convert pages in batches to manage memory
        batch_size = 10
        start_page = 1
        
        try:
            for batch_start in range(1, max_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, max_pages)
                logger.info(f"Converting pages {batch_start}-{batch_end}/{max_pages}")
                
                pdf_pages = convert_from_path(
                    pdf_path,
                    dpi=self.image_config.dpi,
                    first_page=batch_start,
                    last_page=batch_end
                )
                
                for idx, page in enumerate(pdf_pages):
                    page_num = batch_start + idx
                    
                    # Handle cover page separately
                    if page_num == 1 and include_cover:
                        cover_path = self.temp_dir / "cover.jpg"
                        page.save(cover_path, "JPEG")
                        page_images.append(PageImage(
                            page_number=0,
                            file_path=cover_path,
                            width=page.width,
                            height=page.height,
                            is_cover=True
                        ))
                        logger.info(f"Saved cover image: {cover_path}")
                        continue
                    
                    # Save regular page
                    adjusted_page_num = page_num - 1 if include_cover else page_num
                    filename = f"page_{adjusted_page_num:03d}.jpg"
                    file_path = self.temp_dir / filename
                    page.save(file_path, "JPEG")
                    
                    page_images.append(PageImage(
                        page_number=adjusted_page_num,
                        file_path=file_path,
                        width=page.width,
                        height=page.height,
                        is_cover=False
                    ))
                    
                logger.info(f"Batch {batch_start}-{batch_end} converted")
                
        except Exception as e:
            raise PDFProcessorError(f"Failed to convert PDF pages: {e}") from e
        
        logger.info(f"Converted {len(page_images)} images")
        return page_images
    
    def preprocess_image(self, image_path: Path) -> Path:
        """
        Apply preprocessing to an image.
        
        Preprocessing steps:
        1. Convert to grayscale
        2. Apply Gaussian blur
        3. Apply Otsu thresholding
        4. Optional: page dewarping
        
        Args:
            image_path: Path to image file
            
        Returns:
            Path to preprocessed image
            
        Raises:
            PDFProcessorError: If preprocessing fails
        """
        try:
            # Load image
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise PDFProcessorError(f"Failed to load image: {image_path}")
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, self.image_config.gaussian_blur_kernel, 2)
            
            # Apply Otsu thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to BGR for consistency
            processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Save preprocessed image
            cv2.imwrite(str(image_path), processed)
            
            # Apply page dewarping if enabled
            if self.image_config.use_page_dewarp:
                self._apply_page_dewarp(image_path)
            
            return image_path
            
        except Exception as e:
            raise PDFProcessorError(f"Failed to preprocess image {image_path}: {e}") from e
    
    def _apply_page_dewarp(self, image_path: Path) -> None:
        """
        Apply page dewarping using external tool.
        
        Args:
            image_path: Path to image file
        """
        try:
            cmd = [
                "page-dewarp",
                "-oscreen",
                "-d0",
                "-f", "1.2",
                f"-x", str(self.image_config.x_margin),
                f"-y", str(self.image_config.y_margin),
                image_path.name
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.temp_dir,
                capture_output=True,
                timeout=30,
                text=True
            )
            
            if result.returncode != 0:
                logger.warning(
                    f"page-dewarp failed for {image_path.name}: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            logger.warning(f"page-dewarp timeout for {image_path.name}")
        except Exception as e:
            logger.warning(f"page-dewarp error for {image_path.name}: {e}")
    
    def process_all_images(
        self,
        page_images: list[PageImage],
        parallel: bool = True
    ) -> list[PageImage]:
        """
        Preprocess all images, optionally in parallel.
        
        Args:
            page_images: List of page images to process
            parallel: Whether to process in parallel
            
        Returns:
            List of processed PageImage objects
        """
        # Don't process cover
        images_to_process = [img for img in page_images if not img.is_cover]
        
        if not parallel or len(images_to_process) <= 1:
            # Sequential processing
            for img in images_to_process:
                logger.info(f"Preprocessing page {img.page_number}")
                self.preprocess_image(img.file_path)
            return page_images
        
        # Parallel processing
        logger.info(f"Preprocessing {len(images_to_process)} images in parallel")
        
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.preprocess_image, img.file_path): img
                for img in images_to_process
            }
            
            for future in as_completed(futures):
                img = futures[future]
                try:
                    future.result()
                    logger.debug(f"Preprocessed page {img.page_number}")
                except Exception as e:
                    logger.error(f"Failed to preprocess page {img.page_number}: {e}")
        
        return page_images
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if not self.output_config.keep_intermediate_files:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
