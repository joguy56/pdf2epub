"""AI-powered text proofreading module."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from pdf2epub.config import AIProofreadingConfig
from pdf2epub.utils import CHAPTER_MARKER, FOOTNOTE_MARKER, SECTION_MARKER

logger = logging.getLogger(__name__)


class AIProofreaderError(Exception):
    """Base exception for AI proofreading errors."""
    pass


class AIProofreader(ABC):
    """Abstract base class for AI proofreading providers."""
    
    def __init__(self, config: AIProofreadingConfig):
        """
        Initialize AI proofreader.
        
        Args:
            config: AI proofreading configuration
        """
        self.config = config
        self.api_key = config.get_api_key()
        
        if not self.api_key:
            raise AIProofreaderError(
                f"API key not configured for {config.provider}. "
                f"Set it in config or environment variable."
            )
    
    @abstractmethod
    def proofread_chunk(self, text: str) -> str:
        """
        Proofread a chunk of text.
        
        Args:
            text: Text chunk to proofread
            
        Returns:
            Corrected text
            
        Raises:
            AIProofreaderError: If proofreading fails
        """
        pass
    
    def proofread_text(self, text: str) -> str:
        """
        Proofread full text, splitting into chunks if needed.
        
        Args:
            text: Full text to proofread
            
        Returns:
            Corrected text
        """
        logger.info(f"Starting AI proofreading with {self.config.provider}")
        
        # Split into chunks
        chunks = self._split_text(text, self.config.chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Proofread each chunk
        corrected_chunks = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Proofreading chunk {i}/{len(chunks)}")
            try:
                corrected = self.proofread_chunk(chunk)
                corrected_chunks.append(corrected)
            except Exception as e:
                logger.error(f"Failed to proofread chunk {i}: {e}")
                # Fall back to original text for this chunk
                corrected_chunks.append(chunk)
        
        # Rejoin chunks
        result = "".join(corrected_chunks)
        logger.info("AI proofreading complete")
        return result
    
    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Split text into chunks, preserving structure markers.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        # If text fits in one chunk, return as-is
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs
        paragraphs = text.split("\n")
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 1 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If single paragraph is too large, split by sentences
                if len(para) > chunk_size:
                    sentences = para.split(". ")
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 > chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence + ". "
                        else:
                            current_chunk += sentence + ". "
                else:
                    current_chunk = para + "\n"
            else:
                current_chunk += para + "\n"
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _build_prompt(self, text: str) -> str:
        """
        Build prompt for AI proofreading.
        
        Args:
            text: Text to proofread
            
        Returns:
            Formatted prompt
        """
        return f"""Corrige les fautes d'orthographe et de grammaire dans ce texte OCR français.

RÈGLES IMPORTANTES:
1. Corrige UNIQUEMENT les erreurs évidentes d'OCR et de grammaire
2. Préserve la mise en forme (sauts de ligne, espaces, indentation)
3. NE MODIFIE PAS les marqueurs spéciaux: {CHAPTER_MARKER}, {SECTION_MARKER}, {FOOTNOTE_MARKER}
4. NE CHANGE PAS le sens ou le style du texte
5. Retourne UNIQUEMENT le texte corrigé, sans commentaire ni explication

TEXTE À CORRIGER:
{text}"""


class GeminiProofreader(AIProofreader):
    """Google Gemini proofreader implementation."""
    
    def __init__(self, config: AIProofreadingConfig):
        """Initialize Gemini proofreader."""
        super().__init__(config)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(config.model)
        except ImportError:
            raise AIProofreaderError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with Gemini."""
        try:
            prompt = self._build_prompt(text)
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistency
                    "max_output_tokens": len(text) + 1000,
                }
            )
            return response.text
        except Exception as e:
            raise AIProofreaderError(f"Gemini proofreading failed: {e}") from e


class OpenAIProofreader(AIProofreader):
    """OpenAI proofreader implementation."""
    
    def __init__(self, config: AIProofreadingConfig):
        """Initialize OpenAI proofreader."""
        super().__init__(config)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise AIProofreaderError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with OpenAI."""
        try:
            prompt = self._build_prompt(text)
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Tu es un correcteur expert en français."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=len(text) + 1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise AIProofreaderError(f"OpenAI proofreading failed: {e}") from e


class ClaudeProofreader(AIProofreader):
    """Anthropic Claude proofreader implementation."""
    
    def __init__(self, config: AIProofreadingConfig):
        """Initialize Claude proofreader."""
        super().__init__(config)
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise AIProofreaderError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with Claude."""
        try:
            prompt = self._build_prompt(text)
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=len(text) + 1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise AIProofreaderError(f"Claude proofreading failed: {e}") from e


def create_proofreader(config: AIProofreadingConfig) -> Optional[AIProofreader]:
    """
    Factory function to create appropriate proofreader.
    
    Args:
        config: AI proofreading configuration
        
    Returns:
        AIProofreader instance or None if disabled
        
    Raises:
        AIProofreaderError: If provider is invalid
    """
    if not config.enabled:
        return None
    
    providers = {
        "gemini": GeminiProofreader,
        "openai": OpenAIProofreader,
        "claude": ClaudeProofreader
    }
    
    provider_class = providers.get(config.provider)
    if not provider_class:
        raise AIProofreaderError(
            f"Unknown AI provider: {config.provider}. "
            f"Must be one of: {', '.join(providers.keys())}"
        )
    
    return provider_class(config)
