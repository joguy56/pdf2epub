"""AI-powered text proofreading module."""

import logging
import os
import ssl
import warnings
from abc import ABC, abstractmethod
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from pdf2epub.config import AIProofreadingConfig
from pdf2epub.utils import CHAPTER_MARKER, FOOTNOTE_MARKER, SECTION_MARKER

# Disable SSL verification warnings (for environments with SSL issues)
try:
    import urllib3
    warnings.filterwarnings('ignore')
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Disable SSL verification for gRPC and Python
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = ''
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey patch requests to disable SSL verification
try:
    import requests.sessions
    original_request_method = requests.sessions.Session.request
    
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request_method(self, method, url, **kwargs)
    
    requests.sessions.Session.request = patched_request
except ImportError:
    pass

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
        
        # Will be detected automatically on first use
        self._detected_output_limit = None
        self._optimal_chunk_size = None
    
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
        
        # Detect output limit on first use (with better error handling)
        if self._detected_output_limit is None:
            try:
                self._detect_output_limit()
            except Exception as e:
                logger.error(f"⚠️  Limit detection failed: {e}")
                logger.warning("   Using fallback: 8000 tokens limit (Gemini default)")
                self._detected_output_limit = 8000
                self._optimal_chunk_size = int(8000 * 3.5 * 0.8)  # ~22k chars
        
        # Use detected optimal chunk size instead of config value
        chunk_size = self._optimal_chunk_size or self.config.chunk_size
        if self._optimal_chunk_size and self._optimal_chunk_size != self.config.chunk_size:
            logger.info(
                f"📊 Using auto-detected chunk_size: {chunk_size:,} chars "
                f"(config: {self.config.chunk_size:,}) based on {self._detected_output_limit:,} token limit"
            )
        
        # Split into chunks
        chunks = self._split_text(text, chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Proofread each chunk
        corrected_chunks = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Proofreading chunk {i}/{len(chunks)}")
            try:
                corrected = self.proofread_chunk(chunk)
                corrected_chunks.append(corrected)
                
                # Add delay between chunks to avoid rate limiting (except for last chunk)
                if i < len(chunks):
                    import time
                    delay = self.config.delay_between_chunks
                    rate_info = " (free tier: 15 RPM)" if self.config.free_tier else ""
                    logger.info(f"Waiting {delay}s before next chunk to avoid rate limiting{rate_info}...")
                    time.sleep(delay)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to proofread chunk {i}: {error_msg}")
                
                # Check if it's a quota error
                if "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower() or "429" in error_msg:
                    logger.error("❌ QUOTA LIMITE ATTEINTE !")
                    logger.error("   Plan gratuit Gemini : 15 requêtes/minute, 1500 requêtes/jour")
                    logger.error(f"   Chunks traités : {i-1}/{len(chunks)}")
                    logger.error("   💡 Solution : Attendre 24h ou passer au plan payant")
                    logger.error(f"   📁 Progression sauvegardée dans : {self.config.provider}_checkpoint_{i-1}.txt")
                    # Save checkpoint
                    checkpoint_text = "".join(corrected_chunks)
                    checkpoint_file = f"{self.config.provider}_checkpoint_{i-1}_of_{len(chunks)}.txt"
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        f.write(checkpoint_text)
                    raise AIProofreaderError(f"Quota limite atteinte au chunk {i}/{len(chunks)}. Checkpoint sauvegardé.") from e
                
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
        return f"""Fix spelling and grammar errors in this French OCR text.

IMPORTANT RULES:
1. Fix ONLY obvious OCR and grammar errors
2. Preserve formatting (line breaks, spaces, indentation)
3. DO NOT MODIFY special markers: {CHAPTER_MARKER}, {SECTION_MARKER}, {FOOTNOTE_MARKER}
4. DO NOT CHANGE the meaning or style of the text
5. Return ONLY the corrected text, without comments or explanations

TEXT TO CORRECT:
{text}"""
    
    def _detect_output_limit(self) -> None:
        """
        Detect the actual output token limit by testing with sample text.
        Sets self._detected_output_limit and self._optimal_chunk_size.
        """
        logger.info("🔍 Detecting AI output token limit...")
        
        # Create a test text that will expand when corrected
        # Use repetitive text with errors to ensure AI returns something substantial
        test_text = "Ceci est un texte de test pour detecter la limite de tokens. " * 200  # ~12k chars
        
        # Test with increasing output limits to find the actual ceiling
        test_limits = [4000, 8000, 16000, 32000]
        detected_limit = None
        max_result_tokens = 0
        
        for limit in test_limits:
            try:
                logger.debug(f"Testing with max_output_tokens={limit}")
                result = self._test_with_limit(test_text, limit)
                
                # Estimate tokens in result (conservative: 3 chars/token)
                result_tokens = len(result) // 3
                max_result_tokens = max(max_result_tokens, result_tokens)
                
                logger.debug(f"  Got {result_tokens:,} tokens (requested {limit:,})")
                
                # If result is much smaller than requested, we hit the real limit
                if result_tokens < limit * 0.7:  # Less than 70% of requested
                    detected_limit = int(result_tokens * 1.1)  # Add 10% margin
                    logger.info(f"✅ Detected output limit: ~{detected_limit:,} tokens (hit ceiling at {limit:,})")
                    break
                    
                # If we got close to what we asked, this limit works
                if result_tokens >= limit * 0.8:
                    detected_limit = limit
                    logger.debug(f"  Limit {limit:,} seems OK, trying higher...")
                    continue  # Try next higher limit
                    
            except Exception as e:
                logger.debug(f"Test with limit {limit} failed: {e}")
                # If we got some results before, use those
                if max_result_tokens > 0:
                    detected_limit = int(max_result_tokens * 1.1)
                    logger.warning(f"⚠️  Test failed at {limit:,} tokens, using previous max: {detected_limit:,}")
                    break
        
        # Fallback if detection failed completely
        if detected_limit is None or detected_limit < 4000:
            # Very low limit detected or total failure - use conservative default
            if max_result_tokens > 0:
                detected_limit = max(max_result_tokens, 8000)
                logger.warning(
                    f"⚠️  Detection unclear (max observed: {max_result_tokens:,}), "
                    f"using safe default: {detected_limit:,} tokens"
                )
            else:
                detected_limit = 8000  # Conservative default for Gemini
                logger.warning(
                    f"⚠️  Could not detect limit (all tests failed), "
                    f"using conservative default: {detected_limit:,} tokens"
                )
        
        self._detected_output_limit = detected_limit
        
        # Calculate optimal chunk size
        # Formula: limit_tokens × 3.5 chars/token × 0.8 safety margin
        self._optimal_chunk_size = int(detected_limit * 3.5 * 0.8)
        
        logger.info(
            f"📊 Auto-configured chunk_size: {self._optimal_chunk_size:,} chars "
            f"(~{self._optimal_chunk_size // 3:,} tokens input → ~{detected_limit:,} tokens output)"
        )
    
    @abstractmethod
    def _test_with_limit(self, text: str, limit: int) -> str:
        """
        Test proofreading with a specific output token limit.
        
        Args:
            text: Test text
            limit: Max output tokens to test
            
        Returns:
            Corrected text
        """
        pass


class GeminiProofreader(AIProofreader):
    """Google Gemini proofreader implementation."""
    
    def __init__(self, config: AIProofreadingConfig):
        """Initialize Gemini proofreader."""
        super().__init__(config)
        
        try:
            import google.generativeai as genai
            # Use REST transport to avoid gRPC SSL issues
            genai.configure(
                api_key=self.api_key,
                transport='rest'
            )
            # Use gemini-2.5-flash instead of gemini-1.5-flash
            model_name = config.model
            if model_name == 'gemini-1.5-flash':
                model_name = 'gemini-2.5-flash'
                logger.info(f"Using {model_name} instead of gemini-1.5-flash")
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise AIProofreaderError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
    
    def _test_with_limit(self, text: str, limit: int) -> str:
        """Test with specific output limit."""
        prompt = self._build_prompt(text)
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": limit,
            }
        )
        return response.text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with Gemini."""
        try:
            prompt = self._build_prompt(text)
            
            # Use detected limit (auto-configured at startup)
            if self._detected_output_limit is None:
                # Fallback if detection was skipped somehow
                max_output = 8000
            else:
                # Calculate based on input size, but never exceed detected limit
                estimated_input_tokens = len(text) // 3
                max_output = min(
                    self._detected_output_limit,
                    max(4000, int(estimated_input_tokens * 1.3))
                )
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": max_output,
                }
            )
            
            # Verify we didn't hit the limit (which would cause truncation)
            response_tokens = len(response.text) // 3
            if response_tokens >= max_output * 0.95:
                logger.warning(
                    f"⚠️  Chunk may be truncated! Output: {response_tokens:,} tokens, "
                    f"limit: {max_output:,}. Detected limit: {self._detected_output_limit:,}"
                )
            
            return response.text
        except Exception as e:
            error_msg = str(e)
            # Détection d'erreurs de quota spécifiques
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                raise AIProofreaderError(
                    f"Quota Gemini atteint (plan gratuit: 15 RPM, 1500/jour). "
                    f"Erreur: {error_msg}"
                ) from e
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
    
    def _test_with_limit(self, text: str, limit: int) -> str:
        """Test with specific output limit."""
        prompt = self._build_prompt(text)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an expert French proofreader."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=limit
        )
        return response.choices[0].message.content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with OpenAI."""
        try:
            prompt = self._build_prompt(text)
            
            # Use detected limit or fallback
            if self._detected_output_limit is None:
                max_tokens = min(4096, len(text) + 1000)
            else:
                estimated_input_tokens = len(text) // 3
                max_tokens = min(
                    self._detected_output_limit,
                    max(2000, int(estimated_input_tokens * 1.3))
                )
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert French proofreader."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            # Check for truncation
            if response.choices[0].finish_reason == "length":
                logger.warning(
                    f"⚠️  OpenAI response truncated! Detected limit: {self._detected_output_limit:,} tokens"
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
    
    def _test_with_limit(self, text: str, limit: int) -> str:
        """Test with specific output limit."""
        prompt = self._build_prompt(text)
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=limit,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def proofread_chunk(self, text: str) -> str:
        """Proofread with Claude."""
        try:
            prompt = self._build_prompt(text)
            
            # Use detected limit or fallback
            if self._detected_output_limit is None:
                max_tokens = min(4096, len(text) + 1000)
            else:
                estimated_input_tokens = len(text) // 3
                max_tokens = min(
                    self._detected_output_limit,
                    max(2000, int(estimated_input_tokens * 1.3))
                )
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Check for truncation
            if response.stop_reason == "max_tokens":
                logger.warning(
                    f"⚠️  Claude response truncated! Detected limit: {self._detected_output_limit:,} tokens"
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
