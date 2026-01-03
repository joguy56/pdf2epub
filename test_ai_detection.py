#!/usr/bin/env python3
"""Test AI output limit detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf2epub.config import AIProofreadingConfig
from pdf2epub.ai_proofreader import create_proofreader


def test_detection():
    """Test automatic limit detection."""
    print("=" * 80)
    print("AI OUTPUT LIMIT DETECTION TEST")
    print("=" * 80)
    
    # Create config
    config = AIProofreadingConfig(
        enabled=True,
        provider="gemini",
        model="gemini-2.5-flash",
        chunk_size=50000  # Will be auto-adjusted
    )
    
    try:
        print("\n1️⃣  Creating AI proofreader...")
        proofreader = create_proofreader(config)
        
        print("\n2️⃣  Triggering limit detection...")
        print("   (This will send a small test to the AI)\n")
        
        # Trigger detection by creating test text
        test_text = "Test" * 100
        
        # Force detection
        if proofreader._detected_output_limit is None:
            proofreader._detect_output_limit()
        
        print("\n" + "=" * 80)
        print("DETECTION RESULTS")
        print("=" * 80)
        print(f"✅ Detected output limit: {proofreader._detected_output_limit:,} tokens")
        print(f"📊 Optimal chunk size: {proofreader._optimal_chunk_size:,} chars")
        print(f"📝 Config chunk size: {config.chunk_size:,} chars")
        
        if proofreader._optimal_chunk_size < config.chunk_size:
            print(f"\n💡 System will use {proofreader._optimal_chunk_size:,} chars/chunk")
            print(f"   (Config value {config.chunk_size:,} would cause truncation)")
        else:
            print(f"\n✅ Config value {config.chunk_size:,} is safe")
        
        # Calculate example chunking
        example_size = 377450  # RobinHoodT9 size
        chunks_optimal = (example_size // proofreader._optimal_chunk_size) + 1
        chunks_config = (example_size // config.chunk_size) + 1
        
        print(f"\n📚 For a {example_size:,} char book:")
        print(f"   - Optimal: {chunks_optimal} chunks")
        print(f"   - Config: {chunks_config} chunks")
        print(f"   - Saved: {chunks_config - chunks_optimal} API calls")
        
        print("\n" + "=" * 80)
        print("✅ TEST PASSED - Automatic detection works!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_detection()
    sys.exit(0 if success else 1)
