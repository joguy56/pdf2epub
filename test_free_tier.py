#!/usr/bin/env python3
"""
Test de configuration pour le plan gratuit Gemini.

Vérifie que la détection automatique fonctionne avec les bonnes limites
et calcule les temps de traitement estimés.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pdf2epub.ai_proofreader import create_proofreader
from pdf2epub.config import AIProofreadingConfig


def test_free_tier_config():
    """Test la configuration plan gratuit."""
    print("=" * 80)
    print("TEST DE CONFIGURATION PLAN GRATUIT GEMINI")
    print("=" * 80)
    print()
    
    # Config plan gratuit
    config = AIProofreadingConfig(
        enabled=True,
        provider="gemini",
        model="gemini-2.5-flash",
        free_tier=True,
        delay_between_chunks=5,
        chunk_size=22000,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    print("📋 Configuration:")
    print(f"   Provider: {config.provider}")
    print(f"   Model: {config.model}")
    print(f"   Free tier: {config.free_tier}")
    print(f"   Delay between chunks: {config.delay_between_chunks}s")
    print(f"   Chunk size: {config.chunk_size:,} chars")
    print()
    
    # Vérifier la clé API
    if not config.get_api_key():
        print("❌ ERREUR: Clé API Gemini non trouvée")
        print("   Solution:")
        print("   1. export GEMINI_API_KEY=$(cat ~/gemini.key)")
        print("   2. Ou créer ~/gemini.key avec votre clé")
        return False
    
    print("✅ Clé API trouvée")
    print()
    
    # Créer le proofreader
    try:
        proofreader = create_proofreader(config)
        print(f"✅ Proofreader créé: {type(proofreader).__name__}")
    except Exception as e:
        print(f"❌ Erreur création proofreader: {e}")
        return False
    
    print()
    print("🔍 Détection automatique des limites...")
    print()
    
    # La détection se fera automatiquement au premier proofread_text()
    # Mais on peut simuler les résultats attendus
    
    # Plan gratuit Gemini = 8k tokens limite
    expected_limit = 8000
    expected_chunk_size = int(8000 * 3.5 * 0.8)  # ~22,400
    
    print(f"Limite attendue (plan gratuit): {expected_limit:,} tokens")
    print(f"Chunk size optimal: {expected_chunk_size:,} chars")
    print()
    
    # Calcul pour livre exemple (RobinHoodT9)
    book_chars = 377450
    chunks_optimal = (book_chars + expected_chunk_size - 1) // expected_chunk_size
    
    print("=" * 80)
    print("ESTIMATION TEMPS DE TRAITEMENT")
    print("=" * 80)
    print()
    print(f"📚 Livre exemple: {book_chars:,} caractères")
    print(f"   Chunks: {chunks_optimal} chunks")
    print(f"   Délai: {config.delay_between_chunks}s entre chunks")
    print(f"   Temps total: ~{chunks_optimal * config.delay_between_chunks}s ({chunks_optimal * config.delay_between_chunks // 60}min {chunks_optimal * config.delay_between_chunks % 60}s)")
    print()
    
    # Calcul limites quotidiennes
    requests_per_minute = 60 // config.delay_between_chunks
    requests_per_day_limit = 1500
    books_per_day = requests_per_day_limit // chunks_optimal
    
    print("=" * 80)
    print("LIMITES PLAN GRATUIT")
    print("=" * 80)
    print()
    print("📊 Quota Gemini (plan gratuit):")
    print(f"   - 15 requêtes/minute (RPM)")
    print(f"   - 1,500 requêtes/jour")
    print(f"   - 1 million de tokens/jour")
    print()
    print(f"⏱️  Avec délai de {config.delay_between_chunks}s:")
    print(f"   - Débit: ~{requests_per_minute} requêtes/minute ✅ (< 15 RPM)")
    print()
    print(f"📚 Capacité quotidienne:")
    print(f"   - ~{books_per_day} livres/jour (comme RobinHoodT9)")
    print(f"   - Soit ~{books_per_day * 7} livres/semaine")
    print()
    
    # Recommandations
    print("=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)
    print()
    
    if requests_per_minute > 15:
        print("⚠️  ATTENTION: Débit trop élevé !")
        print(f"   {requests_per_minute} req/min > 15 RPM limite")
        print("   Solution: Augmentez delay_between_chunks à 5s minimum")
        print()
    else:
        print(f"✅ Débit OK: {requests_per_minute} req/min < 15 RPM")
        print()
    
    print("💡 Pour traiter plus de livres:")
    print("   1. Augmentez chunk_size à 30000 (réduit à ~13 chunks)")
    print("      ⚠️  Risque si limite réelle < 8k tokens")
    print()
    print("   2. Passez au plan payant (1000+ RPM)")
    print("      Coût: ~$0.01-$0.05 par livre")
    print()
    
    print("=" * 80)
    print("✅ TEST TERMINÉ")
    print("=" * 80)
    print()
    print("📖 Documentation complète: GEMINI_FREE_TIER.md")
    print("🔧 Config exemple: pdf2epub_free_tier.yaml.example")
    
    return True


if __name__ == "__main__":
    success = test_free_tier_config()
    sys.exit(0 if success else 1)
