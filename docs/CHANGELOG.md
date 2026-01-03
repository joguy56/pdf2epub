# Changelog

## [2.0.0] - 2026-01-03

### ✨ Nouvelles Fonctionnalités

#### 🤖 Correction IA avec Détection Automatique
- **Support multi-providers** : Gemini, OpenAI, Claude
- **Détection automatique des limites** : Le système détecte la limite de tokens de sortie de votre API et ajuste automatiquement la taille des chunks
- **Fallback intelligent** : Si la détection échoue, utilise des valeurs conservatives sûres (8000 tokens pour Gemini)
- **Plan gratuit optimisé** : Configuration automatique pour respecter les limites du plan gratuit Gemini (15 RPM, 1500/jour)
- **Gestion de quota** : Détection et sauvegarde automatique de checkpoint quand quota atteint

#### 📖 Validation de Chapitres
- **Détection d'erreurs OCR** : Identifie automatiquement les numéros de chapitres mal reconnus
- **Validation intégrée** : Lance la validation avant le traitement IA pour éviter le gaspillage d'API calls
- **Suggestions de correction** : Affiche les erreurs avec numéros de ligne et suggestions

#### ⚡ Performance
- **Traitement parallèle** : Multiprocessing pour OCR plus rapide
- **Checkpoints automatiques** : Reprise sur erreur sans perte de progression
- **Gestion mémoire optimisée** : Support de gros livres (500+ pages)

### 🔧 Améliorations

#### Configuration
- **Fichiers YAML** : Configuration hiérarchique avec valeurs par défaut
- **Variables d'environnement** : Support pour clés API et paramètres sensibles
- **Modes de traitement** : `--pdf-only`, `--ocr-only`, `--generate-epub-only`, `--ai-proofread`
- **Mode wizard** : Assistant interactif pour débutants

#### Code
- **Architecture moderne** : Séparation claire des responsabilités (PDF → OCR → Text → AI → EPUB)
- **Type hints** : Python 3.10+ avec annotations complètes
- **Gestion d'erreurs** : Exceptions typées et messages clairs
- **Logging** : Logs colorés avec niveaux de détail

### 🐛 Corrections de Bugs

- **Troncature IA** : Résolution du problème de perte de chapitres (détection automatique des limites)
- **Ordre des pages** : Correction du traitement parallèle qui pouvait mélanger les pages
- **Numérotation chapitres** : Validation OCR pour éviter les erreurs de reconnaissance
- **Gestion mémoire** : Optimisation pour éviter les OOM sur gros fichiers
- **Encodage** : Support UTF-8 complet pour caractères spéciaux et accents

### 📚 Documentation

- **README complet** : Guide utilisateur avec exemples
- **QUICKSTART** : Démarrage rapide en 5 minutes
- **GEMINI_FREE_TIER** : Guide détaillé du plan gratuit
- **AI_AUTO_DETECTION** : Explication technique de la détection automatique
- **CHAPTER_VALIDATION** : Guide de validation des chapitres

### 🧪 Tests

- **Tests unitaires** : Coverage des fonctions critiques
- **Tests d'intégration** : Validation du pipeline complet
- **Tests de régression** : Vérification automatique des résultats

### ⚙️ Configuration Plan Gratuit Gemini

Par défaut, le système est optimisé pour le plan gratuit :
- `free_tier: true` - Active les délais adaptés
- `delay_between_chunks: 5` - 5 secondes entre chunks (12 req/min < 15 RPM)
- `chunk_size: 22000` - Taille sûre pour limite 8k tokens

**Temps de traitement** : ~1min 30s pour un livre de 380k caractères (17 chunks)  
**Capacité quotidienne** : ~88 livres/jour maximum

---

## [1.0.0] - Prototype Original

### Fonctionnalités Initiales
- Conversion PDF → EPUB avec OCR Tesseract
- Détection basique de chapitres
- Post-traitement du texte (hyphénation, dialogues)
- Interface en ligne de commande

### Limitations
- Code monolithique difficile à maintenir
- Pas de gestion d'erreurs robuste
- Configuration en dur dans le code
- Pas de tests automatisés
