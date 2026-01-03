# Guide de Dépannage

## 🔍 Table des Matières

1. [Problèmes d'Installation](#problèmes-dinstallation)
2. [Erreurs OCR](#erreurs-ocr)
3. [Validation des Chapitres](#validation-des-chapitres)
4. [Correction IA et Quota](#correction-ia-et-quota)
5. [Performance et Mémoire](#performance-et-mémoire)

---

## Problèmes d'Installation

### Tesseract non trouvé
```
Error: tesseract not found
```

**Solution :**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Vérifier l'installation
tesseract --version
```

### Variable TESSDATA_PREFIX
```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

### page-dewarp absent

L'outil fonctionne sans, mais la qualité d'image peut être moindre.  
Installation : https://github.com/mzucker/page_dewarp

---

## Erreurs OCR

### Faible confiance OCR

Si beaucoup de texte est rejeté :
```yaml
ocr:
  confidence_threshold: 60  # Réduire de 80 à 60
```

### Langue incorrecte

```bash
# Vérifier les langues disponibles
tesseract --list-langs

# Installer une langue
sudo apt-get install tesseract-ocr-eng  # Anglais
sudo apt-get install tesseract-ocr-deu  # Allemand
```

---

## Validation des Chapitres

### ⚠️ Erreurs de Numérotation OCR

OCR (Tesseract) peut mal lire les numéros de chapitres :
- `17. TITRE` → `47. TITRE` (1 non reconnu)
- `20. AUTRE` → `0. AUTRE` (2 manquant)

### Validation Automatique

La validation s'exécute automatiquement dans le pipeline :
```
✅ Validation runs after OCR, before AI proofreading
❌ Stops if errors found
```

**Exemple d'erreur détectée :**
```
❌ ERREUR: Saut de numérotation détecté
   Chapitre 16 → Chapitre 47 (ligne 1234)
   Chapitre attendu: 17
   💡 Correction suggérée: Remplacer "47" par "17"
```

### Validation Manuelle

```bash
python3 validate_chapters.py ../livre_tesseract.txt
```

### Correction des Erreurs

1. **Ouvrir le fichier** : `vim livre_tesseract.txt`
2. **Aller à la ligne** : `:1234` (numéro affiché dans l'erreur)
3. **Corriger** : `47. Titre` → `17. Titre`
4. **Sauvegarder** : `:wq`
5. **Relancer** : `./pdf2epub.sh -i livre.pdf --generate-epub-only`

### Pourquoi c'est Important

- ✅ Évite le gaspillage d'appels API IA sur du texte cassé
- ✅ Assure un EPUB propre avec navigation correcte
- ✅ Économise 20+ minutes de retraitement

**Documentation complète** : [CHAPTER_VALIDATION.md](CHAPTER_VALIDATION.md)

---

## Correction IA et Quota

### 🆓 Plan Gratuit Gemini

**Limites :**
- 15 requêtes/minute (RPM)
- 1 500 requêtes/jour
- 1 million tokens/jour

**Configuration par défaut :**
```yaml
ai_proofreading:
  free_tier: true
  delay_between_chunks: 5  # 5s = 12 req/min < 15 RPM
  chunk_size: 22000        # Auto-ajusté
```

**Temps de traitement** : ~1min 30s pour un livre de 380k chars (17 chunks)

### Quota Atteint

```
❌ QUOTA LIMITE ATTEINTE !
   Plan gratuit Gemini : 15 requêtes/minute, 1500 requêtes/jour
   Chunks traités : 12/17
   💡 Solution : Attendre 24h ou passer au plan payant
   📁 Progression sauvegardée dans : gemini_checkpoint_12_of_17.txt
```

**Solutions :**
1. **Attendre 24h** : Le quota se réinitialise
2. **Plan payant** : 1000+ RPM, ~$0.01-$0.05/livre
3. **Réduire les chunks** : Augmenter `chunk_size` à 30000 (⚠️ risqué)

### Détection Automatique des Limites

Le système détecte automatiquement votre limite :
```
🔍 Detecting AI output token limit...
⚠️  Could not detect limit (all tests failed), using conservative default: 8,000 tokens
📊 Auto-configured chunk_size: 22,400 chars
Split text into 17 chunks
```

**Avantages :**
- ✅ S'adapte automatiquement si vous upgradez
- ✅ Prévient la troncature (perte de chapitres)
- ✅ Optimise l'usage API selon vos limites

**Documentation technique** : [AI_AUTO_DETECTION.md](AI_AUTO_DETECTION.md)

### Texte Tronqué (Chapitres Manquants)

Si des chapitres manquent après correction IA :

```bash
# Vérifier l'intégrité
python3 verify_ai.py livre

# Si troncature détectée
# → La détection automatique devrait empêcher cela maintenant
# Mais vous pouvez forcer des chunks plus petits :
vim pdf2epub.yaml
# chunk_size: 20000

# Relancer la correction IA uniquement
./pdf2epub.sh -i livre.pdf --generate-epub-only --ai-proofread
```

### RetryError / Erreurs Réseau

```
RetryError[<Future raised AIProofreaderError>]
```

**Causes possibles :**
- Rate limiting (trop rapide) → augmentez `delay_between_chunks`
- Quota quotidien atteint → attendez 24h
- Problème réseau → vérifiez connectivité
- Clé API invalide → vérifiez `~/gemini.key`

**Test manuel :**
```bash
python3 test_ai_detection.py
```

### Clé API Non Trouvée

```
AIProofreaderError: API key not configured for gemini
```

**Solution :**
```bash
# Créer ~/gemini.key
echo "votre-clé-api" > ~/gemini.key
chmod 600 ~/gemini.key

# Ou variable d'environnement
export GEMINI_API_KEY="votre-clé-api"
```

---

## Performance et Mémoire

### Out of Memory (OOM)

Pour les très gros livres (500+ pages) :

```bash
# Réduire le nombre de workers parallèles
./pdf2epub.sh -i gros-livre.pdf --max-workers 2

# Ou en config
vim pdf2epub.yaml
# performance:
#   max_workers: 2
```

### Traitement Lent

```yaml
performance:
  parallel_processing: true  # Activer le parallélisme
  max_workers: null          # Auto (nombre de CPUs)
```

### Disque Plein

Les fichiers temporaires sont dans `./tmp/` :

```bash
# Nettoyer manuellement si besoin
rm -rf tmp/

# Ou configurer auto-cleanup
vim pdf2epub.yaml
# output:
#   keep_intermediate_files: false
```

### Ordre des Pages Mélangé

Le traitement parallèle préserve l'ordre (indexation correcte).  
Si problème détecté :

```yaml
performance:
  parallel_processing: false  # Désactiver
```

---

## Documentation Complète

- **README.md** - Guide principal
- **QUICKSTART.md** - Démarrage rapide
- **GEMINI_FREE_TIER.md** - Plan gratuit détaillé
- **AI_AUTO_DETECTION.md** - Détection automatique
- **CHAPTER_VALIDATION.md** - Validation chapitres
- **CHANGELOG.md** - Historique des versions
