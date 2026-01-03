# Configuration pour le Plan Gratuit Gemini

## Limites du Plan Gratuit

Le plan gratuit de Gemini AI a des limites strictes :

- **15 requêtes par minute** (RPM)
- **1 500 requêtes par jour**
- **1 million de tokens par jour**
- **Limite de sortie : 8 192 tokens** par requête

## Configuration Recommandée

### Dans `pdf2epub.yaml` :

```yaml
ai_proofreading:
  enabled: true
  provider: gemini
  model: gemini-2.5-flash  # Ou gemini-1.5-flash
  free_tier: true          # ✅ IMPORTANT : Active les délais pour plan gratuit
  delay_between_chunks: 5  # 5 secondes = max 12 chunks/minute (sous 15 RPM)
  chunk_size: 22000        # Taille sûre pour limite 8k tokens
  max_retries: 3
  timeout: 60
```

### Calcul des Délais

Pour rester sous la limite de **15 RPM** :
- **5 secondes** entre chunks = 12 chunks/minute ✅
- **4 secondes** entre chunks = 15 chunks/minute ⚠️ (risqué)
- **3 secondes** entre chunks = 20 chunks/minute ❌ (trop rapide)

## Temps de Traitement Estimés

Pour un livre de **~380 000 caractères** (comme RobinHoodT9) :

| chunk_size | Nombre chunks | Délai | Temps total |
|------------|---------------|-------|-------------|
| 22 000     | 17 chunks     | 5s    | **~1min 30s** |
| 15 000     | 25 chunks     | 5s    | **~2min 10s** |
| 10 000     | 38 chunks     | 5s    | **~3min 15s** |

## Gestion du Quota Quotidien

### Combien de livres par jour ?

Avec **1 500 requêtes/jour** :
- Livre de 17 chunks : **~88 livres/jour max**
- Livre de 25 chunks : **~60 livres/jour max**

### Que se passe-t-il si quota atteint ?

Le code détecte automatiquement les erreurs de quota :

```
❌ QUOTA LIMITE ATTEINTE !
   Plan gratuit Gemini : 15 requêtes/minute, 1500 requêtes/jour
   Chunks traités : 12/17
   💡 Solution : Attendre 24h ou passer au plan payant
   📁 Progression sauvegardée dans : gemini_checkpoint_12_of_17.txt
```

Le traitement s'arrête et sauvegarde un **checkpoint** avec le texte déjà traité.

### Reprendre après quota

Actuellement, vous devez :
1. Attendre 24h pour que le quota se réinitialise
2. Relancer le traitement complet (le code réessaiera les chunks échoués)

## Optimisations Possibles

### 1. Réduire le nombre de chunks

Augmentez `chunk_size` pour moins de requêtes :
```yaml
chunk_size: 30000  # Réduit à ~13 chunks au lieu de 17
```

⚠️ **Risque** : Si votre limite réelle est < 8000 tokens, vous risquez la troncature.

### 2. Traiter seulement certains chapitres

Ajoutez des filtres pour ne corriger que les chapitres avec beaucoup d'erreurs.

### 3. Passer au plan payant

**Pay-as-you-go** :
- 1000+ RPM (au lieu de 15)
- Pas de limite quotidienne stricte
- Coût : ~$0.075 par million de tokens d'entrée, ~$0.30 par million de tokens de sortie

Pour un livre de 380k chars (~127k tokens) :
- Coût estimé : **$0.01 - $0.05** par livre

## Commandes

### Traitement avec plan gratuit (défaut)

```bash
./pdf2epub.sh -i livre.pdf -a "Auteur" -t "Titre" -l fra --ai-proofread
```

### Traitement avec plan payant (si vous upgradez)

Modifiez `pdf2epub.yaml` :
```yaml
ai_proofreading:
  free_tier: false
  delay_between_chunks: 1  # Plus rapide
```

## Détection Automatique

Le code détecte automatiquement la limite de sortie :
- Teste avec 4k, 8k, 16k, 32k tokens
- Si tous les tests échouent → fallback **8 000 tokens** (plan gratuit)
- Calcule `chunk_size` optimal : `8000 × 3.5 × 0.8 = 22 400 chars`

Cette détection fonctionne même quand les tests réseau échouent grâce au fallback intelligent.

## Erreurs Courantes

### `RESOURCE_EXHAUSTED`
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```
**Solution** : Attendez 24h ou passez au plan payant.

### `RetryError`
```
RetryError[<Future raised AIProofreaderError>]
```
**Causes possibles** :
- Quota RPM atteint (trop de requêtes/minute) → augmentez `delay_between_chunks`
- Quota quotidien atteint → attendez 24h
- Problème réseau → réessayez

### Tous les chunks échouent
**Solution** :
1. Vérifiez votre clé API : `cat ~/gemini.key`
2. Testez manuellement : `python3 test_ai_detection.py`
3. Augmentez `delay_between_chunks` à 10 secondes

## Monitoring

Surveillez les logs pour voir les délais :
```
INFO - Waiting 5s before next chunk to avoid rate limiting (free tier: 15 RPM)...
```

Si vous voyez beaucoup d'erreurs, augmentez le délai.
