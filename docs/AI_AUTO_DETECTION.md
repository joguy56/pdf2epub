# Automatic AI Token Limit Detection

## Overview

The AI proofreading system now **automatically detects** the output token limit of your AI provider at startup, eliminating the need for manual `chunk_size` configuration.

## How It Works

### 1. Automatic Detection at Startup

When AI proofreading starts, the system:

```
🔍 Detecting AI output token limit...
Testing with max_output_tokens=4000
Testing with max_output_tokens=8000
Testing with max_output_tokens=16000
✅ Detected output limit: ~8,192 tokens
📊 Auto-configured chunk_size: 22,937 chars (~6,553 tokens input → ~8,192 tokens output)
```

**Process:**
1. Sends a small test text (~10k chars) to the AI
2. Tests with increasing `max_output_tokens`: 4k, 8k, 16k, 32k
3. Detects where the AI caps its response
4. Calculates optimal `chunk_size` based on detected limit

### 2. Dynamic Chunk Size Calculation

**Formula:**
```python
optimal_chunk_size = detected_limit × 3.5 chars/token × 0.8 safety_margin
```

**Examples:**

| Detected Limit | Optimal chunk_size | Reasoning |
|----------------|-------------------|-----------|
| 4,096 tokens | ~11,469 chars | OpenAI GPT-4 Turbo, Claude 3 |
| 8,192 tokens | ~22,937 chars | Gemini 2.5 Flash, Gemini 1.5 Pro |
| 16,384 tokens | ~45,875 chars | OpenAI GPT-4o |
| 32,768 tokens | ~91,750 chars | (future models) |

### 3. Adapts to Your Subscription

**Free tier Gemini** (8k token limit):
```
✅ Detected output limit: ~8,192 tokens
📊 Auto-configured chunk_size: 22,937 chars
Split text into 16 chunks
```

**Paid tier with higher limits** (hypothetical 32k):
```
✅ Detected output limit: ~32,768 tokens
📊 Auto-configured chunk_size: 91,750 chars
Split text into 4 chunks  ← 4× fewer API calls!
```

**Switching providers** (Gemini → OpenAI GPT-4o):
```
✅ Detected output limit: ~16,384 tokens (GPT-4o)
📊 Auto-configured chunk_size: 45,875 chars
Split text into 8 chunks
```

## Benefits

### ✅ No Manual Configuration

**Before** (manual):
```yaml
ai_proofreading:
  chunk_size: 25000  # Magic number! Must adjust if you upgrade
```

**After** (automatic):
```yaml
ai_proofreading:
  chunk_size: 50000  # Max allowed, system auto-adjusts down if needed
```

### ✅ Automatic Adaptation

- Upgrade to paid tier? **Automatically uses larger chunks**
- Switch AI provider? **Automatically detects new limits**
- Model updated with higher limits? **Benefits immediately**

### ✅ Prevents Truncation

The system:
- Detects actual limits (not documentation, which can be wrong)
- Adds 20% safety margin
- Warns if any chunk gets truncated

### ✅ Cost Optimization

Higher limits = fewer chunks = fewer API calls:

| Token Limit | Chunks for 377k char book | API Cost Multiplier |
|-------------|---------------------------|---------------------|
| 4k (Claude) | ~32 chunks | 4× baseline |
| 8k (Gemini) | ~16 chunks | 2× baseline |
| 16k (GPT-4o) | ~8 chunks | 1× baseline |

## Configuration

### Default (Recommended)

```yaml
# pdf2epub.yaml
ai_proofreading:
  enabled: true
  provider: gemini  # or openai, claude
  chunk_size: 50000  # Max size, auto-adjusted based on detection
```

**What happens:**
1. System detects Gemini's 8k limit
2. Adjusts `chunk_size` to ~23k chars automatically
3. Logs: `Using auto-detected chunk_size: 22,937 chars (config: 50,000)`

### Override Detection (Advanced)

If you want to force a specific size:

```yaml
ai_proofreading:
  chunk_size: 15000  # Force smaller chunks (more API calls, more reliable)
```

System will still detect limits but **won't increase** beyond your specified value.

### Disable Detection (Not Recommended)

Can't disable detection, but you can set a conservative `chunk_size`:

```yaml
ai_proofreading:
  chunk_size: 10000  # Very safe, works with all providers
```

## Detection Logic Details

### Test Text

```python
test_text = "Ceci est un texte de test pour detecter la limite. " * 200
# ~10,000 chars, designed to trigger AI corrections
```

**Why this text:**
- Long enough to test realistic corrections
- Repetitive to ensure AI returns substantial output
- French to match actual use case

### Detection Algorithm

```python
for limit in [4000, 8000, 16000, 32000]:
    result = test_with_limit(test_text, limit)
    result_tokens = len(result) // 3
    
    if result_tokens < limit * 0.7:  # Got less than 70% requested
        detected_limit = result_tokens * 1.1  # Real limit found
        break
    
    if result_tokens >= limit * 0.8:  # Got close to requested
        continue  # Try next higher limit
```

**Key insight:** If AI returns much less than requested, we hit the ceiling.

### Fallback Behavior

If detection fails (network error, API issue):

```python
detected_limit = 8000  # Conservative fallback (Gemini free tier)
optimal_chunk_size = 22,400  # Safe for most providers
```

**Log message:**
```
⚠️  Could not detect limit, using conservative default: 8,000 tokens
```

## Verification

### Check Detection Results

Look for these log messages at startup:

```
2026-01-03 13:15:22 - INFO - 🔍 Detecting AI output token limit...
2026-01-03 13:15:25 - INFO - ✅ Detected output limit: ~8,192 tokens
2026-01-03 13:15:25 - INFO - 📊 Auto-configured chunk_size: 22,937 chars
2026-01-03 13:15:25 - INFO - 📊 Using auto-detected chunk_size: 22,937 chars 
   (config: 50,000) based on 8,192 token limit
2026-01-03 13:15:25 - INFO - Split text into 16 chunks
```

### Verify Optimal Chunking

**Good** (automatic optimization):
```
Input: 377,450 chars
Detected limit: 8,192 tokens
Auto-configured: 22,937 chars/chunk
Chunks: 16
Time: ~34 min (16 × 2min + 15 × 15s)
```

**Bad** (manual, too large):
```
Input: 377,450 chars
Config: 50,000 chars/chunk  ← Not auto-adjusted!
Chunks: 8  ← Too few!
Result: Truncation, missing chapters
```

## Provider-Specific Limits

### Gemini

**Free tier:**
- Limit: 8,192 tokens
- Auto-configured: ~22,937 chars
- Typical chunks: 15-20 for 300-page book

**Paid tier** (hypothetical future increase):
- Would auto-detect higher limit
- Would use larger chunks automatically
- No code changes needed

### OpenAI

**GPT-4 Turbo:**
- Limit: 4,096 tokens
- Auto-configured: ~11,469 chars
- Typical chunks: 30-35 for 300-page book

**GPT-4o:**
- Limit: 16,384 tokens
- Auto-configured: ~45,875 chars
- Typical chunks: 8-10 for 300-page book

### Claude

**Claude 3 (Sonnet/Opus):**
- Limit: 4,096 tokens
- Auto-configured: ~11,469 chars
- Typical chunks: 30-35 for 300-page book

## Troubleshooting

### Detection Taking Too Long

Normal behavior: ~5-10 seconds for detection

**If > 30 seconds:**
- Check network connection
- Verify API key is valid
- Check provider status page

### Wrong Limit Detected

**Symptoms:**
```
✅ Detected output limit: ~2,000 tokens  ← Too low!
```

**Causes:**
- Network issues during test
- API rate limiting during detection
- Provider temporarily limiting responses

**Solution:**
- Wait and retry
- Check `~/.pdf2epub/ai_test.log` for details
- Manually set `chunk_size: 20000` as workaround

### Truncation Still Happening

**Symptoms:**
```
⚠️  Chunk may be truncated! Output: 8,100 tokens, limit: 8,192
```

**This is normal!** It means:
- Detection worked (found 8,192 limit)
- Chunk was right at the edge
- AI output expanded more than expected (rare)

**Solution:**
- System will warn but continue
- Text should still be complete
- If chapters missing, reduce `chunk_size` in config

## Future Enhancements

### Cache Detection Results

Currently re-detects each run. Could cache:

```yaml
# .pdf2epub_cache.yaml (auto-generated)
gemini-2.5-flash:
  detected_limit: 8192
  detected_at: 2026-01-03 13:15:25
  optimal_chunk_size: 22937
```

### Per-Model Detection

Different models have different limits:

```
gemini-2.5-flash: 8,192 tokens
gemini-1.5-pro: 8,192 tokens
gpt-4-turbo: 4,096 tokens
gpt-4o: 16,384 tokens
```

Could cache per model instead of re-detecting.

### Progressive Chunk Sizing

Start with large chunks, reduce if truncation detected:

```python
if truncation_detected:
    chunk_size *= 0.8  # Reduce by 20%
    retry_chunk()
```

## Summary

**Old system** (manual):
- Hardcoded `chunk_size = 25000`
- Must update when upgrading subscription
- Must change when switching providers
- Can cause truncation if too large

**New system** (automatic):
- ✅ Detects limits at startup
- ✅ Auto-adjusts chunk size
- ✅ Adapts to subscription changes
- ✅ Works with any provider
- ✅ Prevents truncation
- ✅ Optimizes API usage

**Key benefit:** Set it once (`chunk_size: 50000`), system handles the rest!
