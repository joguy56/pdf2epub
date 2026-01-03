# Gemini Free Tier Configuration

## Free Tier Limits

Gemini AI free tier has strict limits:

- **15 requests per minute** (RPM)
- **1,500 requests per day**
- **1 million tokens per day**
- **Output limit: 8,192 tokens** per request

## Recommended Configuration

### In `pdf2epub.yaml`:

```yaml
ai_proofreading:
  enabled: true
  provider: gemini
  model: gemini-2.5-flash  # Or gemini-1.5-flash
  free_tier: true          # ✅ IMPORTANT: Enables delays for free tier
  delay_between_chunks: 5  # 5 seconds = max 12 chunks/minute (under 15 RPM)
  chunk_size: 22000        # Safe size for 8k token limit
  max_retries: 3
  timeout: 60
```

### Delay Calculation

To stay under the **15 RPM** limit:
- **5 seconds** between chunks = 12 chunks/minute ✅
- **4 seconds** between chunks = 15 chunks/minute ⚠️ (risky)
- **3 seconds** between chunks = 20 chunks/minute ❌ (too fast)

## Estimated Processing Times

For a book of **~380,000 characters** (like RobinHoodT9):

| chunk_size | Chunk count | Delay | Total time |
|------------|-------------|-------|------------|
| 22,000     | 17 chunks   | 5s    | **~1min 30s** |
| 15,000     | 25 chunks   | 5s    | **~2min 10s** |
| 10,000     | 38 chunks   | 5s    | **~3min 15s** |

## Daily Quota Management

### How many books per day?

With **1,500 requests/day**:
- 17-chunk book: **~88 books/day max**
- 25-chunk book: **~60 books/day max**

### What happens when quota is reached?

The code automatically detects quota errors:

```
 QUOTA LIMIT REACHED!
   Gemini free tier: 15 requests/minute, 1500 requests/day
   Chunks processed: 12/17
   💡 Solution: Wait 24h or upgrade to paid plan
   📁 Progress saved in: gemini_checkpoint_12_of_17.txt
```

Processing stops and saves a **checkpoint** with already processed text.

### Resume after quota

Currently, you must:
1. Wait 24h for quota to reset
2. Restart the full processing (code will retry failed chunks)

## Possible Optimizations

### 1. Reduce chunk count

Increase `chunk_size` for fewer requests:
```yaml
chunk_size: 30000  # Reduces to ~13 chunks instead of 17
```

 **Risk**: If your actual limit is < 8000 tokens, you risk truncation.

### 2. Process only certain chapters

Add filters to correct only chapters with many errors.

### 3. Upgrade to paid plan

**Pay-as-you-go**:
- 1000+ RPM (instead of 15)
- No strict daily limit
- Cost: ~$0.075 per million input tokens, ~$0.30 per million output tokens

For a 380k char book (~127k tokens):
- Estimated cost: **$0.01 - $0.05** per book

## Commands

### Processing with free tier (default)

```bash
./pdf2epub.sh -i book.pdf -a "Author" -t "Title" -l fra --ai-proofread
```

### Processing with paid plan (if you upgrade)

Modify `pdf2epub.yaml`:
```yaml
ai_proofreading:
  free_tier: false        # Disable strict limits
  delay_between_chunks: 1 # Faster
```

## Automatic Detection

The code automatically detects the output limit:
- Tests with 4k, 8k, 16k, 32k tokens
- If all tests fail → fallback **8,000 tokens** (free tier)
- Calculates optimal `chunk_size`: `8000 × 3.5 × 0.8 = 22,400 chars`

This detection works even when network tests fail thanks to intelligent fallback.

## Common Errors

### `RESOURCE_EXHAUSTED`
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```
**Solution**: Wait 24h or upgrade to paid plan.

### `RetryError`
```
RetryError[<Future raised AIProofreaderError>]
```
**Possible causes**:
- RPM quota reached (too many requests/minute) → increase `delay_between_chunks`
- Daily quota reached → wait 24h
- Network problem → retry
- Invalid API key → check `~/gemini.key`

### All chunks fail
**Solution**:
1. Check your API key: `cat ~/gemini.key`
2. Test manually: `python3 test_ai_detection.py`
3. Increase `delay_between_chunks` to 10 seconds

## Monitoring

Watch logs for delays:
```
INFO - Waiting 5s before next chunk to avoid rate limiting (free tier: 15 RPM)...
```

If you see many errors, increase the delay.
