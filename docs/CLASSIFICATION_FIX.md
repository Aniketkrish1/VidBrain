# Classification Fix for Off-Topic Results

## Problem
When querying for "Tim sort", the video included clips about heap sort at the beginning because:

1. **Vector search returned marginally related content** - The sentence "Almost exactly like insertion sort, but does not do a linear scan of unsorted numbers" (which describes heap sort) scored 0.44 because it mentions "insertion sort" and Timsort uses insertion sort internally.

2. **Classifier was too lenient** - The LLM classifier marked it as "important" because it saw mentions of sorting concepts related to Timsort, even though the sentence itself was describing a completely different algorithm (heap sort).

## Solution

Updated `utils/summarizer.py` with stricter classification:

### 1. Improved Classification Prompt
- Now requires EXPLICIT mention of the queried topic by name
- Clarifies that similar/related algorithms should be rejected
- Provides concrete examples to guide the LLM

### 2. Added Keyword Safety Check
```python
# Fallback: check if topic keywords appear in text
if topic_lower in text_lower:
    return (True, "fallback-keyword-match")
else:
    return (False, "fallback-no-keyword-match")
```

### 3. Post-Classification Validation
```python
# If LLM says "important" but topic name not in text, double-check
if important and topic_keywords not in text_normalized:
    # Reject if reason is vague ("similar", "related", "like")
    if "similar" in reason.lower() or "related" in reason.lower():
        return (False, f"rejected-vague-match: {reason}")
```

## Expected Behavior Now

When you query for "Tim sort":
- ✅ Keeps: "Timsort was created in 2002..."
- ✅ Keeps: "Timsort separates an array into small subarrays..."
- ✅ Keeps: "Also, the inventor of Timsort named it after himself..."
- ❌ Rejects: "Almost exactly like insertion sort..." (heap sort description)
- ❌ Rejects: "BubbleSort is one of the most popular..." (different algorithm)

## Testing

Run your pipeline again with the same query:
```bash
python main.py
# Enter query: Tim sort
```

You should now see only clusters that explicitly discuss Timsort, without the heap sort introduction.
