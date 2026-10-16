# Initial Report and Proposed Fix

**Target Issue:** [langchain-ai/langchain#41259](https://github.com/langchain-ai/langchain/issues/41259)  
**Package:** `langchain-core` (`libs/core/langchain_core/messages/utils.py`)  
**Branch:** `jadibdev:fix/trim-messages-last-allow-partial-over-budget`  
**Commit:** `6fb74a758`  
**Date:** October 16, 2026  

---

## 1. Executive Summary

This document provides a comprehensive technical overview of **Issue #41259** in `langchain-core`, our diagnosis of the underlying defect, what we set out to accomplish, and the verified implementation developed on branch `fix/trim-messages-last-allow-partial-over-budget`.

### The Core Defect
When `trim_messages()` is invoked with `strategy="last"` and `allow_partial=True`, it frequently returns truncated messages that **exceed the user's `max_tokens` budget**. In production LLM workflows (such as LangGraph agents and conversational chains), this defect causes downstream LLM API calls to unexpectedly fail with context-length or token-budget exceeded exceptions.

### The Root Cause
In `_first_max_tokens()` (`libs/core/langchain_core/messages/utils.py`), candidate text lines and content blocks are evaluated by `token_counter` in **reversed order** during binary search, before being re-reversed back to forward order prior to returning. Because lines and blocks lack trailing newlines at boundaries, measuring joined text in reverse order produces token-boundary fusion under BPE/subword tokenizers (e.g., `tiktoken`). This artificially deflates the measured token count during search, causing the search to accept candidates that exceed `max_tokens` once restored to forward order.

### The Solution
Eliminate candidate reversing entirely. Evaluate candidate splits and blocks directly in their **natural forward reading order** using tail slicing (`split_texts[-mid:]` and `content[-k:]`). This guarantees mathematical equivalence between the text measured during search and the text returned to the caller.

---

## 2. Issue Background & Context

### What `trim_messages` Does
`trim_messages` is the standard LangChain utility for managing conversation context windows. Its goal is to take a sequence of `BaseMessage` objects and trim them to fit within a specified `max_tokens` limit according to a user-supplied or model-derived `token_counter`.

When `strategy="last"` is selected:
* The user wants to preserve the most recent messages (e.g. latest user prompts and assistant outputs).
* If `allow_partial=True` is enabled, the utility does not simply drop entire messages; it attempts to truncate the oldest retained message down to its trailing lines or content blocks to maximize context window utilization.

### The Bug in Action
Consider a message whose content consists of lines:
```text
"a\nb\nc\nd\ne\nf\ng"
```
When trimmed with `strategy="last"`, `allow_partial=True`, and a `max_tokens=5` budget using a BPE tokenizer (or tiktoken `cl100k_base`):
* **Expected:** The function returns content whose measured token cost is $\le 5$ (e.g., `'g'`, cost = 3).
* **Actual Output:** The function returns `'f\ng'`, which costs **6 tokens** ($\gt 5$). The message exceeds budget.

---

## 3. Deep Root-Cause Analysis

Examining `_first_max_tokens` in `libs/core/langchain_core/messages/utils.py` (lines 2026–2088) reveals the sequence leading to this failure:

### A. The Text-Splitting Search Path (Lines 2063–2088)
```python
split_texts = text_splitter(text)
base_message_count = token_counter(messages[:idx])
if partial_strategy == "last":
    split_texts = list(reversed(split_texts))  # <-- Step 1: Reversed

# Binary search
for _ in range(max_iterations):
    ...
    mid = (left + right + 1) // 2
    excluded.content = "".join(split_texts[:mid])  # <-- Step 2: Joined while REVERSED
    if base_message_count + token_counter([excluded]) <= max_tokens:
        left = mid
    ...

if left > 0:
    content_splits = split_texts[:left]
    if partial_strategy == "last":
        content_splits = list(reversed(content_splits))  # <-- Step 3: Re-reversed
    excluded.content = "".join(content_splits)
```

1. **Step 1:** To retain the last `mid` lines, `split_texts` is reversed from `['f\n', 'g']` to `['g', 'f\n']`.
2. **Step 2:** The binary search evaluates `"".join(split_texts[:mid])`. This joins the lines in reverse order:
   $$\text{"g"} + \text{"f\textbackslash n"} = \text{"gf\textbackslash n"}$$
   In standard prose and code, the final line (`"g"`) does not have a trailing newline. When joined in reverse order, `"g"` and `"f"` collide directly at the boundary. Subword tokenizers (Byte-Pair Encoding, WordPiece, tiktoken) encode `"gf"` into fewer tokens than `"f\ng"`.
   - Measured cost of `"gf\n"`: **5 tokens** $\le 5$ (Accepted by binary search).
3. **Step 3:** The search finishes with `left = 2`. The function then calls `reversed(content_splits)` and joins them:
   $$\text{"f\textbackslash n"} + \text{"g"} = \text{"f\textbackslash ng"}$$
   - Actual cost of `"f\ng"`: **6 tokens** ($> 5$).
   - The returned message silently exceeds `max_tokens`!

### B. The Content-Blocks Search Path (Lines 2026–2041)
The multi-block content path (`isinstance(messages[idx].content, list)`) exhibits the identical defect:
* Line 2031: `excluded.content = list(reversed(excluded.content))`
* Line 2034: `token_counter([*messages[:idx], excluded]) <= max_tokens` evaluates `excluded.content` while reversed.
* Line 2040: `excluded.content = list(reversed(excluded.content))` restores forward order before returning.
Any token counter sensitive to block order or concatenated block text experiences the same discrepancy.

---

## 4. Objectives (What We Set Out to Accomplish)

1. **Strict Budget Invariant:** Guarantee that `trim_messages()` never returns a message sequence whose token count exceeds `max_tokens` according to the provided `token_counter`.
2. **Order-Preserving Evaluation:** Ensure that any candidate evaluated during partial trimming is measured in the **exact same string representation and reading order** as the returned output.
3. **Zero API Changes & Full Backward Compatibility:** Keep all function signatures, parameters, and return types strictly unchanged.
4. **Clean Code & Simplicity:** Eliminate redundant reversing and re-reversing passes in favor of direct slice indexing.

---

## 5. Technical Implementation

The fix was applied in [`libs/core/langchain_core/messages/utils.py`](file:///C:/Projects/github-portfolio/langchain/libs/core/langchain_core/messages/utils.py):

### 1. Text-Splitting Path
Instead of reversing `split_texts` at the start and re-reversing at the end, the binary search extracts the candidate using a negative slice (`split_texts[-mid:]`) when `partial_strategy == "last"`:

```python
split_texts = text_splitter(text)
base_message_count = token_counter(messages[:idx])

# Binary search for the maximum number of splits we can include
left, right = 0, len(split_texts)
max_iterations = len(split_texts).bit_length()
for _ in range(max_iterations):
    if left >= right:
        break
    mid = (left + right + 1) // 2
    candidate_splits = (
        split_texts[-mid:]
        if partial_strategy == "last"
        else split_texts[:mid]
    )
    excluded.content = "".join(candidate_splits)
    if base_message_count + token_counter([excluded]) <= max_tokens:
        left = mid
    else:
        right = mid - 1

if left > 0:
    content_splits = (
        split_texts[-left:]
        if partial_strategy == "last"
        else split_texts[:left]
    )
    excluded.content = "".join(content_splits)
    messages = [*messages[:idx], excluded]
    idx += 1
```

### 2. Content-Blocks Path
Similarly, content blocks are sliced in forward order:

```python
if isinstance(messages[idx].content, list):
    excluded = messages[idx].model_copy(deep=True)
    copied = True
    num_block = len(excluded.content)
    for k in range(num_block - 1, 0, -1):
        candidate_blocks = (
            excluded.content[-k:]
            if partial_strategy == "last"
            else excluded.content[:k]
        )
        excluded.content = candidate_blocks
        if token_counter([*messages[:idx], excluded]) <= max_tokens:
            messages = [*messages[:idx], excluded]
            idx += 1
            included_partial = True
            break
```

---

## 6. Verification & Test Coverage

Two regression test cases were added to [`libs/core/tests/unit_tests/messages/test_utils.py`](file:///C:/Projects/github-portfolio/langchain/libs/core/tests/unit_tests/messages/test_utils.py):

### Test 1: `test_trim_messages_last_allow_partial_respects_token_budget_order`
* **Purpose:** Verifies that when a token counter exhibits boundary-order sensitivity (such as subword token fusion across unpunctuated lines), `trim_messages` evaluates candidate splits in forward order and strictly respects `max_tokens`.
* **Behavior:**
  - With old logic: Evaluated reversed `'gf\n'` (cost 4 $\le 5$), returned `'f\ng'` (cost 6 $> 5$) $\rightarrow$ **FAILED**.
  - With new logic: Evaluated forward `'f\ng'` (cost 6 $> 5$), correctly trimmed to `'g'` (cost 3 $\le 5$) $\rightarrow$ **PASSED**.

### Test 2: `test_trim_messages_last_allow_partial_blocks_respects_token_budget_order`
* **Purpose:** Verifies that multi-block messages evaluated with `partial_strategy="last"` slice and measure blocks in forward reading order without token fusion leakage.
* **Behavior:** Slices `[{"type": "text", "text": "b2"}]` forwardly and verifies `block_counter(actual) <= max_tokens` $\rightarrow$ **PASSED**.

---

## 7. Status & Next Steps

1. **Local Repository:** Cleanly committed to [`C:\Projects\github-portfolio\langchain`](file:///C:/Projects/github-portfolio/langchain).
2. **Remote Branch:** Pushed to [`jadibdev:fix/trim-messages-last-allow-partial-over-budget`](https://github.com/jadibdev/langchain/tree/fix/trim-messages-last-allow-partial-over-budget) (commit `6fb74a758`).
3. **Contribution Workflow:** In accordance with LangChain's automated contribution rules, the technical proposal will be posted on [Issue #41259](https://github.com/langchain-ai/langchain/issues/41259) to obtain maintainer assignment prior to opening the formal PR.
