# Initial Report and Proposed Fix

**Target Issue:** [langchain-ai/langchain#41259](https://github.com/langchain-ai/langchain/issues/41259)  
**Package:** `langchain-core` (`libs/core/langchain_core/messages/utils.py`)  
**Branch:** `jadibdev/core/trim-messages-last-allow-partial`  
**Date:** October 16, 2026  

---

## 1. Executive Summary

This document provides a comprehensive technical overview of **Issue #41259** in `langchain-core`, our diagnosis of the underlying defect, what we set out to accomplish, and the verified implementation developed to resolve it in full compliance with LangChain's repository guidelines (`AGENTS.md`).

### The Core Defect
When `trim_messages()` is invoked with `strategy="last"` and `allow_partial=True`, it can return truncated messages that **exceed the user's `max_tokens` budget**. In production LLM workflows (such as LangGraph agents and conversational chains), this defect causes downstream LLM API calls to unexpectedly fail with context-length or token-budget exceeded exceptions.

### The Root Cause
In `_first_max_tokens()` (`libs/core/langchain_core/messages/utils.py`), candidate text lines and content blocks are evaluated by `token_counter` in **reversed order** during binary search, before being re-reversed back to forward order prior to returning. Because lines and blocks lack trailing newlines at boundaries, measuring joined text in reverse order produces token-boundary fusion under BPE/subword tokenizers (e.g., `tiktoken` or regex word-boundary counters). This artificially deflates the measured token count during search, causing the search to accept candidates that exceed `max_tokens` once restored to forward order.

### The Solution
Eliminate candidate reversing entirely. Evaluate candidate splits and blocks directly in their **natural forward reading order** using tail slicing (`split_texts[-mid:]` and `original_blocks[-k:]`). This guarantees mathematical equivalence between the text/blocks measured during search and the text/blocks returned to the caller.

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
"a\nb\nc"
```
When trimmed with `strategy="last"`, `allow_partial=True`, and a `max_tokens=5` budget using a BPE tokenizer or a rule-based counter (`3` message overhead $+ \text{tokens}$):
* **Expected:** The function returns content whose measured token cost is $\le 5$ (specifically `'c'`, cost $= 3 + 1 = 4$).
* **Actual Output:** The function returns `'b\nc'`, which costs **6 tokens** ($3 + 3 = 6 > 5$). The returned message exceeds budget.

---

## 3. Deep Root-Cause Analysis

Examining `_first_max_tokens` in `libs/core/langchain_core/messages/utils.py` reveals the exact mechanism:

### A. The Text-Splitting Search Path
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

1. **Step 1:** To retain the last `mid` lines, `split_texts` is reversed from `['b\n', 'c']` to `['c', 'b\n']`.
2. **Step 2:** The binary search evaluates `"".join(split_texts[:mid])`. For `mid = 2`, this joins the lines in reverse order:
   $$\text{"c"} + \text{"b\textbackslash n"} = \text{"cb\textbackslash n"}$$
   In standard text, the final line (`"c"`) lacks a trailing newline. When joined in reverse order, `"c"` and `"b"` collide directly across the boundary. While subword tokenizers (e.g. `tiktoken` BPE) exhibit boundary fusion in production, our unit test employs a lightweight regex counter (`re.findall(r"\w+|\n", ...)`) as an explicit test model for boundary fusion (rather than simulating full BPE tokenization tables). Under this test model, `"cb"` fuses into a single token:
   - Measured cost of `"cb\n"`: $3 \text{ overhead} + 2 \text{ tokens} = \mathbf{5\text{ tokens}} \le 5$ (Accepted by binary search, setting `left = 2`).
3. **Step 3:** The search finishes with `left = 2`. The function calls `reversed(content_splits)` and joins them:
   $$\text{"b\textbackslash n"} + \text{"c"} = \text{"b\textbackslash nc"}$$
   - Actual cost of `"b\nc"`: $3 \text{ overhead} + 3 \text{ tokens} = \mathbf{6\text{ tokens}} > 5$.
   - The returned message silently exceeds `max_tokens`!

### B. The Content-Blocks Search Path
The multi-block content path (`isinstance(messages[idx].content, list)`) exhibited the identical defect:
* `excluded.content = list(reversed(excluded.content))` reversed blocks before evaluation.
* The loop popped items and evaluated concatenated block text while reversed.
* `excluded.content = list(reversed(excluded.content))` restored forward order before returning.
Any token counter sensitive to block order or concatenated block boundaries experienced the same boundary undercounting.

---

## 4. Objectives (What We Set Out to Accomplish)

1. **Order-Invariant Candidate Evaluation:** Ensure that any candidate evaluated during partial trimming (`strategy="last"`) is measured in the **exact same string representation and forward reading order** as the returned output, eliminating reverse-order boundary undercounting.
2. **Robust Slicing Hygiene:** Avoid in-place list mutation during block iteration by capturing `original_blocks = list(excluded.content)` and slicing candidates cleanly.
3. **Boundary Index Safety:** Ensure all negative slices (`split_texts[-mid:]`, `split_texts[-left:]`, `original_blocks[-k:]`) are strictly positive offsets ($\ge 1$), guaranteeing Python never evaluates `seq[-0:]`.
4. **Zero API Changes & Full Backward Compatibility:** Keep all function signatures, parameters, and return types strictly unchanged.

---

## 5. Technical Implementation

The fix was applied in [`libs/core/langchain_core/messages/utils.py`](file:///C:/Projects/github-portfolio/langchain/libs/core/langchain_core/messages/utils.py):

### 1. Content-Blocks Path
Capture `original_blocks` and slice candidates directly in forward order:

```python
if isinstance(messages[idx].content, list):
    excluded = messages[idx].model_copy(deep=True)
    copied = True
    original_blocks = list(excluded.content)
    num_block = len(original_blocks)
    for k in range(num_block - 1, 0, -1):
        excluded.content = (
            original_blocks[-k:]
            if partial_strategy == "last"
            else original_blocks[:k]
        )
        if token_counter([*messages[:idx], excluded]) <= max_tokens:
            messages = [*messages[:idx], excluded]
            idx += 1
            included_partial = True
            break
```

*(Note: When `included_partial` remains `False` after `k = 1`, `excluded.content` holds `[original_blocks[-1]]` for `partial_strategy == "last"` and `[original_blocks[0]]` for `"first"`, perfectly preserving the downstream text fallback when extracting text from the remaining block.)*

### 2. Text-Splitting Path
Extract candidates using forward tail slicing (`split_texts[-mid:]` and `split_texts[-left:]`):

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

### 3. Boundary Safety Proof (`seq[-0:]`)
- In binary search: `left < right` and `left >= 0` $\implies right \ge 1 \implies mid = (left + right + 1) // 2 \ge 1$. Hence, `-mid \le -1`.
- Post-search slice: Guarded by `if left > 0:`, guaranteeing `left >= 1`.
- Content blocks: `range(num_block - 1, 0, -1)` stops at `1`, guaranteeing `k >= 1`.
- Consequently, negative slice indices are always non-zero.

---

## 6. Verification & Test Coverage

Two regression test cases were implemented in [`libs/core/tests/unit_tests/messages/test_utils.py`](file:///C:/Projects/github-portfolio/langchain/libs/core/tests/unit_tests/messages/test_utils.py):

### Test 1: `test_trim_messages_last_allow_partial_respects_token_budget_order`
* **Mechanism:** Uses a rule-based regex counter (`re.findall(r"\w+|\n", text) + 3`) as a deterministic test model for boundary fusion (simulating order-dependent token counts without introducing external tokenizer dependencies).
* **Input:** `HumanMessage("a\nb\nc")`, `max_tokens=5`.
* **Behavior:**
  - **Pre-fix:** Evaluated reversed `'cb\n'` (cost 5 $\le 5$), accepted `left = 2`, returned `'b\nc'` (cost 6 $> 5$) $\rightarrow$ **FAILED (Exceeded budget)**.
  - **Post-fix:** Evaluated forward `'b\nc'` (cost 6 $> 5$, rejected), evaluated `'c'` (cost 4 $\le 5$, accepted), returned `'c'` $\rightarrow$ **PASSED**.

### Test 2: `test_trim_messages_last_allow_partial_blocks_respects_token_budget_order`
* **Mechanism:** 3 text blocks `[{"type": "text", "text": "a\n"}, {"type": "text", "text": "b\n"}, {"type": "text", "text": "c"}]`.
* **Behavior:**
  - **Pre-fix:** `k = 2` evaluated reversed blocks `[c, b\n]` $\rightarrow$ `"cb\n"` (cost 5 $\le 5$), accepted candidate, returned forward `[b\n, c]` $\rightarrow$ `"b\nc"` (cost 6 $> 5$) $\rightarrow$ **FAILED (Exceeded budget)**.
  - **Post-fix:** `k = 2` evaluated forward blocks `[b\n, c]` $\rightarrow$ `"b\nc"` (cost 6 $> 5$, rejected), `k = 1` evaluated `[c]` $\rightarrow$ `"c"` (cost 4 $\le 5$, accepted), returned `[{"type": "text", "text": "c"}]` $\rightarrow$ **PASSED**.

Both tests were verified:
1. **Failing against pre-fix code:** Both text and block paths consistently reproduced the over-budget failure.
2. **Passing against post-fix code:** Both tests pass with 100% precision.

---

## 7. Status, Repository Conventions & Next Steps

1. **Repository Conventions (`AGENTS.md`):**
   - Branch name: `jadibdev/core/trim-messages-last-allow-partial` (following `<github-username>/<scope>/<short-description>`).
   - Commit title: ``fix(core): evaluate candidate splits in forward order in `trim_messages` with `strategy="last"` ``.
   - PR description format: Leading `Closes #41259\n---\n`, `## Release note`, and AI assistant acknowledgment.
2. **Maintainer Etiquette & Coordination:**
   - As observed on [Issue #41259](https://github.com/langchain-ai/langchain/issues/41259), the issue author requested assignment, and PR #41261 was auto-closed due to lack of maintainer assignment.
   - We will post our technical analysis on Issue #41259 and wait for maintainer assignment before opening the pull request.
