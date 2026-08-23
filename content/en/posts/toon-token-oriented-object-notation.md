---
title: "Token-Oriented Object Notation: An Alternative to JSON?"
date: 2025-11-27
categories: [My Takes]
draft: false
slug: toon-token-oriented-object-notation
---

I first came across **TOON (Token-Oriented Object Notation)** on LinkedIn, where developers
were discussing its promise: **lossless compression of JSON and YAML**, specifically
optimized for large language models (LLMs). The core idea? Reduce token count without
sacrificing information, which directly translates to **lower costs** and potentially
**faster processing**.

![Comparison between TOON and JSON for nested objects](/images/toon-token-oriented-object-notation/nested-comparison.png)
_Comparison between TOON and JSON for nested objects_

At first glance, TOON looks familiar yet distinct. For instance, a TOON object always begins
with the number of items (e.g., `[2]` in the figure above), and unlike JSON it drops
quotation marks (`"`) around keys. This minimalism becomes especially powerful with flat
data structures. In those cases, TOON starts to resemble CSV format: a header row followed
by values. The result? Significant token savings (I used their Python version for my
experiments).

![Comparison between JSON and TOON for flat objects](/images/toon-token-oriented-object-notation/flat-comparison.png)
_Comparison between JSON and TOON for flat objects_

According to the project's evaluation, which tested 209 questions across several tasks,
TOON achieves "higher accuracy while using fewer tokens."

![Evaluation of TOON on 209 questions](/images/toon-token-oriented-object-notation/evaluation-results.png)
_Evaluation of TOON on 209 questions_

The question distribution breaks down as follows: Field retrieval (33%), Aggregation (30%),
Filtering (23%), Structure awareness (12%), Structural validation (2%). With this
distribution you can see why there's a large gap between JSON and TOON in structural
validation — they validated it on 2% of 209 questions (4 or 5 questions).

Digging deeper into their methodology and prompt design, I'm skeptical that the reported
accuracy gains are as robust as claimed. The token savings, however, are real, especially
for flat data. Here is the prompt used for the evaluation:

```text
Given the following data in toon format:

TOON: Indentation-based. Arrays declare length and fields (e.g., items[N]{f1,f2}:). Rows use single delimiter. Values may be quoted.

{data_in_toon_format}

Question: {question}

Answer format requirements:
- Provide only the value itself, no explanation
- For numbers: output digits only (no commas, currency symbols, or units)
- For dates/field names: use the exact string from the data
- For lists: output comma-separated values with no spaces

Answer:
```

That said, there's a caveat worth considering: LLMs are typically pretrained on standard
formats like JSON and YAML, complete with proper spacing, quotes, and structure. Introducing
a non-standard notation like TOON might increase the risk of hallucinations.

So while TOON is a clever optimization for specific use cases — especially cost-sensitive,
flat-data applications — it's not a drop-in replacement for traditional serialization
formats.

## References

- [GitHub — toon-format/toon](https://github.com/toon-format/toon)
- [toonformat.dev](https://toonformat.dev/)
