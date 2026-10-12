# Formula Benchmark — Math-Correctness Results

Five decoupled stages. Passage detection is measured on all items; formula stages are measured on items that specify a formula, **independently of detection**.

## Stage 1 — Passage detection

| TP | FP | TN | FN | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1 | 4 | 8 | 0.571 | 0.889 | 0.500 | 0.640 |

| Item | Gold | Predicted | Conf | Top trigger |
| --- | --- | --- | --- | --- |
| F01 | True | False | 0.00 | — |
| F02 | True | False | 0.00 | — |
| F03 | True | True | 0.60 | Numeric content detected |
| F04 | True | False | 0.00 | — |
| F05 | True | False | 0.00 | — |
| F06 | True | True | 0.70 | Devanagari math keyword: 'वर्ग' |
| F07 | True | False | 0.00 | — |
| F08 | True | False | 0.00 | — |
| F09 | True | True | 0.70 | Devanagari math keyword: 'क्षेत्र' |
| F10 | True | True | 0.70 | Devanagari math keyword: 'क्षेत्र' |
| F11 | True | True | 0.70 | Devanagari math keyword: 'क्षेत्र' |
| F12 | True | True | 0.70 | Devanagari math keyword: 'क्षेत्र' |
| F13 | True | True | 0.70 | Devanagari math keyword: 'वर्ग' |
| F14 | True | False | 0.00 | — |
| F15 | True | False | 0.00 | — |
| F16 | True | True | 0.70 | Devanagari math keyword: 'संख्या' |
| N01 | False | False | 0.00 | — |
| N02 | False | False | 0.00 | — |
| N03 | False | False | 0.00 | — |
| N04 | False | False | 0.00 | — |
| N05 | False | True | 0.70 | Devanagari math keyword: 'क्षेत्र' |

## Stages 2–5 — Formula pipeline

| Stage | Scored | Correct | Rate |
| --- | --- | --- | --- |
| 2. Formula retrieval (any known formula) | 15 | 10 | 0.667 |
| 3. Formula extraction (exact registry key) | 9 | 1 | 0.111 |
| 4a. Symbolic parse valid | 13 | 13 | 1.000 |
| 4b. Symbolic equivalent to expected | 13 | 1 | 0.077 |
| 5. Interpretation name match | 15 | 1 | 0.067 |

## Per-item detail

| Item | Path | Expected key | Retrieved key | Extract OK | Parse | Equiv | Interp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | generated | pythagorean | — | False | True | False | False |
| F02 | registry | square_diagonal_double_area | pythagorean | — | True | False | False |
| F03 | generated | pythagorean_triples | — | — | True | — | False |
| F04 | registry | sqrt2_approximation | pythagorean | — | True | False | False |
| F05 | registry | pi_approximation | circle_area | — | True | — | False |
| F06 | registry | triangle_area | pythagorean | False | True | False | False |
| F07 | registry | circle_area | circle_area | True | True | True | True |
| F08 | generated | arithmetic_sum | — | False | True | False | False |
| F09 | registry | square_area | pythagorean | False | True | False | False |
| F10 | registry | rectangle_area | pythagorean | — | True | False | False |
| F11 | registry | triangle_area | pythagorean | False | True | False | False |
| F12 | registry | circle_area | triangle_area | False | True | False | False |
| F13 | registry | sphere_surface | circle_area | — | True | False | False |
| F14 | generated | quadratic | — | False | True | False | False |
| F15 | generated | vedic_square | — | False | True | False | False |

## Notes

- Interpretation scoring is an automatic name-match *proxy*; final interpretation correctness must be scored by a domain reviewer (`provenance.needs_reviewer` is true for every item).
- Stage 4b is `null` when the item has no expected closed-form expression.
- The extractor is run text-only (no knowledge-graph `shloka_id`), which isolates the keyword/registry logic from the KG lookup path.
