# SummEval demo data

Source: Fabbri, A., Kryscinski, W., McCann, B., Xiong, C., Socher, R., and Radev, D. (2020).
SummEval: Re-evaluating summarization evaluation. arXiv:2007.12626.
Repository: https://github.com/Yale-LILY/SummEval
License: MIT.

## Upstream license notice

This demo corpus is a derived subset redistributed under SummEval's MIT license.
The MIT terms require the upstream copyright and permission notice to travel with
any redistribution, so it is reproduced here in full.

> MIT License
>
> Copyright (c) 2021 Alex Fabbri, Wojciech Kryściński, Bryan McCann, Caiming Xiong,
> Richard Socher, and Dragomir Radev
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Subset

Dimension: coherence. Annotators: 3 expert raters. Items: 1600 (100 documents x 16 systems).
Scores: integers 1-5. Stratum: system family (extractive or abstractive) per the Fabbri et al.
(2020) model taxonomy. Prepared by scripts/prep_demo.py (not in wheel).

## Annotator identity

SummEval's expert_annotations field contains a list of 3 annotation dicts per summary.
The released JSONL contains no annotator ID fields. Slot ordering (index 0, 1, 2 mapped to
expert_0, expert_1, expert_2) is assumed consistent across all 1600 items. prep_demo.py
checks for annotator ID keys at prep time and logs the result.

Consequences:
- Krippendorff alpha: unaffected; it measures within-item agreement regardless of slot identity.
- ICC(2,1) and ICC(2,k): valid under the slot-stability assumption. Treat ICC estimates as
  conditional on this assumption.
- DIF (ordinal logistic regression): the default leave-one-rater-out rest score conditioner
  is computed per (item, rater) row from the other raters' scores for that item, so slot
  identity affects which score is left out but not the item-level quality signal.
