# 🎮 ConSol: Confident Solver

Solve problems confidently and efficiently using a statistical approach with LLM

[![PyPI version](https://badge.fury.io/py/consol.svg)](https://badge.fury.io/py/consol)

## 🤗 Getting Started

```bash
# This command installs the ConSol package from PyPI.
pip install consol
```

### Usage as a CLI

```bash
$ consol --prompt "1 + 1 = ?"
2
```

### Usage as a SDK

```python
from consol import ConfidentSolver

# Initialize the ConfidentSolver with the following parameters:
# llm_model: The language model to use, e.g., "gpt-4o-mini", "o3-mini-low".
# confidence_model: The statistical model to determine confidence, e.g., "pvalue", "sprt", "bayesian".
# output_schema: The format of the output, e.g., "reasoned_float", "float".
consol = ConfidentSolver(
    llm_model="gpt-4o-mini",
    confidence_model="pvalue",
    output_schema="reasoned_float",
)
answer = consol.invoke("1 + 1 = ?")
print(answer)
# => 2
```

## 🤔 What is ConSol?

**ConSol** is a framework designed to solve various problems, primarily mathematical, by leveraging a statistical approach. This approach suppresses randomness and results in higher accuracy cost-efficiently.

* **Higher Accuracy**: ConSol improves [OpenAI's GPT-o3-mini-medium](.) performance on [AIME24 Benchmark](.) by 20 percentage points from 73% to 93%.
* **Cost Efficiency**: ConSol can reduce from 50% to 66% of output tokens of [OpenAI's GPT-o3-mini-medium](.) for [AIME24 Benchmark](.). The number of output tokens is directly linked to the money cost. ConSol saves $50, $16, and $4 for o3-mini-high, o3-mini-medium, and o3-mini-low, respectively.

For the details, please [refer to the publication](.).
