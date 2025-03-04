# 🎮 ConSol: Confident Solver

Solve various problems using LLM confidently and efficiently with a statistical approach

## 🤗 Getting Started

```bash
pip install consol
consol --prompt "1 + 1 = ?"
```

or

```python
from consol import ConfidentSolver

consol = ConfidentSolver(
    llm_model="gpt-4o-mini",
    confidence_model="pvalue",
    output_schema="reasoned_float",
)
answer = consol.invoke("1 + 1 = ?")
print(answer)
```

## 🤔 What is ConSol?

**ConSol** is a framework to solve any problems, primarily mathematical problems but applicable to any, with leveraging a statistical approach to suppress randomness, and to result higher accuracy cost-efficiently.

* **Higher Accuracy**: ConSol improves [OpenAI's GPT-o3-mini-medium](.) performance on [AIME24 Benchmark](.) by 20%p from 73% to 93%.
* **Cost Efficiency**: ConSol can reduce from 50% to 66% of output tokens of [OpenAI's GPT-o3-mini-medium](.) for [AIME24 Benchmark](.). The number of output tokens is directly linked to the money cost. The money saving with ConSol is $50, $16, $4 for o3-mini-high, o3-mini-medium, o3-mini-low, respectively.

For the details, please [refer to the publication](.).

