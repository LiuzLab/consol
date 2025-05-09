# 🎮 ConSol: Confident Solver

**ConSol** enables you to solve problems confidently and efficiently by leveraging Large Language Models (LLMs) enhanced with robust statistical approaches, such as the [Sequential Probability Ratio Test (SPRT)](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test).

[![PyPI version](https://badge.fury.io/py/consol.svg)](https://badge.fury.io/py/consol)
![PyPI - Downloads](https://img.shields.io/pypi/dm/consol)


---

## 🚀 Quick Start

### Installation

Install ConSol directly from PyPI:

```bash
pip install consol
```

### Command Line Usage

Solve a problem using ConSol via the CLI:

```bash
$ consol --prompt "1 + 1 = ?"
2
```

### SDK Usage

Integrate ConSol into your Python applications:

```python
from consol import ConfidentSolver

# Initialize ConSol with custom parameters
consol = ConfidentSolver(
    llm_model="gpt-4o-mini",         # Supported models: "gpt-4o-mini", "o3-mini-low", etc.
    confidence_model="msprt",        # Options: "msprt", "sprt", "pvalue", "bayesian_posterior", "vote40", "vote1"
    output_schema="float"            # Output formats: "abcd", "float", etc.
)

# Invoke ConSol to solve a problem
answer = consol.invoke("1 + 1 = ?")
print(answer)
# Output: 2
```

---

## 🔍 About ConSol

**ConSol** is an innovative framework designed for precise and cost-effective problem-solving. It integrates statistical validation with LLMs to significantly enhance accuracy and efficiency, especially for mathematical computations and reasoning tasks.

### Key Benefits

- 🚩 **Higher Accuracy**: ConSol improves [OpenAI's o3-mini-low and o3-mini-medium](https://openai.com/index/openai-o3-mini/) performance on the [AIME24 Benchmark](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) by 10-17 percentage points—boosting accuracy from 60% to 70% (o3-mini-low) and 73% to 90% (o3-mini-medium).

- 💰 **Cost Efficiency**: ConSol reduces the number of output tokens generated by [OpenAI's GPT-o3-mini models](https://openai.com/index/openai-o3-mini/) on the [AIME24 Benchmark](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) by 63.9%–84.8%, directly lowering computational and monetary costs.

For a comprehensive understanding and detailed methodology, please refer to our [research publication](https://www.alphaxiv.org/abs/2503.17587).

## 📚 Examples

### Accuracy

#### Without ConSol (Inconsistent Results)

```bash
$ consol --confidence_model vote1 --llm_model gpt-4o-mini --prompt "Jen enters a lottery by picking \(4\) distinct numbers from \(S=\{1,2,3,\dots,10\}\). \(4\) numbers are randomly chosen from \(S\). She wins a prize if at least two of her numbers match the randomly chosen numbers, and wins the grand prize if all four match. The probability of winning the grand prize given she has already won a prize is \(\frac{m}{n}\), with \(m,n\) relatively prime positive integers. Find \(m+n\)."
# or simply consol --confidence_model vote1
# => 11.0 (varies upon re-run)
```

Results vary with each run due to model inconsistency.

#### With ConSol (Consistent Results)

```bash
$ consol --confidence_model msprt --llm_model gpt-4o-mini --prompt "Jen enters a lottery by picking \(4\) distinct numbers from \(S=\{1,2,3,\dots,10\}\). \(4\) numbers are randomly chosen from \(S\). She wins a prize if at least two of her numbers match the randomly chosen numbers, and wins the grand prize if all four match. The probability of winning the grand prize given she has already won a prize is \(\frac{m}{n}\), with \(m,n\) relatively prime positive integers. Find \(m+n\)."
# or simply consol
# => 116.0 (consistent)
```

Typically, ConSol achieves consistency within 20-100 samples.

### Efficiency

#### Prior Method (Ada-Cons)

```bash
$ consol --confidence_model bayesian_posterior --prompt "random integer between 0 to 20"
# => Random number
```

Typically requires 40 samples (maximum reached frequently).

#### Our Method (mixture SPRT)

```bash
$ consol --confidence_model msprt --prompt "random integer between 0 to 20"
# => Random number
```

Typically concludes early within 10-30 samples due to effective randomness detection.

### Verbose Output for Debugging

```bash
$ consol --debug
index,answer,token_usage
0,116.0,484
1,47.0,81
2,11.0,69
3,116.0,483
4,37.0,337
5,7.0,221
6,3.0,168
7,37.0,116
8,116.0,585
9,21.0,114
10,9.0,156
11,49.0,67
12,19.0,165
13,20.0,81
14,22.0,222
15,116.0,514
16,145.0,117
17,49.0,97
18,15.0,135
19,116.0,427
```

Debugging mode provides a detailed table of answers and corresponding token usage for diagnostics.

