#!/usr/bin/env python

import argparse
import dotenv
dotenv.load_dotenv()

from .output_formats import ReasonedFloatOutput, FloatOutput
from .confidence_models import BayesianConfidenceModel
from .confident_solver import ConfidentSolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")
    parser.add_argument("--llm_model", choices=["gpt-4o", "gpt-4o-mini", "o3-mini-low", "o3-mini-medium", "o3-mini-high", "ollama:llama3.2:8b"], default="gpt-4o-mini")
    parser.add_argument("--confidence_model", choices=["sbft", "sprt", "bayesian", "pvalue", "vote"], default="sbft")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.llm_model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
        output_schema = FloatOutput
    else:
        output_schema = ReasonedFloatOutput

    consol = ConfidentSolver(
        llm_model=args.llm_model,
        confidence_model=args.confidence_model,
        output_schema=output_schema,
    )
    res = consol.invoke(args.prompt, debug=args.debug)
    if args.debug:
        print(res.to_csv())
    else:
        print(res)


if __name__ == "__main__":
    main()