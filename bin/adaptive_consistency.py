#!/usr/bin/env python

import argparse
import dotenv
dotenv.load_dotenv()

from output_formats import ReasonedFloatOutput, FloatOutput
from confidence_models import BayesianConfidenceModel
from confident_solver import ConfidentSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini", "o3-mini-low", "o3-mini-medium", "o3-mini-high"], default="gpt-4o-mini")

    args = parser.parse_args()

    if args.model in ["gpt-4o", "gpt-4o-mini"]:
        output_schema = ReasonedFloatOutput
    elif args.model in ["o3-mini-low", "o3-mini-medium", "o3-mini-high"]:
        output_schema = FloatOutput
    else:
        raise ValueError("Unknown Model")

    consol = ConfidentSolver(
        confidence_model=BayesianConfidenceModel(confidence_threshold=0.95, priori="uniform"),
        llm_model=args.model,
        output_schema=output_schema,
    )
    res = consol.invoke(args.prompt, debug=True)
    print(res.to_csv())
