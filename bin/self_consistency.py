#!/usr/bin/env python

import sys
import argparse

import langchain
import langchain_core
import langchain_openai
import pandas as pd
import dotenv
dotenv.load_dotenv()

from output_formats import ReasonedOutput, Output

def self_consistency(prompt, llm, schema, trials=40):
    raw_outputs = []
    while len(raw_outputs) < trials:
        k = trials - len(raw_outputs)
        _ = llm.with_structured_output(schema, include_raw=True).batch([[
            langchain_core.messages.HumanMessage(prompt),
        ]] * k)
        raw_outputs += [x for x in _ if x['parsed']]
    return pd.DataFrame({
        'answer': [x['parsed'].answer for x in raw_outputs],
        'token_usage': [x['raw'].response_metadata['token_usage']['completion_tokens'] for x in raw_outputs]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.")
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini", "o3-mini-low", "o3-mini-medium"], default="gpt-4o-mini")

    args = parser.parse_args()

    if args.model in ["gpt-4o", "gpt-4o-mini"]:
        llm = langchain_openai.ChatOpenAI(
            model=args.model,
        )
        schema = ReasonedOutput
    elif args.model in ["o3-mini-low", "o3-mini-medium"]:
        rate_limiter = langchain_core.rate_limiters.InMemoryRateLimiter(
            requests_per_second=1,
        )

        llm = langchain_openai.ChatOpenAI(
            model="o3-mini",
            reasoning_effort=args.model.split("-")[-1],
            rate_limiter=rate_limiter,
        )
        schema = Output

    res = self_consistency(args.prompt, llm=llm, schema=schema)
    print(res.to_csv())
