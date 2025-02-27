#!/usr/bin/env python

import sys
import argparse

import langchain
import langchain_core
import langchain_openai
import tqdm
import tqdm.auto
import pandas as pd
import scipy
import scipy.stats
import dotenv
dotenv.load_dotenv()

from output_formats import ReasonedOutput, Output

def get_opportunistic_trials(first, second):
    assert first >= second
    for k in range(0, 40):
        if first+k == 0:
            continue
        if scipy.stats.binomtest(first+k, first+second+k, p=0.5, alternative='greater').pvalue <= 0.05:
            return k
    raise "Opportunistic Trials Not Found"

def adaptive_consistency(prompt, llm, schema, max_trials=40):
    total_raw_outputs = []
    with tqdm.auto.tqdm(total=max_trials) as pbar:
        while True:
            total_ss = pd.Series([x['parsed'].answer for x in total_raw_outputs]).value_counts()

            two = total_ss.sort_values(ascending=False).head(2).to_list()

            while len(two) < 2:
                two += [0]
            trials = get_opportunistic_trials(two[0], two[1])
            if trials >= max_trials - len(total_raw_outputs):
                trials = max_trials - len(total_raw_outputs)

            if trials == 0:
                break

            raw_outputs = []
            while len(raw_outputs) < trials:
                try:
                    k = trials - len(raw_outputs)
                    _ = llm.with_structured_output(schema, include_raw=True).batch([[
                        langchain_core.messages.HumanMessage(prompt),
                    ]] * k)
                    raw_outputs += [x for x in _ if x['parsed']]
                except Exception as e:
                    print("unknown error", e, file=sys.stderr)
                    continue
            total_raw_outputs += raw_outputs
            pbar.update(trials)

    return pd.DataFrame({
        'answer': [x['parsed'].answer for x in total_raw_outputs],
        'token_usage': [x['raw'].response_metadata['token_usage']['completion_tokens'] for x in total_raw_outputs]
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

    res = adaptive_consistency(args.prompt, llm=llm, schema=schema)
    print(res.to_csv())
