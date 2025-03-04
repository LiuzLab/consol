import pandas as pd
import typing

def load_dataset(name: typing.Literal['asdiv', 'gsm', 'aime24']) -> pd.DataFrame:
    if name == 'aime24':
        df = pd.read_parquet("hf://datasets/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet")
        df = df.rename(columns={'Problem': 'input', 'Answer': 'target'})
        df = df[['input', 'target']]
        return df
    if name == 'asdiv':
        df = pd.read_json('./AdaptiveConsistency/adaptive_consistency_datasets/asdiv.jsonl', lines=True)
        return df
