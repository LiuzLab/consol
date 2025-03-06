import pandas as pd
import typing

def load_dataset(name: typing.Literal['asdiv', 'gsm', 'aime24']) -> pd.DataFrame:
    if name == 'aime24':
        df = pd.read_parquet("hf://datasets/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet")
        df = df.rename(columns={'Problem': 'input', 'Answer': 'target'})
        df = df[['input', 'target']]
        return df
    elif name == 'asdiv':
        df = pd.read_json('./resources/data/asdiv.jsonl', lines=True)
        return df
    elif name == 'gsm':
        df = pd.read_json('./resources/data/gsm.jsonl', lines=True)
        return df

    raise ValueError("Unknown dataset")
