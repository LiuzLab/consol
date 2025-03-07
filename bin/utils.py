import pandas as pd
import typing

def load_dataset(name: typing.Literal['asdiv', 'gsm', 'aime24', 'medqa']) -> pd.DataFrame:
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
    elif name == 'medqa':
        df = pd.read_json('./resources/data/medqa_org.jsonl', lines=True)
        letter_to_number = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

        sorted_keys = list("ABCDE")
        formatted_options = ["\n".join([str(i+1)+". " +x[key] for i, key in enumerate(sorted_keys)]) for x in df["options"]]

        instruction_text = "Please select the number between 1 and 5 that corresponds to the correct answer."

        formatted_question = [f"{q}\n\n{instruction_text}\n\n{o}" for q,o in zip(df['question'], formatted_options)]

        answer_number = df["answer_idx"].apply(lambda x: letter_to_number.get(x))

        if answer_number is None:
            raise ValueError("The answer index does not match any known option.")

        df = pd.DataFrame({
            "input": formatted_question,
            "target": answer_number
        })

        return df

    raise ValueError("Unknown dataset")

