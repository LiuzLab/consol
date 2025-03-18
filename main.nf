params.input = "gpqa_diamond"
params.publishDirSuffix = ""
params.llm_model = "o3-mini-low"
params.confidence_model = "vote"

process JSONL_TO_CSV {
    output:
    stdout

    """
    #!/usr/bin/env python
import pandas as pd
import sys
import typing
import random

def load_dataset(name: typing.Literal['asdiv', 'gsm', 'aime24', 'medqa', 'gpqa_diamond']) -> pd.DataFrame:
    if name == 'aime24':
        df = pd.read_parquet("hf://datasets/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet")
        df = df.rename(columns={'Problem': 'input', 'Answer': 'target'})
        df = df[['input', 'target']]
        return df
    elif name == 'asdiv':
        df = pd.read_json("${projectDir}/resources/data/asdiv.jsonl", lines=True)
        return df
    elif name == 'gsm':
        df = pd.read_json('${projectDir}/resources/data/gsm.jsonl', lines=True)
        return df
    elif name == 'medqa':
        df = pd.read_json('${projectDir}/resources/data/medqa.jsonl', lines=True)
        letter_to_number = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        sorted_keys = list("ABCDE")
        formatted_options = ['\\n'.join([str(i+1)+". " +x[key] for i, key in enumerate(sorted_keys)]) for x in df['options']]

        instruction_text = 'Please select the number between 1 and 5 that corresponds to the correct answer.'

        formatted_question = [f'{q}\\n\\n{instruction_text}\\n\\n{o}' for q,o in zip(df['question'], formatted_options)]

        answer_number = df["answer_idx"].apply(lambda x: letter_to_number.get(x))

        if answer_number is None:
            raise ValueError("The answer index does not match any known option.")

        df = pd.DataFrame({
            "input": formatted_question,
            "target": answer_number
        })
            df = pd.DataFrame({
                "input": formatted_question,
                "target": answer_number
            })

        return df

    elif name == 'gpqa_diamond':
        SEED = 42
        random.seed(SEED)

        df = pd.read_csv("${projectDir}/resources/data/gpqa_diamond.csv")
        
        answer_columns = [
            "Pre-Revision Correct Answer",
            "Pre-Revision Incorrect Answer 1",
            "Pre-Revision Incorrect Answer 2",
            "Pre-Revision Incorrect Answer 3"
        ]

        options_labels = ["A", "B", "C", "D"]

        formatted_questions = []
        correct_answers = []

        for _, row in df.iterrows():

            shuffled_columns = random.sample(answer_columns, len(answer_columns))

            shuffled_options = {opt: row[col] for opt, col in zip(options_labels, shuffled_columns)}

            question_text = f"Question: {row['Pre-Revision Question']}\\n"
            choices_text = "\\n".join([f"{opt}. {shuffled_options[opt]}" for opt in options_labels])
            prompt_text = f"{question_text}{choices_text}"

            correct_option = [opt for opt, col in zip(options_labels, shuffled_columns) if col == "Pre-Revision Correct Answer"][0]

            formatted_questions.append(prompt_text)
            correct_answers.append(correct_option)

        gpqa_df = pd.DataFrame({
            "input": formatted_questions,
            "target": correct_answers
        })

        return gpqa_df

df = load_dataset('${params.input}').reset_index()
# Escape to avoid CSV misinterpretation from Nextflow.
df['input'] = df.input.apply(lambda x: x.replace('\\n', '<br/>').replace('"', "&quot;"))
print(df.to_csv(index=False))
    """
}

process CONSOL {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 32
    publishDir "published/${params.publishDirSuffix}/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash
    consol --prompt "$safe_input" --debug --output_type abcdef --llm_model ${params.llm_model} --confidence_model ${params.confidence_model} > ${id}.csv
    """
}

workflow {
    tuples_ch = JSONL_TO_CSV().splitCsv(header:true, quote: '"')
    CONSOL(tuples_ch)
}
