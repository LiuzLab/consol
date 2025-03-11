params.input = "simple_bench_public"
params.publishDirSuffix = "test"
params.llm_model = "o3-mini-low"
params.confidence_model = "sprt"

process JSONL_TO_CSV {
    output:
    stdout

    """
    #!/usr/bin/env python
    import pandas as pd
    import sys
    import typing
    import json

    def load_dataset(name: typing.Literal['asdiv', 'gsm', 'aime24', 'medqa']) -> pd.DataFrame:
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

            return df
        elif name == 'simple_bench_public':
            with open('${projectDir}/resources/data/simple_bench_public.json', "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data["eval_data"])
            df = pd.DataFrame({
                "input": df['prompt'],
                "target": df['answer']
            })

            return df
        else:
            raise ValueError("Unknown dataset")

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
    consol --prompt "$safe_input" --debug --llm_model ${params.llm_model} --confidence_model ${params.confidence_model} > ${id}.csv
    """
}

workflow {
    tuples_ch = JSONL_TO_CSV().splitCsv(header:true, quote: '"')
    CONSOL(tuples_ch)
}
