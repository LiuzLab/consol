params.input_jsonl = file("./resources/data/gsm.jsonl")
params.publishDirSuffix = ""

process JSONL_TO_CSV {
    input:
    stdin

    output:
    stdout

    """
    #!/usr/bin/env python
    import pandas as pd
    import sys

    df = pd.read_json(sys.stdin, lines=True).reset_index()

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
    consol --prompt "$safe_input" --debug --llm_model gpt-4o-mini --confidence_model vote > ${id}.csv
    """
}

workflow {
    tuples_ch = JSONL_TO_CSV(file(params.input_jsonl).text).splitCsv(header:true, quote: '"')
    CONSOL(tuples_ch)
}