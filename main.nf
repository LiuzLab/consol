params.input_jsonl = file("./resources/data/aime24.jsonl")

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

process ADAPTIVE_CONSISTENCY {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 32
    publishDir "published/adaptive_consistency/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash

    adaptive_consistency.py --prompt "$safe_input" --model o3-mini-low > ${id}.csv
    """
}

process SELF_CONSISTENCY {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 1
    publishDir "published/self_consistency/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash

    self_consistency.py --prompt "$safe_input" --model o3-mini-low > ${id}.csv
    """
}

workflow {
    tuples_ch = JSONL_TO_CSV(file(params.input_jsonl).text).splitCsv(header:true, quote: '"')
    // ADAPTIVE_CONSISTENCY(tuples_ch)
    SELF_CONSISTENCY(tuples_ch)
}