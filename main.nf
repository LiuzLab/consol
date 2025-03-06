params.input = "medqa"
params.publishDirSuffix = ""

process JSONL_TO_CSV {
    output:
    stdout

    """
    #!/usr/bin/env python
    import pandas as pd
    import sys
    import consol.utils
    df = consol.utils.load_dataset('${params.input}').reset_index()
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
    tuples_ch = JSONL_TO_CSV().splitCsv(header:true, quote: '"')
    CONSOL(tuples_ch)
}
