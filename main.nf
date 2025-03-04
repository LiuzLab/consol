params.input_jsonl = file("./resources/data/aime24.jsonl")
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

process CONSOL_CONSISTENCY {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 5
    publishDir "published/${params.publishDirSuffix}/consol_consistency/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash

    export PATH=\$PATH:$moduleDir/bin/consol
    main.py --prompt "$safe_input" --debug --llm_model o3-mini-high > ${id}.csv
    """
}

process ADAPTIVE_CONSISTENCY {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 5
    publishDir "published/${params.publishDirSuffix}/adaptive_consistency/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash

    export PATH=\$PATH:$moduleDir/bin/consol
    adaptive_consistency.py --prompt "$safe_input" --model o3-mini-high > ${id}.csv
    """
}

process SELF_CONSISTENCY {
    input:
    tuple(val(id), val(input), val(target))

    output:
    file("${id}.csv")

    maxForks 1
    publishDir "published/${params.publishDirSuffix}/self_consistency/"
    tag "${id}.csv"

    script:
    // Escape to avoid bash misinterpretation.
    def safe_input = input.replace("\$", "\\\$")
    """
    #!/usr/bin/env bash

    export PATH=\$PATH:$moduleDir/bin/consol
    self_consistency.py --prompt "$safe_input" --model o3-mini-high > ${id}.csv
    """
}

workflow {
    tuples_ch = JSONL_TO_CSV(file(params.input_jsonl).text).splitCsv(header:true, quote: '"')
    CONSOL_CONSISTENCY(tuples_ch)
    SELF_CONSISTENCY(tuples_ch)
    // ADAPTIVE_CONSISTENCY(tuples_ch)
}