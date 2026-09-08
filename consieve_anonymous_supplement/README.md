# ConSieve Anonymous Supplement

This archive contains the frozen manifests, neutral checkpoint/configuration registry,
compact cached hosted-model outputs, aggregate result artifacts, and analysis/evaluation
code used by the anonymous submission. It intentionally excludes the working repository,
Git history, raw API ledgers, credentials, machine logs, internal milestone reports, and
multi-gigabyte model weights.

## Layout
- manifests/: frozen evaluation IDs and diagnostic/SQuAD constructs.
- configs/: frozen protocol and training/evaluation configurations.
- model_registry/: checkpoint and baseline revisions/hashes; no weights.
- results/: aggregate results, bootstrap outputs, human-audit artifacts, and compact cached outputs.
- prompts/: hosted Qwen generator/Judge parameters and prompt-bearing frozen protocol.
- code/: evaluation, training, aggregation, and statistical analysis code.
- figures/: source for the reported budget-sensitivity plot.

Hosted model caveat: the provider alias qwen-plus exposed no immutable snapshot identifier.
The dates, parameters, prompts, retry rule, denominator-preserving failure rule, and cached
per-query outputs are included. Future API calls therefore need not be bitwise identical.

All paths in text/config files use neutral placeholders such as ${CONSIEVE_ROOT}.
