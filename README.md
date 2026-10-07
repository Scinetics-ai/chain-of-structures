# Chain of Structures

Code for *Chain of Structures: Connecting Reasoning and Generation for Protein Design*.

A language reasoning expert writes a design record over four rounds: catalytic core,
active site, global backbone, then complete sequence and structure. Each written round
re-conditions an all-atom generator, whose output becomes the next round's context.

- **Weights:** [`Scinetics-ai/chain-of-structures`](https://huggingface.co/Scinetics-ai/chain-of-structures)
  on the Hugging Face Hub. `chain_of_structures.pt` is a delta checkpoint carrying the trained
  parameters only, 230.4M in total: 160.8M unfrozen generator layers, 58.2M of rank-32 LoRA on
  the reasoner, 7.4M heads, 3.3M ground, 0.7M embeddings. Both base models are mirrored there
  unmodified under `base/`, with their own licences.

## What is here

The data-construction and training pipeline for the reasoning expert, as it ran on our cluster.
Directories are dated by run, and the `v11` and `v12` suffixes mark successive iterations of the
candidate-shortlist format. The paper's section *Four-Round Data and Supervision* describes the
corpus and the supervision that these scripts build.

| Path | What it does |
|---|---|
| `pipeline/r1_table_20261006/build_r1_table.py` | Builds a round's candidate shortlist. Every column is looked up from same-EC3 training enzymes or corpus priors; none is measured on the target. |
| `pipeline/r1_table_20261006/make_splits_v12.py` | Three-way split with the two properties asserted on the written files: `val_overlap` reactions all appear in train, `val_strict` reactions are disjoint from it. |
| `pipeline/r1_table_20261006/build_r1_requests_v12.py` | The teacher prompt. One candidate per line; justifications must argue chemistry and may not transcribe column values. |
| `pipeline/r1_table_20261006/assemble_sft_v12.py` | The four gates: picks in the shortlist, no out-of-table numbers, no recitation, residue distribution over the *selected* positions. |
| `pipeline/r1_table_20261006/merge_split_v12.py` | Eight hard assertions over the merged corpus, including split disjointness after back-fill. |
| `pipeline/r1_table_20261006/train_sft_v12.py` | LoRA SFT. Evaluates and reports the two validation sets separately; they are never averaged. |
| `pipeline/r1_table_20261006/lookup_only_baseline.py` | The lookup-only bar the reasoner has to beat. |
| `pipeline/r1_table_20261006/table_info_ceiling.py` | Fits a gradient-boosted classifier on the shortlist columns to measure what is left in them. |
| `pipeline/r1_table_20261006/eval_pairs_v12.py` | Offline pair evaluation, stratified by template distance. |
| `pipeline/teacher/` | Teacher serving and dispatch. |
| `pipeline/cvm_sglang_20261003/` | SGLang replica management for teacher generation. |
| `pipeline/r1_info_gate_20261006/` | Per-channel measurement of what each candidate information source is worth. |
| `pipeline/r1_regime_gate_20261005/` | Whether a round's draft is informative enough to reason over. |
| `pipeline/metrics_deep4b_20261006/` | Structure decoding and the EC, F1 and MMD metrics. |
| `pipeline/train_v11_20261006/` | The preceding training iteration, kept for comparison. |
| `pipeline/lthpc/` | Protocol-run preparation, auditing and release hashing. |

## Three things the format exists to prevent

Each produced plausible-looking output while being wrong, and none raised an error.

1. **A language model cannot index into a long digit string.** Packed onto one line, it
   invents positions on an integer grid while appearing to cite the table. The shortlist is
   printed one candidate per line, and out-of-table picks are measured as an acceptance gate.
2. **Alphabetical key order puts the answer before the reasoning.** Serialising with sorted
   keys places `picks` before `reasoning`. This affected 3,764 of 3,764 records in one batch.
   Output serialisation never sorts keys.
3. **A column the model cannot trust is worse than no column.** A shuffled tool report scored
   1.23x random; *no* report scored 2.13x. Beyond the trust bucket the column is withheld.

## Reproducing

The scripts take absolute paths from the cluster they ran on, and are published as a record of
what was run rather than as a turnkey package. The corpus itself is not included.

## Citation

See the paper. Code and weights are released for inspection of the procedure it describes.
