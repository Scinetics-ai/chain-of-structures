# Chain of Structures

Code for *Chain of Structures: Connecting Reasoning and Generation for Protein Design*.

A language reasoning expert writes a design record over four rounds: catalytic core, active
site, global backbone, then complete sequence and structure. Each written round re-conditions
an all-atom generator, whose output becomes the next round's context.

This repository holds the four-stage injection arm end to end, which is the arm the paper
reports and the arm whose weights are released. Everything here ran on our own cluster, and the
absolute paths are left in place so that each script still says exactly what it read.

- **Weights:** [`Scinetics-ai/chain-of-structures`](https://huggingface.co/Scinetics-ai/chain-of-structures)
  on the Hugging Face Hub. `chain_of_structures.pt` is a delta checkpoint carrying the trained
  parameters only, 230.4M in total: 160.8M unfrozen generator layers, 58.2M of rank-32 LoRA on
  the reasoner, 7.4M heads, 3.3M ground, 0.7M embeddings. Both base models are mirrored there
  unmodified under `base/`, with their own licences.

## The model

`model/dplm2bit_textcond_hybrid.py` is the whole thing: the conditioned generator, the staged
trainer, and the multi-round sampler. The generator is DPLM-2-bit 650M and the reasoner is
Qwen3.5-9B under a rank-32 LoRA.

Text reaches the generator at **each of the last eight layers** rather than at a single entry
point (`--use_deep_per_token_cross --deep_cross_layers 8`), and the written residue identities
reach it as a separate discrete channel through a gated injector (`--ground_mode gated`). The
released run has that discrete channel open at **all four** rounds. Earlier runs had it only at
rounds two and three, because the round-four and round-five documents named no residue at all,
so the extractor returned zero pairs there and the injector was a no-op. `data/patch_text_v2.py`
and `data/shells_v1.py` are what gave those two rounds something to name.

## Layout

| Path | What it is |
|---|---|
| `model/dplm2bit_textcond_hybrid.py` | Model, staged trainer, multi-round sampler. |
| `train/run_deep2like.sh` | The recipe. Its header lists, flag by flag, what is shared with the two-stage run and the three things that differ. |
| `train/run_stage_2node.sh` | The two-node, sixteen-GPU harness, including why `NCCL_IB_DISABLE=1` is mandatory on these boxes. |
| `train/wait_and_launch_A.sh` | Verifies the cards really are empty before launching. A refused or duplicated launch still fires. |
| `train/after_stageA.sh` | Lets the writer write, parses it, and fits the per-round noise profile. |
| `data/patch_text_v2.py` | Gives rounds four and five a residue-naming sentence, writing beside the originals so the two-stage arm stays bit-identical. |
| `data/shells_v1.py` | The two new position layers behind those sentences, from the same CIF and the same heavy-atom minimum. |
| `data/build_pairs_v2.py` | The discrete channel's pair table, with two guards the earlier builder did not need while rounds four and five were always empty. |
| `data/build_docs_v2.py` | Tokenised five-round documents. The original builder was not kept, so its conventions were recovered and reproduced on 20 of 20 records first. |
| `data/stage_means_v2.py` | Per-round hidden-state means. Not optional: the mean is subtracted before the states reach the conditioner, and the new text moves it. |
| `data/refit_noise_v2.py` | Fits the writer's measured reliability per round. A round with pairs but no profile would be injected as an oracle with no error raised. |
| `data/build_text_hash_cache.py`, `data/wire_hash_cache.sh` | Precomputes the text hash table. Without it every rank spends seventeen minutes at startup. |
| `data/run_build.sh` | The whole build in order, stopping at the first step whose own assertion fails. |
| `generate/gen_cot_docs_v2.py` | The writer writes the four rounds. Asserts the LoRA keys, shapes and values after loading. |
| `generate/gen_d2like.sh` | End-to-end generation from the released checkpoint, sixteen shards balanced by length rather than by count. |
| `generate/launch_writer.sh`, `generate/eval_worker.py`, `generate/continue_eval.py` | Per-shard dispatch, with a refusal on a shard that already has a status file. |
| `generate/generation_ready.json` | The pre-generation record: sample count, arms, and the sha256 of each script that ran. |
| `score/band_final.py` | The EC criterion. |
| `score/clean_d2like.sh`, `score/score_clean_shard.py` | EC1 to EC4 through CLEAN, per shard. |
| `score/eval_f1_ec.py` | Micro and macro F1 in the official baseline's own terms, which is the ruler that reproduced their published numbers to within 0.003. |
| `score/eval_mmd.py` | MMD on k-mer spectra, reported at k = 1, 2, 3 rather than silently choosing one. |
| `score/decode_struct_to_pdb.py`, `score/run_decode.sh` | Structure tokens to PDB for the self-consistency step, with a geometry gate: exit code zero is not evidence of a fold. |
| `score/eval_direct_esmfold_consistency_local.py`, `score/run_esmfold.sh` | pLDDT, TM and scRMSD: the generated sequence is folded with ESMFold and compared against the generated structure. |
| `score/watch_metrics.sh` | Samples the folding run cumulatively rather than per shard, so a part-finished shard cannot move the number. |
| `score/diag_macf1.py` | Why macro F1 is low, by class support rather than by explanation. |
| `score/evals.py`, `score/peek_d2like.py` | Training-curve readout and the checkpoint's component breakdown. |

## What the released run actually was

| | |
|---|---|
| Run | `q35_deep2like_4stage_16g`, sixteen GPUs across two nodes |
| Checkpoint | `FINAL_d2like_step9000_val0.7807.pt`, step 9000, best validation 0.780707 |
| Recipe | `train/run_deep2like.sh`, with `cond_lr` 6e-4 and `qwen_lr` 2e-4, the four-GPU values scaled by the square root of the sixteenfold batch |
| `--ground_noise` | **not passed** |

The last line matters, and it is checked rather than remembered. The trainer prints
`[GROUND] pairs degraded per record` only when that flag is on
(`model/dplm2bit_textcond_hybrid.py`, in the grounding setup), and that line does not appear in
the run's log. Passing it costs the arm a great deal: training then corrupts the injected pairs
to a reliability that is not the writer's own.

**Two revisions of the trainer exist, and this is the later one.** Training ran a revision from
30 September. The file here is the 2 October revision, whose sha256 is pinned in
`generate/generation_ready.json` and which produced every reported generation. It differs in
four places, all of them guards rather than behaviour: a per-index cache in `tokenize_pdbs`, a
budget line in the multi-round sampler, `_qwen_hidden_size` moved above its use and taught to
read `text_config` because Qwen3.5 is multimodal and puts the width there, and a key and shape
assertion on the LoRA load.

## Reproducing

The scripts take absolute paths from the cluster they ran on, and are published as a record of
what was run rather than as a turnkey package. The corpus itself is not included.

## Citation

See the paper. Code and weights are released for inspection of the procedure it describes.
