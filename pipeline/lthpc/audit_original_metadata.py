#!/usr/bin/env python3
"""Read-only inventory of historical chemistry metadata; JSON to stdout."""
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

ROOT = Path("/data/share/enzyme_design")
WORK = Path("/data/workspace/caiqingran")
SPLIT = WORK / "protein_design/DISCO_repo/runs/cot_split_v2"
AUDITED = WORK / "agent_dplmbit_20261003/data"


def present(v):
    return bool(v and str(v).strip().lower() not in {"[]", "{}", "nan", "none", "null", "-"})


def csv_rows(path):
    with path.open() as f:
        yield from csv.DictReader(f)


def main():
    csv.field_size_limit(10000000)
    reqs = json.loads((SPLIT / "train_rows_cotstep1.json").read_text())
    rejected = {json.loads(line)["parent_id"] for line in (AUDITED / "quarantine_train.jsonl").open()}
    reqs = {r["idx"]: r for r in reqs if r["idx"] not in rejected}
    with (WORK / "baselines/data_ours/train.pkl").open("rb") as f:
        raw = pickle.load(f)
    rows = {r["idx"]: {k: r[k] for k in ("idx", "uniprot_id", "sequence", "ec_list", "native_pdb",
                                            "go_numbers", "ipr_numbers", "domain_sites", "real_domain", "motif")}
            for r in raw if r["idx"] in reqs}
    del raw
    ids, uids = set(rows), {r["uniprot_id"] for r in rows.values()}
    report = {"scope": "accepted_training_records_only", "records": len(rows), "unique_uniprot_ids": len(uids),
              "read_only": True, "no_reference_protein_promoted_to_public_input": True}
    report["baseline_fields"] = {
        "complete_sequence_and_structure_codes": len(rows),
        "EC": sum(present(r["ec_list"]) for r in rows.values()),
        "GO_molecular_function": sum(bool(r["go_numbers"].get("F")) for r in rows.values()),
        "GO_cellular_component": sum(bool(r["go_numbers"].get("C")) for r in rows.values()),
        "GO_biological_process": sum(bool(r["go_numbers"].get("P")) for r in rows.values()),
        "InterPro": sum(bool(r["ipr_numbers"]) for r in rows.values()),
        "domain_sites": sum(bool(r["domain_sites"]) for r in rows.values()),
        "motif_field_includes_full_target_sequence": sum(any(m.get("motif_segment") == r["sequence"] for m in r["motif"]) for r in rows.values()),
    }
    metadata_path = ROOT / "enzyme_data/enzyme_data.csv"
    metadata = {r["IDX"]: r for r in csv_rows(metadata_path) if r["IDX"] in ids}
    report["reaction_metadata"] = {"source": str(metadata_path), "matched_records": len(metadata),
        "nonempty_fields": {k: sum(present(r.get(k)) for r in metadata.values()) for k in (
            "MASTER_ID", "RHEA_ID", "DIRECTION", "smarts", "SUBSTRATE", "Cofactor", "COFACTOR", "COFACTOR_SMILES",
            "NOTE", "COFACTOR_NAME", "COFACTOR_IDX", "BINDING_NUMBER", "REMARK")},
        "unique_reaction_strings": len({r["smarts"] for r in metadata.values() if present(r["smarts"])}),
        "reaction_strings_with_atom_map_syntax": sum(bool(re.search(r":\d+\]", r.get("smarts", ""))) for r in metadata.values()),
        "sequence_exact_match_with_training_target": sum(r["Sequence"] == rows[idx]["sequence"] for idx, r in metadata.items()),
        "direction_counts": dict(Counter(r["DIRECTION"] for r in metadata.values())),
        "direction_interpretation": "UN is retained as the raw source label; this audit does not establish a forward physiological reaction direction.",
        "validity_limit": "presence counted; atom mapping/chemical parsing/mechanism are not validated by this inventory"}
    print("reaction metadata joined", len(metadata), file=sys.stderr, flush=True)
    uni_path = ROOT / "enzyme_data/uniprot/uniprot_data.csv"
    uni = {r["Entry"]: r for r in csv_rows(uni_path) if r["Entry"] in uids}
    mapped = [(idx, uni[r["uniprot_id"]]) for idx, r in rows.items() if r["uniprot_id"] in uni]
    report["uniprot_metadata"] = {"source": str(uni_path), "matched_unique_ids": len(uni), "matched_records": len(mapped),
        "nonempty_fields_by_record": {k: sum(present(r.get(k)) for _, r in mapped) for k in ("Active Site", "Binding Site", "Cofactor", "Subunit", "PDB")},
        "sequence_exact_match_with_training_target": sum(r["Sequence"] == rows[idx]["sequence"] for idx, r in mapped),
        "active_site_with_any_ECO_evidence": sum("ECO:" in r.get("Active Site", "") for _, r in mapped),
        "active_site_with_any_experimental_ECO_0000269": sum("ECO:0000269" in r.get("Active Site", "") for _, r in mapped),
        "active_site_with_any_PubMed": sum("PubMed:" in r.get("Active Site", "") for _, r in mapped),
        "binding_site_with_any_experimental_ECO_0000269": sum("ECO:0000269" in r.get("Binding Site", "") for _, r in mapped),
        "interpretation": "feature-specific evidence retained; coordinates/positions remain private reference labels"}
    active_path = ROOT / "cot_design/source/all_active_sites.json"
    annotations = json.loads(active_path.read_text())
    report["parsed_site_labels"] = {"source": str(active_path), "matched_records": sum(r["uniprot_id"] in annotations for r in rows.values()),
        "nonempty_by_record": {kind: sum(bool(annotations.get(r["uniprot_id"], {}).get(kind)) for r in rows.values()) for kind in ("active", "binding", "site")},
        "label_scope": "reference endpoint identities/addresses; not current-draft observations"}
    kb_path = ROOT / "cot_design/source/ec_kb.json"
    kb = json.loads(kb_path.read_text())
    by_ec = {idx: [kb[ec] for ec in re.findall(r"[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)", r["ec_list"]) if ec in kb] for idx, r in rows.items()}
    report["curated_EC_knowledge"] = {"source": str(kb_path), "database_entries": len(kb),
        "records_with_any_EC_match": sum(bool(v) for v in by_ec.values()),
        "records_with_reaction_text": sum(any(present(x.get("reaction")) for x in v) for v in by_ec.values()),
        "records_with_cofactor_text": sum(any(present(x.get("cofactor")) for x in v) for v in by_ec.values()),
        "fields": ["name", "reaction", "cofactor"], "mechanism_limit": "no validated detailed mechanism or transition-state coordinates field"}
    print("UniProt and EC joined", len(mapped), file=sys.stderr, flush=True)
    complex_root = ROOT / "enzyme_data/boltz-1x"
    inputs = {p.stem for p in (complex_root / "input").iterdir() if p.suffix == ".yaml"}
    output_dirs = {p.name for p in (complex_root / "output").iterdir() if p.is_dir()}
    record_dirs = {p.name.removeprefix("cot_") for p in (ROOT / "cot_design/records").iterdir() if p.is_dir()}
    pdb_names = {p.name for p in (ROOT / "enzyme_text_data/data").iterdir()}
    possible = sorted(ids & output_dirs)
    def model_zero_exists(idx):
        path = complex_root / "output" / idx / ("boltz_results_" + idx) / "predictions" / idx / (idx + "_model_0.cif")
        return idx, path.is_file()
    with ThreadPoolExecutor(max_workers=8) as pool:
        model_zero = {idx for idx, ok in pool.map(model_zero_exists, possible) if ok}
    report["reference_structure_assets"] = {
        "protein_pdb_filename_present": sum(Path(r["native_pdb"]).name in pdb_names for r in rows.values()),
        "boltz_input_yaml_present": len(ids & inputs), "boltz_output_directory_present": len(ids & output_dirs),
        "boltz_model0_CIF_present": len(model_zero),
        "core_pocket_claims_record_directory_present": len(ids & record_dirs),
        "full_geometry_match_not_certified": True,
        "epistemic_status": "Boltz-1x predicted reference complexes; not experimental native poses, never current-draft observations"}
    # Sample 64 deterministic metadata joins; no scan of millions of model files.
    sample_ids = sorted(model_zero & inputs, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:64]
    sample = Counter()
    for idx in sample_ids:
        text = (complex_root / "input" / (idx + ".yaml")).read_text()
        seqs = re.findall(r"^\s+sequence:\s*([A-Z]+)\s*$", text, re.M)
        sample["yaml_target_sequence_match"] += rows[idx]["sequence"] in seqs
        sample["yaml_has_ligand_entity"] += bool(re.search(r"^\s*- ligand:\s*$", text, re.M))
        cif = complex_root / "output" / idx / ("boltz_results_" + idx) / "predictions" / idx / (idx + "_model_0.cif")
        sample["model0_has_HETATM_records"] += "HETATM" in cif.read_text()
    report["predicted_complex_sample_audit"] = {"sample_size": len(sample_ids), "sampling": "deterministic SHA256(ID) lowest64", "counts": dict(sample), "not_exhaustive_geometry_validation": True}
    report["missing_structured_fields_in_audited_primary_sources"] = ["organism", "pH", "temperature", "kcat", "Km", "validated_reaction_mechanism_steps", "transition_state_geometry"]
    report["important_caveats"] = [
        "Public goal fields must also be available at inference; reference-protein annotation values cannot be silently assumed user-provided.",
        "Active/binding site addresses, full sequence, motif_segment, reference geometry and target-specific contact claims are private endpoint supervision.",
        "EC-prefix hand rules labelled IUBMB mechanism in older claims are manual summaries, not database-derived mechanistic proof.",
        "Source CSV presence does not certify chemical correctness; preserve ECO/PubMed and predicted/observed distinctions."]
    examples = []
    for idx in ["19989_Q7NP62", "24813_P74921"]:
        if idx not in rows or idx not in metadata:
            continue
        meta = metadata[idx]
        examples.append({
            "public_goal_candidate": {"function_request": reqs[idx]["comment"], "final_length": len(rows[idx]["sequence"]),
                "EC": rows[idx]["ec_list"], "reaction_identifier": {"RHEA": meta["RHEA_ID"], "master_RHEA": meta["MASTER_ID"]},
                "direction_as_stored": meta["DIRECTION"],
                "reaction_SMILES": meta["smarts"], "substrate_SMILES_as_stored": meta["SUBSTRATE"],
                "cofactor_SMILES_as_stored": meta["COFACTOR_SMILES"],
                "curated_function_knowledge": by_ec[idx]},
            "private_provenance_sidecar": {"source_record_idx": idx, "uniprot_id": rows[idx]["uniprot_id"],
                "public_request_source": str(SPLIT / "train_rows_cotstep1.json"),
                "chemistry_source": str(metadata_path), "EC_source": str(kb_path),
                "inference_availability_requirement": "Must be requested/supplied chemistry or function-level retrieved knowledge; not inferred from unknown target protein during deployment.",
                "not_in_public_goal": ["native_sequence", "native_residue_positions", "native_structure", "UniProt_ID", "predicted_reference_contact_distances"]},
            "source_text_chemistry_consistency": "not_validated",
            "approved_for_SFT": False})
        if idx == "24813_P74921":
            examples[-1]["source_text_chemistry_consistency"] = "example_conflict_detected"
            examples[-1]["issue"] = {
                "type": "reaction_text_vs_substrate_SMILES_mismatch",
                "description": "The request and EC knowledge name L-leucine / 4-methyl-2-oxopentanoate, but the stored substrate is valine and the product corresponds to 3-methyl-2-oxobutanoate.",
                "stored_substrate_SMILES": "CC(C)[C@H]([NH3+])C(=O)[O-]",
                "stored_substrate_formula_RDKIT_checked": "C5H11NO2",
                "leucine_comparison_SMILES": "CC(C)C[C@H]([NH3+])C(=O)[O-]",
                "leucine_comparison_formula_RDKIT_checked": "C6H13NO2",
                "resolution": "Raw sources retained unchanged. Do not choose either source as correct without reviewing the specific RHEA reaction; do not use this example as approved SFT input.",
            }
    report["sanitized_initial_goal_examples"] = examples
    report["original_evidence_addendum"] = {
        "aggregate_cotfull": "Does not retain structured per-claim evidence citations; its reasoning is not automatically verified.",
        "recovered_original_sources": "Original UniProt CSV contains feature-specific ECO/PubMed evidence, and cot_design/records/cot_<IDX>/claims.json retains source fields. Re-auditing these original sources can recover some provenance.",
        "limitation": "Original claims also contain manual EC-prefix mechanism rules and predicted-reference geometric observations. Source labels alone do not certify an experimentally established mechanism or a valid current-draft observation.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
