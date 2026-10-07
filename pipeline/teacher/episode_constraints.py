"""Versioned constrained decoding for new offline teacher episodes only.

Hashes and citation tuples are runtime constants, never a copying exercise.
The original strict parser and semantic review still apply. This does not repair
past completions, approve chemical claims, or supply expert protein actions.
"""
from __future__ import annotations

from copy import deepcopy

from agent_dplmbit.reasoned_response import evidence_catalog, make_evidence_ref
from agent_dplmbit.schema import AA_ALPHABET, ContractError, PublicExample

CONSTRAINT_VERSION = "offline_episode_structural_tag_v2_20261003"
ALLOWED_TOOLS = ("lookup_reaction_evidence", "get_current_state", "inspect_current_geometry",
                 "get_dplm_proposals", "inspect_candidate_evidence")
PROMPT_SUFFIX = """
This version uses runtime-constrained JSON: state hashes and whole-item citation tuples are supplied constants.
Before an actual lookup_reaction_evidence observation exists, the protocol requires that tool next.
This lookup requirement is a workflow rule, not evidence of autonomous scientific tool selection.
Before lookup, summarize only what evidence has not yet been queried and why a lookup is needed.
Do not interpret raw SMILES, name a specific reaction, assert mechanisms/residues, or claim the tool lacks evidence before querying it.
After lookup, remain within the returned source scope; names-only evidence cannot justify catalytic AA preferences.
Do not emit strength, probabilities, REOPEN or KEEP. A valid address never certifies a beneficial edit.
role_id is protocol metadata fixed by candidate order: candidate_1, candidate_2, candidate_3; it is not a chemical role name.
All-mask length specifies an empty design canvas, not an observed protein scaffold.
""".strip()


def lookup_completed(public: PublicExample) -> bool:
    """A failed/unavailable actual lookup still counts as a queried observation.

The collector/replayer, not this grammar, verifies the observation's execution.
"""
    return any(obs.tool == "lookup_reaction_evidence" and obs.state_hash == public.state.state_hash
               for obs in public.observations)


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def _slot(public):
    return {"anyOf": [{"type": "integer", "minimum": 1, "maximum": public.state.length},
                       {"type": "string", "enum": list(public.state.residue_ids)}]}


def _tool_schema(public, name):
    slot = _slot(public)
    if name in {"lookup_reaction_evidence", "get_dplm_proposals"}:
        arguments = _object({})
    elif name == "inspect_current_geometry":
        pair = {"type": "array", "items": slot, "minItems": 2, "maxItems": 2}
        arguments = _object({"pairs": {"type": "array", "items": pair, "maxItems": 64}})
    elif name in {"get_current_state", "inspect_candidate_evidence"}:
        arguments = _object({"slots": {"type": "array", "items": slot, "maxItems": 64}}, [])
    else:
        raise ContractError("Unknown fixed-state episode tool")
    return _object({"type": {"const": "tool_call"}, "name": {"const": name},
                    "arguments": arguments, "state_hash": {"const": public.state.state_hash}})


def _hypothesis(public, review, candidate_index):
    base = {"role_id": {"const": f"candidate_{candidate_index}"},
            "address_binding": {"const": "pending"}, "residue_id": {"type": "null"},
            "position_1based": {"type": "null"}, "action": {"enum": ["NONE", "PREFER"]},
            "aa_preference": {"enum": list(AA_ALPHABET)}}
    pending = _object(base)
    all_unknown = all(a is None for a in public.state.aa) and all(b is None for b in public.state.bits)
    if public.state.phase == "initial" or all_unknown or not review["allow_bound_positions"]:
        return pending
    proposed = deepcopy(base)
    proposed["address_binding"] = {"const": "proposed"}
    proposed["residue_id"] = {"type": "string", "enum": list(public.state.residue_ids)}
    proposed["position_1based"] = {"type": "integer", "minimum": 1, "maximum": public.state.length}
    # The unchanged strict parser checks ID/index agreement, uniqueness and
    # protected slots; a grammar-valid address does not bypass those checks.
    return {"anyOf": [pending, _object(proposed)]}


def response_schemas(public: PublicExample, review: dict):
    if not isinstance(public, PublicExample) or type(review.get("allow_bound_positions")) is not bool:
        raise ContractError("Typed current public state and explicit address permission are required")
    refs = [make_evidence_ref(public, key, "") for key in evidence_catalog(public)]
    reasoning = _object({"state_hash": {"const": public.state.state_hash},
                         "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                         "evidence_refs": {"type": "array", "items": {"enum": refs},
                                           "minItems": 1, "maxItems": 16},
                         "uncertainty": {"type": "string", "minLength": 1, "maxLength": 1000}})
    if not lookup_completed(public):
        return reasoning, _tool_schema(public, "lookup_reaction_evidence")
    # Installed XGrammar treats every prefixItems entry as required, even with
    # minItems=0. Separate exact-length branches preserve valid abstention and
    # one/two-candidate responses without forcing three chemical hypotheses.
    hypotheses = {"anyOf": [
        {"type": "array", "prefixItems": [_hypothesis(public, review, index)
                                          for index in range(1, count + 1)],
         "items": False, "minItems": count, "maxItems": count}
        for count in range(4)
    ]}
    condition = _object({"type": {"const": "condition"}, "state_hash": {"const": public.state.state_hash},
                         "execution": {"const": "soft"}, "coordinate_lock": {"const": False},
                         "hypotheses": hypotheses})
    return reasoning, {"anyOf": [_tool_schema(public, name) for name in ALLOWED_TOOLS] + [condition]}


def make_response_format(public: PublicExample, review: dict) -> dict:
    """SGLang 0.5.19 / XGrammar structured-tag request (no runtime imports).

Two ordered JSON tags with whitespace only outside them. XGrammar's whole-object
enum emits the reference tuple atomically, including its exact hash and path.
"""
    reasoning, final = response_schemas(public, review)
    whitespace = {"type": "regex", "pattern": r"[ \t\r\n]*"}
    return {"type": "structural_tag", "format": {"type": "sequence", "elements": [
        whitespace,
        {"type": "tag", "begin": "<reasoning>",
         "content": {"type": "json_schema", "json_schema": reasoning}, "end": "</reasoning>"},
        whitespace,
        {"type": "tag", "begin": "<final>",
         "content": {"type": "json_schema", "json_schema": final}, "end": "</final>"},
        whitespace,
    ]}}
