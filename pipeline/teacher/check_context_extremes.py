#!/usr/bin/env python3
"""CPU tokenizer check only; synthetic states are never teacher/SFT examples."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from agent_dplmbit.schema import PublicExample, PublicRequest, PublicState
from generate_drafts import (Request, TokenCounter, audit_context, canonical, digest,
                             expand_state, messages_for, read_config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    selected, scores = {}, {}
    with open(args.public_manifest) as stream:
        for line in stream:
            row = json.loads(line)
            goal = row['public_goal']
            criteria = {'longest_protein': goal['final_length'],
                        'longest_goal_text': len(canonical(goal)),
                        'combined_size_proxy': len(canonical(goal)) + 12 * goal['final_length']}
            for name, score in criteria.items():
                if score > scores.get(name, -1):
                    scores[name], selected[name] = score, goal
    cfg = read_config(args.config)
    counter = TokenCounter(cfg['tokenizer_path'], cfg['enable_thinking'])
    results = []
    for selection, goal in selected.items():
        n = goal['final_length']
        for kind in ('all_mask', 'synthetic_mixed_observation'):
            aa = (None,) * n if kind == 'all_mask' else tuple(i % 20 if i % 3 else None for i in range(n))
            bits = (None,) * n if kind == 'all_mask' else tuple((0,) * 13 if i % 2 else None for i in range(n))
            state = PublicState(n, 'initial' if kind == 'all_mask' else 'denoising', aa, bits,
                                tuple(bool(i % 2) for i in range(n)))
            public = PublicExample(PublicRequest(canonical(goal), n), state)
            request = Request('context_test', 'synthetic_not_training', 'train', public, digest(public.to_dict()))
            messages = messages_for(request, {'allow_bound_positions': kind != 'all_mask'})
            rendered = json.loads(messages[1]['content'])
            assert expand_state(rendered['evidence_catalog']['state']['value']) == public.prompt_dict()['state']
            result = audit_context(messages, {}, cfg, counter)
            results.append({'selection': selection, 'synthetic_state_kind': kind, 'length': n,
                            'lossless_roundtrip': True, **result})
    output = {'scope': 'CPU context extremes; no model inference, no chemistry certification',
              'actual_tool_report_lengths_validated': False,
              'all_results_fit': all(x['fits'] for x in results), 'results': results}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    if not output['all_results_fit']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
