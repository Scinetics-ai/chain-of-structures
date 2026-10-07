"""Opt-in v4 protocol; old structural-tag schema remains pinned and unchanged.

Only tool availability changes. State counts are produced separately from the
current PublicState; this module cannot infer CA coordinates from visible bits.
"""
from copy import deepcopy

import episode_constraints as base

CONSTRAINT_VERSION = 'offline_episode_structural_tag_v4_availability_20261003'


def lookup_completed(public):
    return base.lookup_completed(public)


def allowed_tool_names(public):
    if not lookup_completed(public):
        return ('lookup_reaction_evidence',)
    return tuple(name for name in base.ALLOWED_TOOLS if name != 'lookup_reaction_evidence')


def make_response_format(public, review):
    result = deepcopy(base.make_response_format(public, review))
    if lookup_completed(public):
        final = result['format']['elements'][3]['content']['json_schema']
        allowed = set(allowed_tool_names(public))
        final['anyOf'] = [branch for branch in final['anyOf']
                          if branch['properties']['type'].get('const') == 'condition'
                          or branch['properties']['name']['const'] in allowed]
    return result
