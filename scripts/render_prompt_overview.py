"""Render selected main prompt instructions as readable JSON-style blocks."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/figures/prompt_overview'
PROMPTS = (
    {
        'title': 'Answer generation',
        'source': 'src/verifiqa/generation/answer_generator.py',
        'symbol': '_PROMPT',
        'anchors': [
            'Do not use outside knowledge.',
            'Do not invent missing numbers.',
            'Return exactly one JSON object and nothing else.',
            'You must evaluate any arithmetic yourself.',
        ],
        'summary': {
            'task': 'Answer the financial question using only the retrieved filing evidence.',
            'inputs': ['question', 'evidence', 'retrieval plan', 'calculation guidance (if provided)'],
            'rules': [
                'Use only supplied evidence; do not invent missing numbers.',
                'Return one JSON object, with no markdown or surrounding prose.',
                'Compute the final value; include its explicit unit and display scale.',
                'Follow calculation guidance; preserve the requested quantity and scale.',
                'If the evidence or required unit is insufficient, say so in the answer.',
            ],
            'response': {'answer': '<concise final value, unit, scale, and calculation when possible>'},
        },
    },
    {
        'title': 'Verification planning',
        'source': 'src/verifiqa/agent/claim_classifier.py',
        'symbol': '_DECOMPOSE_PROMPT',
        'anchors': [
            'You must not use, infer from, or anticipate any proposed answer.',
            'roles must list every variable that appears in the formula.',
            'Every reported observation must be a role variable; never copy source numbers into the formula.',
            'If none of the above apply: formula_source = "no_formula", formula = "", roles = [].',
        ],
        'summary': {
            'task': 'Resolve the metric, formula, operands, and expected unit for verification.',
            'inputs': ['question', 'evidence', 'operation hint', 'formula registry'],
            'rules': [
                'Plan independently of the proposed answer; return JSON only.',
                'Use an authorized formula or an explicit source definition.',
                'Preserve the requested measure, scope, and periods.',
                'Bind every observed quantity to an operand role, not a numeric literal.',
                'If no supported formula applies, use no_formula and empty formula/roles.',
            ],
            'response': {
                'verifiable': '<true|false>',
                'metric': '<requested metric>',
                'formula': '<arithmetic using role names>',
                'roles': [{'name': '<operand>', 'aliases': ['<source label>'], 'period': '<period>'}],
                'claim_unit': '<expected answer unit>',
                'formula_source': '<policy_registry|generic_operation|document_derived|no_formula>',
            },
        },
    },
    {
        'title': 'Fact grounding',
        'source': 'src/verifiqa/agent/fact_extractor.py',
        'symbol': '_PROMPT',
        'anchors': [
            'Extract the value for ONE financial line item from the evidence text below.',
            'Do NOT convert units.',
            'source_quote must be copied verbatim from the evidence and must contain the numeric value.',
            'Do not invent values not present in the evidence.',
        ],
        'summary': {
            'task': 'Extract one financial operand from the source evidence.',
            'inputs': ['question', 'role and aliases', 'period', 'formula', 'evidence'],
            'rules': [
                'Select the requested measure, row, and period; return JSON only.',
                'Copy the source value without converting units or changing its scale.',
                'Report the unit established by the source header or caption; do not guess.',
                'Copy a verbatim source quotation containing the numeric value.',
                'If the requested value is unavailable, return found=false and a reason.',
            ],
            'response': {
                'value': '<number as printed>',
                'unit': '<as-reported unit and scale>',
                'source_quote': '<verbatim evidence containing the value>',
                'period': '<requested period>',
                'row_label': '<exact source label>',
            },
            'if_missing': {'found': False, 'reason': '<brief reason>'},
        },
    },
)


def validate_excerpts() -> None:
    """Check supporting source excerpts without importing model code."""
    for prompt in PROMPTS:
        tree = ast.parse((ROOT / prompt['source']).read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == prompt['symbol']
                                  for t in node.targets))
        source = ' '.join(ast.literal_eval(assignment.value).split())
        for anchor in prompt['anchors']:
            if ' '.join(anchor.split()) not in source:
                raise ValueError(f"Source anchor no longer matches {prompt['source']}")


def format_block(summary: dict) -> str:
    """Keep inputs and response fields compact while retaining valid JSON."""
    def encode(value):
        return json.dumps(value, ensure_ascii=False)
    lines = ['{']
    for key, value in summary.items():
        if key == 'rules':
            lines.append('  "rules": [')
            lines.extend('    ' + encode(rule) + (',' if i < len(value) - 1 else '')
                         for i, rule in enumerate(value))
            lines.append('  ],')
        elif key == 'response':
            lines.append('  "response": {')
            fields = list(value.items())
            lines.extend('    ' + encode(k) + ': ' + encode(v)
                         + (',' if i < len(fields) - 1 else '')
                         for i, (k, v) in enumerate(fields))
            lines.append('  },')
        else:
            lines.append('  ' + encode(key) + ': ' + encode(value) + ',')
    lines[-1] = lines[-1].removesuffix(',')
    lines.append('}')
    result = '\n'.join(lines)
    assert json.loads(result) == summary
    return result


TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|[^"a-z]+|[a-z]+')


def draw_code(ax, code: str, x: float, y: float) -> None:
    """Draw JSON with restrained syntax colors and a fixed-width font."""
    for row, line in enumerate(code.splitlines()):
        for match in TOKEN.finditer(line):
            token = match.group()
            if token.startswith('"'):
                color = '#245E8A' if line[match.end():].lstrip().startswith(':') else '#344453'
            elif token in ('true', 'false', 'null'):
                color = '#76549B'
            else:
                color = '#73808C'
            ax.text(x + match.start() * 9.63, y + row * 21, token,
                    fontfamily='DejaVu Sans Mono', fontsize=11.55,
                    color=color, va='top')


def main() -> None:
    validate_excerpts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blocks = [format_block(prompt['summary']) for prompt in PROMPTS]
    assert max(len(line) for block in blocks for line in block.splitlines()) <= 110
    heights = [len(block.splitlines()) * 21 + 77 for block in blocks]
    height = 83 + sum(heights) + 20 * (len(blocks) - 1) + 24
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(12, height / 100), facecolor='white')
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 1200), ylim=(height, 0))
        ax.axis('off')
        ax.text(24, 27, 'Main prompts', fontsize=21, weight='bold',
                va='center', color='#203345')
        ax.text(24, 57, 'Selected instructions and response schemas from current prompts',
                fontsize=11.5, color='#64748B', va='center')
        y = 83
        for prompt, block, block_height in zip(PROMPTS, blocks, heights):
            ax.add_patch(FancyBboxPatch((24, y), 1152, block_height,
                         boxstyle='round,pad=0,rounding_size=8',
                         facecolor='#F8FAFC', edgecolor='#DFE5EC', linewidth=0.8))
            ax.text(46, y + 25, prompt['title'], fontsize=13.5, weight='bold',
                    va='center', color='#244B6B')
            ax.plot((24, 1176), (y + 49, y + 49), color='#DFE5EC', linewidth=0.7)
            draw_code(ax, block, 46, y + 63)
            y += block_height + 20
        fig.savefig(OUTPUT.with_suffix('.png'), dpi=180, facecolor='white')
        fig.savefig(OUTPUT.with_suffix('.svg'), facecolor='white', metadata={'Date': None})
        plt.close(fig)
    OUTPUT.with_suffix('.json').write_text(
        json.dumps([prompt['summary'] for prompt in PROMPTS], indent=2) + '\n'
    )
    print('Saved prompt overview PNG, SVG, and copyable JSON; all three blocks parse and 12 source anchors match.')


if __name__ == '__main__':
    main()
