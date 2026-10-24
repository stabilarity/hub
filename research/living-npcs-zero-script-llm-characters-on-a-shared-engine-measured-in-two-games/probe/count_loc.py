"""Counts lines of code per layer (engine core, each game) and writes data/loc.json.
Physical lines and non-blank lines that are not pure // or # comments. Vendored libraries (three.js) excluded.
Run on the server where the code lives; the JSON is committed so analysis.py runs offline."""
import json, os, re, sys
GROUPS = {
  'core_server': ['/root/ai-npc-core/sim.js', '/root/ai-npc-core/brain.js', '/root/ai-npc-core/server.js', '/root/ai-npc-core/tileworld.js'],
  'core_client': ['/root/ai-npc-core/client/ui.js', '/root/ai-npc-core/client/ui.css', '/root/ai-npc-core/client/i18n.js'],
  'quarthaven_server': ['/root/wizard-village/wizards.js', '/root/wizard-village/personas.js', '/root/wizard-village/lore.js', '/root/wizard-village/world.js', '/root/wizard-village/server.js'],
  'quarthaven_client': ['/root/wizard-village/public/index.html', '/root/wizard-village/public/render3d.js', '/root/wizard-village/public/i18n-wiz.js'],
  'oakhollow_port_server': ['/root/living-village/port-core-draft/oakhollow.js', '/root/living-village/port-core-draft/personas.js', '/root/living-village/port-core-draft/world.js', '/root/living-village/port-core-draft/server.js', '/root/living-village/lore.js'],
  'oakhollow_port_client': ['/root/living-village/port-core-draft/public/index.html', '/root/living-village/port-core-draft/public/render2d.js'],
  'oakhollow_v5_standalone_server': ['/root/living-village/village.js', '/root/living-village/brain.js', '/root/living-village/server.js', '/root/living-village/world.js', '/root/living-village/personas.js', '/root/living-village/lore.js'],
  'oakhollow_v5_standalone_client': ['/root/living-village/public/game.js', '/root/living-village/public/index.html', '/root/living-village/public/i18n.js'],
}
def count(p):
    lines = open(p, encoding='utf8').read().split('\n')
    if lines and lines[-1] == '': lines = lines[:-1]
    code = [l for l in lines if l.strip() and not re.match(r'^\s*(//|#|\*|/\*)', l)]
    return len(lines), len(code)
out = {}
for g, files in GROUPS.items():
    rows = {os.path.relpath(f, '/root'): dict(zip(('lines', 'code'), count(f))) for f in files}
    out[g] = {'files': rows, 'lines': sum(r['lines'] for r in rows.values()), 'code': sum(r['code'] for r in rows.values())}
# the game-specific part of the decision schema: actions and JSON fields a game adds to the core prompt
sys.path.insert(0, '.')
out['_note'] = 'counted 2026-10-25; lore.js of the Oakhollow port is the parent directory file the draft requires'
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for g, v in out.items():
    if g[0] != '_': print(f"{g:34s} lines={v['lines']:5d} code={v['code']:5d}")
