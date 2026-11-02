# Builds the committed inputs under data/ (run once on the game server; needs pyarrow only for the LIGHT parquet).
#  - data/sessions.json     every join on the live Oakhollow and Quarthaven servers, 23 Oct - 2 Nov 2026, with its provenance class
#  - data/natural_scenes.json  chat transcripts around the human-operated sessions (no IPs, in-game pseudonyms only)
#  - data/probe_scenes.json transcripts of the article-2 probe sessions (template-scripted players)
#  - data/light_items.json  60 LIGHT test dialogues (human crowdworkers role-playing in a fantasy world), one target turn each
#  - data/code_facts.json   constants and lists read from the game code (speeds, typing rate, palettes, names, lore)
# usage: PYTHONPATH=<pyarrow> python3 prepare_data.py <light test.parquet>
import json, os, random, re, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
A2 = '/root/hub/research/living-npcs-zero-script-llm-characters-on-a-shared-engine-measured-in-two-games/data/raw'

def events(paths):
    out = []
    for p in paths:
        for l in open(p):
            try: out.append(json.loads(l))
            except Exception: pass
    return sorted(out, key=lambda e: e['ts'])

oak = events(sorted(glob.glob('/root/living-village/logs/events-*.jsonl')))
wiz = events(sorted(glob.glob('/root/wizard-village/logs/events-*.jsonl')))

# ---- provenance -------------------------------------------------------------------
# Evidence: time stamps of test scripts in the development session transcripts (tool calls that opened a websocket
# to the local game port and joined), and developer requests written minutes after a manual play session.
SCRIPTED = {  # (game, name, join time prefix) -> evidence
    ('oak', 'Остап', '2026-10-23T18:17'): 'test script bot2.js started 18:17:27 UTC joins as this name, lang uk, tests pause and microphone upload',
    ('oak', 'Остап', '2026-10-23T18:20'): 'second run of the same test script',
    ('oak', 'Hamo', '2026-10-23T18:26'): 'test script bot3.js started 18:26:50 UTC drives two players',
    ('oak', 'Drogo', '2026-10-23T18:26'): 'test script bot3.js (second player)',
    ('oak', 'Hamo', '2026-10-23T19:02'): 'test script bot4.js started 19:02:22 UTC',
    ('oak', 'Drogo', '2026-10-23T19:02'): 'test script bot4.js (private village with custom lore)',
    ('oak', 'Fulk', '2026-10-23T19:02'): 'test script bot4.js (third player)',
    ('oak', 'Hamo', '2026-10-23T19:03'): 'follow-up run of the test script',
    ('oak', 'Serlo', '2026-10-23T19:03'): 'follow-up run of the test script',
    ('oak', 'Probeton', '2026-10-24T01:58'): 'smoke test of the live websocket after deployment',
    ('oak', 'Probetwo', '2026-10-24T02:15'): 'smoke test of the live websocket after deployment',
}
DEVELOPER = {
    ('oak', 'Rose'): 'manual play; at 18:02 UTC the developer asked to be able to pause and to choose a character',
    ('oak', 'Margery'): 'same player as Rose after death (v0 respawned humans under a new name)',
    ('oak', 'Hugh'): 'manual play with keyboard and microphone; at 01:53 UTC the developer asked that a villager asked to follow may refuse',
    ('oak', 'Edwin'): 'same person as Hugh, rejoined 0 s after Hugh left; speech-to-text lines',
}
def classify(game, e):
    k = (game, e['name'], e['ts'][:16])
    if k in SCRIPTED: return 'scripted_test', SCRIPTED[k]
    if (game, e['name']) in DEVELOPER: return 'developer_manual', DEVELOPER[(game, e['name'])]
    return 'unknown', ''

sessions = []
for game, evs in (('oak', oak), ('wiz', wiz)):
    open_ = {}
    for e in evs:
        if e['type'] == 'join': open_[e['name']] = {'game': game, 'name': e['name'], 'join': e['ts'], 'lang': e.get('lang'), 'v': e.get('v')}
        elif e['type'] == 'respawn' and e.get('name'): open_[e['name']] = {'game': game, 'name': e['name'], 'join': e['ts'], 'respawn': True}
        elif e['type'] == 'death' and e.get('kind') == 'human' and e['name'] in open_: open_[e['name']]['died'] = e['ts']
        elif e['type'] == 'leave' and e['name'] in open_:
            s = open_.pop(e['name']); s['leave'] = e['ts']; s['lived_s'] = e.get('lived_s'); sessions.append(s)
    for s in open_.values():
        if 'died' in s: s['leave'] = s['died']
        else:   # no leave logged (world restarted): last event that names the player
            evs = oak if s['game'] == 'oak' else wiz
            s['leave'] = max([e['ts'] for e in evs if s['name'] in (e.get('from'), e.get('who'), e.get('name')) and e['ts'] >= s['join']] or [s['join']])
        sessions.append(s)
for s in sessions:
    t0, t1 = s['join'], s.get('leave', s['join'])
    evs = oak if s['game'] == 'oak' else wiz
    win = [e for e in evs if t0 <= e['ts'] <= t1]
    s['lines'] = sum(1 for e in win if e['type'] == 'say' and e.get('from') == s['name'] and e.get('kind') == 'human')
    s['stt_lines'] = sum(1 for e in win if e['type'] == 'stt' and e.get('who') == s['name'] and e.get('chars'))
    s['acts'] = sum(1 for e in win if e['type'] in ('hit', 'cast') and (e.get('from') == s['name'] or e.get('who') == s['name']))
    s['llm_calls_in_window'] = sum(1 for e in win if e['type'] == 'llm')
    cls, ev = classify(s['game'], {'name': s['name'], 'ts': s['join']})
    if cls == 'unknown' and (s.get('lived_s') is not None and s['lived_s'] <= 1 and not s['lines'] and not s['acts']):
        cls, ev = 'connection_check', 'joined and left within 1 s without any action (page or websocket check)'
    if cls == 'unknown' and s['game'] == 'wiz' and s['join'].startswith('2026-10-24T22:4'):
        ev = 'first minutes after the 3D game went live, while its development session was still open; no chat, no model call'
    if cls == 'unknown' and s['game'] == 'wiz' and s['join'].startswith('2026-10-27'):
        ev = '20 s visit, two spells, no chat, no model call'
    s['class'], s['evidence'] = cls, ev
json.dump(sessions, open(os.path.join(DATA, 'sessions.json'), 'w'), ensure_ascii=False, indent=1)

# ---- natural scenes (human-operated, developer) -------------------------------------
def scene(evs, t0, t1, sid, build, humans):
    lines = []
    for e in evs:
        if e['type'] != 'say' or not (t0 <= e['ts'] <= t1): continue
        lines.append({'ts': e['ts'], 'from': e['from'], 'kind': e['kind'], 'to': e.get('to'), 'text': e['text']})
    return {'id': sid, 'build': build, 't0': t0, 't1': t1, 'humans': humans, 'lines': lines}
nat = [
    scene(oak, '2026-10-23T17:58:49', '2026-10-23T18:00:30', 'A1', 'v0 (routines, reflex attack)', ['Rose']),
    scene(oak, '2026-10-24T01:44:00', '2026-10-24T01:45:00', 'B1', 'v4 (routines, reflex attack)', ['Hugh']),
    scene(oak, '2026-10-24T01:45:00', '2026-10-24T01:47:45', 'B2', 'v4 (routines, reflex attack)', ['Edwin']),
    scene(oak, '2026-10-24T01:47:45', '2026-10-24T01:50:45', 'B3', 'v4 (routines, reflex attack)', ['Edwin']),
]
json.dump(nat, open(os.path.join(DATA, 'natural_scenes.json'), 'w'), ensure_ascii=False, indent=1)

# ---- article-2 probe scenes (template-scripted players vs zero-script NPCs) ----------
probe = []
for run in ('wiz-gemini', 'wiz-mistral', 'oak-gemini'):
    evs = events(glob.glob(f'{A2}/{run}/events-*.jsonl'))
    stim = [json.loads(l) for l in open(f'{A2}/{run}/stimuli.jsonl')]
    bots = sorted({s['bot'] for s in stim})
    says = [e for e in evs if e['type'] == 'say']
    if not says: continue
    t0 = says[0]['ts']
    # first 3 minutes of phase 1: every speaker
    from datetime import datetime, timedelta
    t1 = (datetime.fromisoformat(t0.replace('Z', '+00:00')) + timedelta(seconds=180)).isoformat().replace('+00:00', 'Z')
    probe.append({'id': run, 'build': 'v5 zero-script engine', 't0': t0, 't1': t1, 'humans': bots,
                  'lines': [{'ts': e['ts'], 'from': e['from'], 'kind': e['kind'], 'to': e.get('to'), 'text': e['text']} for e in says if e['ts'] <= t1]})
json.dump(probe, open(os.path.join(DATA, 'probe_scenes.json'), 'w'), ensure_ascii=False, indent=1)

# ---- LIGHT items ---------------------------------------------------------------------
import pyarrow.parquet as pq
rows = pq.read_table(sys.argv[1]).to_pylist()
rng = random.Random(20261103)
cand = [i for i, r in enumerate(rows) if len(r['dialogue']) >= 7 and r['characters']['self_name'] and r['characters']['partner_name']]
pick = sorted(rng.sample(cand, 60))
items = []
for i in pick:
    r = rows[i]; d = list(r['dialogue']); k = rng.choice([2, 4, 6])
    items.append({'light_index': i, 'setting_name': r['setting']['name'], 'setting': r['setting']['description'],
                  'self_name': r['characters']['self_name'], 'partner_name': r['characters']['partner_name'],
                  'self_persona': r['characters']['self_persona'], 'context': d[:k], 'human_reply': d[k]})
json.dump(items, open(os.path.join(DATA, 'light_items.json'), 'w'), ensure_ascii=False, indent=1)

# ---- facts read from the code -----------------------------------------------------------
def grab(path, pat):
    m = re.search(pat, open(path).read(), re.S); return m.group(1) if m else None
sim = open('/root/ai-npc-core/sim.js').read()
facts = {
    'core_speed_human': float(grab('/root/ai-npc-core/sim.js', r'SPEED_HUMAN = ([\d.]+)')),
    'core_speed_npc': float(grab('/root/ai-npc-core/sim.js', r'SPEED_NPC = ([\d.]+)')),
    'core_speed_run': float(grab('/root/ai-npc-core/sim.js', r'SPEED_RUN = ([\d.]+)')),
    'type_ms_per_char': int(grab('/root/ai-npc-core/sim.js', r'TYPE_MS_PER_CHAR = (\d+)')),
    'type_min_ms': int(grab('/root/ai-npc-core/sim.js', r'TYPE_MIN_MS = (\d+)')),
    'type_max_ms': int(grab('/root/ai-npc-core/sim.js', r'TYPE_MAX_MS = (\d+)')),
    'gate_settle_ms': int(grab('/root/ai-npc-core/sim.js', r'GATE_SETTLE_MS = (\d+)')),
    'gate_min_gap_ms': int(grab('/root/ai-npc-core/sim.js', r'GATE_MIN_GAP_MS = (\d+)')),
    'doze_ms': int(grab('/root/ai-npc-core/sim.js', r'DOZE_MS = (\d+)')),
    'human_say_min_gap_ms': int(grab('/root/ai-npc-core/sim.js', r'now - h.lastSayAt > (\d+)')),
    'npc_respawn_ms': int(grab('/root/ai-npc-core/sim.js', r'now - n.deathAt > (\d+)')),
    'live_oak_same_constants': all(x in open('/root/living-village/village.js').read() for x in
        ['SPEED_HUMAN = 3.2, SPEED_NPC = 2.3, SPEED_RUN = 3.1', 'TYPE_MS_PER_CHAR = 90, TYPE_MIN_MS = 1500, TYPE_MAX_MS = 9000', 'DOZE_MS = 60000', "a.kind === 'npc' ? people.map"]),
    'ui_rel_hint_en': grab('/root/ai-npc-core/client/i18n.js', r'rel_hint: "([^"]+)"'),
    'ui_no_out_en': grab('/root/ai-npc-core/client/i18n.js', r'no_out: "([^"]+)"'),
}
oakdef = open('/root/living-village/port-core-draft/oakhollow.js').read()
oakp = open('/root/living-village/port-core-draft/personas.js').read()
facts['oak_human_tunics'] = re.findall(r"'(#[0-9a-f]{6})'", re.search(r'TUNICS = \[(.*?)\]', oakdef).group(1))
facts['oak_npc_tunics'] = re.findall(r"color: '(#[0-9a-f]{6})'", oakp)
facts['oak_npc_names'] = re.findall(r"name: '(\w+)'", oakp)
facts['oak_human_names'] = re.findall(r"'(\w+)'", re.search(r'HUMAN_NAMES = \[(.*?)\]', oakdef, re.S).group(1))
wizp = open('/root/wizard-village/personas.js').read(); wizd = open('/root/wizard-village/wizards.js').read()
facts['wiz_npc_names'] = re.findall(r"name: '(\w+)'", wizp)
facts['wiz_human_names'] = re.findall(r"'(\w+)'", re.search(r'HUMAN_NAMES = \[(.*?)\];', wizd, re.S).group(1))
facts['wiz_npc_dmg'] = [int(x) for x in re.findall(r'dmg: (\d+)', wizp)]
facts['wiz_human_dmg'] = int(re.search(r'initHuman\(e\) \{[^}]*e\.dmg = (\d+)', wizd).group(1))
facts['oak_npc_dmg'] = [int(x) for x in re.findall(r'dmg: (\d+)', oakp)]
facts['oak_human_dmg'] = int(re.search(r"kind: 'human'.*?dmg: (\d+)", sim, re.S).group(1))
for g, path in (('oak', '/root/living-village/lore.js'), ('wiz', '/root/wizard-village/lore.js')):
    lore = re.search(r'DEFAULT_LORE = `(.*?)`', open(path).read(), re.S).group(1)
    names = facts[f'{g}_npc_names']
    facts[f'{g}_lore_names'] = [n for n in names if re.search(r'\b' + n + r'\b', lore)]
json.dump(facts, open(os.path.join(DATA, 'code_facts.json'), 'w'), ensure_ascii=False, indent=1)
print('sessions', len(sessions), 'natural', [len(s['lines']) for s in nat], 'probe', [len(s['lines']) for s in probe], 'light', len(items))
