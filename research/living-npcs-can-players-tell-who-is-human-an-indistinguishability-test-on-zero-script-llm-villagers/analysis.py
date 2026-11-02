# Living NPCs, article 3: can players tell who is human? Offline analysis of the committed data.
# Reads data/*.json(l), writes results.json and charts/chart1..4.png. Deterministic (fixed seeds), no network.
import json, os, re, math, statistics as st
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); CH = os.path.join(HERE, 'charts')
os.makedirs(CH, exist_ok=True)
J = lambda f: json.load(open(os.path.join(DATA, f)))
INK, MID, LIGHT, PALE, GREEN, AMBER, RED = '#111', '#555', '#bbb', '#eee', '#2e7d32', '#f9a825', '#c62828'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MID, 'axes.labelcolor': INK,
                     'xtick.color': MID, 'ytick.color': MID, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
ts = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
R = {}

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]
def acc_block(correct, n):
    return {'n': n, 'correct': correct, 'accuracy': round(correct / n, 4) if n else None, 'ci95': wilson(correct, n),
            'p_vs_chance': float(f'{binomtest(correct, n, 0.5).pvalue:.4g}') if n else None}

# ---- 1. provenance of "human" traffic -----------------------------------------------------
S = J('sessions.json')
cls = Counter(s['class'] for s in S)
R['provenance'] = {
    'window_utc': ['2026-10-23', '2026-11-02'], 'sessions': len(S), 'by_class': dict(cls),
    'pseudonyms_by_class': {c: sorted({s['name'] for s in S if s['class'] == c}) for c in cls},
    'typed_or_spoken_lines_by_class': {c: sum(s['lines'] for s in S if s['class'] == c) for c in cls},
    'speech_to_text_lines_developer': sum(s['stt_lines'] for s in S if s['class'] == 'developer_manual'),
    'hits_or_casts_by_class': {c: sum(s['acts'] for s in S if s['class'] == c) for c in cls},
    'unknown_sessions': [{k: s[k] for k in ('game', 'name', 'join', 'lived_s', 'lines', 'acts', 'llm_calls_in_window', 'evidence')} for s in S if s['class'] == 'unknown'],
    'confirmed_independent_public_players': 0,
    'oak_live_v5_joins_after_v5_deploy': sum(1 for s in S if s['game'] == 'oak' and s['join'] >= '2026-10-24T02:15'),
    'sessions_table': [{k: s.get(k) for k in ('game', 'name', 'join', 'lived_s', 'lines', 'stt_lines', 'acts', 'class', 'evidence')} for s in S],
}

# ---- 2. facts read from the code ---------------------------------------------------------------
F = J('code_facts.json')
R['code'] = {
    'speed_tiles_per_s': {'human': F['core_speed_human'], 'npc_walk': F['core_speed_npc'], 'npc_run': F['core_speed_run']},
    'speed_ratio_human_to_npc_walk': round(F['core_speed_human'] / F['core_speed_npc'], 3),
    'speed_ratio_human_to_npc_run': round(F['core_speed_human'] / F['core_speed_run'], 3),
    'npc_typing': {'ms_per_char': F['type_ms_per_char'], 'min_ms': F['type_min_ms'], 'max_ms': F['type_max_ms']},
    'gate_ms': {'settle': F['gate_settle_ms'], 'min_gap': F['gate_min_gap_ms']}, 'human_min_gap_between_lines_ms': F['human_say_min_gap_ms'],
    'doze_ms': F['doze_ms'], 'npc_respawn_ms': F['npc_respawn_ms'], 'live_oakhollow_v5_same_constants': F['live_oak_same_constants'],
    'ui_relations_hint_en': F['ui_rel_hint_en'], 'ui_profile_hint_en': F['ui_no_out_en'],
    'oak_tunic_overlap': len(set(F['oak_human_tunics']) & set(F['oak_npc_tunics'])), 'oak_npc_tunics': len(F['oak_npc_tunics']), 'oak_human_tunics': len(F['oak_human_tunics']),
    'name_pool_overlap': {'oak': len(set(F['oak_human_names']) & set(F['oak_npc_names'])), 'wiz': len(set(F['wiz_human_names']) & set(F['wiz_npc_names']))},
    'npc_names_in_default_lore': {'oak': f"{len(F['oak_lore_names'])}/{len(F['oak_npc_names'])}", 'wiz': f"{len(F['wiz_lore_names'])}/{len(F['wiz_npc_names'])}"},
    'melee_dmg': {'oak_human': F['oak_human_dmg'], 'oak_npc': F['oak_npc_dmg'], 'oak_npcs_sharing_human_value': F['oak_npc_dmg'].count(F['oak_human_dmg']),
                  'wiz_human': F['wiz_human_dmg'], 'wiz_npc': F['wiz_npc_dmg'], 'wiz_npcs_sharing_human_value': F['wiz_npc_dmg'].count(F['wiz_human_dmg'])},
}

# ---- 3. the engine run, as a player's client saw it -------------------------------------------------
E = J('engine_run.json'); seats = set(E['seats'].values())
tracks = defaultdict(list)
for fr in E['frames']:
    for (eid, name, x, y, a, d, hp) in fr['e']: tracks[eid].append((fr['t'] / 1000, x, y, a, d, name))
def window_speeds(tr):                      # speed between consecutive state frames (about 100 ms apart) while moving
    out = []
    for a, b in zip(tr, tr[1:]):
        dt = b[0] - a[0]
        if 0.05 <= dt <= 0.2 and not a[4] and not b[4]:
            v = math.hypot(b[1] - a[1], b[2] - a[2]) / dt
            if 0.3 < v < 6: out.append(v)
    return out
sp = {'human_seat': [], 'npc': []}; ent_max = {}
for eid, tr in tracks.items():
    v = window_speeds(tr); k = 'human_seat' if eid in seats else 'npc'; sp[k] += v
    if len(v) >= 10: ent_max[eid] = (k, float(np.median(v)))
q = lambda a, p: round(float(np.percentile(a, p)), 3) if a else None
R['engine_run'] = {'duration_s': E['duration_s'], 'frames': len(E['frames']), 'entities_seen': len(tracks), 'human_seats': len(seats),
                   'speed_windows': {k: {'n': len(v), 'p10': q(v, 10), 'p50': q(v, 50), 'p90': q(v, 90), 'max': q(v, 100)} for k, v in sp.items()}}
thr = (F['core_speed_npc'] + F['core_speed_human']) / 2
mv = [(k, v) for k, v in ent_max.values()]
R['engine_run']['speed_windows_above_threshold_pct'] = {k: round(100 * sum(1 for x in v if x > thr) / len(v), 1) for k, v in sp.items()}
R['engine_run']['npc_ran_during_run'] = any(x > F['core_speed_npc'] + 0.05 for x in sp['npc'])
R['engine_run']['speed_oracle'] = {'rule': f'an entity whose median speed while moving exceeds {thr:.2f} tiles/s is human',
    'entities_that_moved': len(mv), 'correct': sum(1 for k, v in mv if (v > thr) == (k == 'human_seat')),
    'humans_moved': sum(1 for k, v in mv if k == 'human_seat'), 'npcs_moved': sum(1 for k, v in mv if k == 'npc'),
    'npcs_above_threshold': sum(1 for k, v in mv if k == 'npc' and v > thr)}
# activity icons
segs = defaultdict(list)
for eid, tr in tracks.items():
    cur, t0 = None, None
    for (t, x, y, a, d, name) in tr:
        if a != cur:
            if cur: segs[eid].append((cur, t0, t))
            cur, t0 = a, t
seg_counts = {'human_seat': Counter(), 'npc': Counter()}
for eid, ss in segs.items():
    for a, t0, t1 in ss: seg_counts['human_seat' if eid in seats else 'npc'][a] += 1
think = [t1 - t0 for eid, ss in segs.items() if eid not in seats for a, t0, t1 in ss if a == 'think']
R['engine_run']['icon_segments'] = {k: dict(v) for k, v in seg_counts.items()}
R['engine_run']['npc_think_icon_s'] = {'n': len(think), 'p50': q(think, 50), 'p90': q(think, 90)}
EV = [json.loads(l) for l in open(os.path.join(DATA, 'engine_events.jsonl'))]
llm = [e for e in EV if e['type'] == 'llm']
dec = [e for e in EV if e['type'] == 'decision']
R['engine_run']['llm_calls'] = len(llm); R['engine_run']['llm_cost_usd'] = round(sum(e.get('cost') or 0 for e in llm), 6)
R['engine_run']['llm_ms_p50'] = q([e['ms'] for e in llm if e.get('ms')], 50)
R['engine_run']['silent_decisions'] = sum(1 for e in dec if not e.get('say')); R['engine_run']['decisions'] = len(dec)
# write icon vs line length, heard lines only (client side)
id2name = {eid: tr[0][5] for eid, tr in tracks.items()}
writes = []
for c in E['chats']:
    if c['id'] in seats or c['faint']: continue
    cand = [(a, t0, t1) for a, t0, t1 in segs.get(c['id'], []) if a == 'write' and abs(t1 - c['t'] / 1000) <= 0.6]
    if cand: a, t0, t1 = cand[-1]; writes.append((len(c['text']), t1 - t0))
R['engine_run']['client_write_icon_vs_chars'] = [{'chars': n, 'icon_s': round(s, 2)} for n, s in writes]
# oracles from panels and names
rel_pairs = [(nul, eid in seats) for r in E['relations'] for nul, eid in zip(r['nullRows'], r['ids'])]
R['engine_run']['relations_oracle'] = {'snapshots': len(E['relations']), 'rows': len(rel_pairs), 'correct': sum(1 for nul, h in rel_pairs if nul == h)}
R['engine_run']['profile_oracle'] = {'profiles': len(E['profiles']), 'correct': sum(1 for p in E['profiles'] if (not p['hasOutgoing']) == (E['seats'].get(p['name']) is not None))}
nm = E['names']
R['engine_run']['name_oracle'] = {'npc_names_tried': sum(1 for x in nm if x['npc']), 'npc_names_refused': sum(1 for x in nm if x['npc'] and x['result'] == 'badname'),
    'free_player_names_tried': sum(1 for x in nm if not x['npc'] and not x['in_use_by_seat']), 'free_player_names_accepted': sum(1 for x in nm if not x['npc'] and not x['in_use_by_seat'] and x['result'] == 'you'),
    'name_in_use_by_player_refused': all(x['result'] == 'badname' for x in nm if x['in_use_by_seat'])}

# ---- 4. typing delay: decision -> spoken line, all zero-script logs ---------------------------------
P = J('decision_say_pairs.json')
expected = lambda n: max(F['type_min_ms'], min(F['type_max_ms'], n * F['type_ms_per_char'])) / 1000
resid = [(ts(p['say_ts']) - ts(p['decision_ts'])) - expected(p['chars']) for p in P]
gaps = [(p['chars'], ts(p['say_ts']) - ts(p['decision_ts'])) for p in P]
R['typing'] = {'pairs': len(P), 'runs': dict(Counter(p['run'] for p in P)),
               'residual_s': {'p50': q(resid, 50), 'p05': q(resid, 5), 'p95': q(resid, 95)},
               'within_0_25_s': sum(1 for r in resid if abs(r) <= 0.25), 'share_within_0_25_s': round(sum(1 for r in resid if abs(r) <= 0.25) / len(resid), 4),
               'clamped_min': sum(1 for p in P if p['chars'] * F['type_ms_per_char'] <= F['type_min_ms']), 'clamped_max': sum(1 for p in P if p['chars'] * F['type_ms_per_char'] >= F['type_max_ms'])}
x = np.array([g[0] for g in gaps if F['type_min_ms'] < g[0] * F['type_ms_per_char'] < F['type_max_ms']], float)
y = np.array([g[1] for g in gaps if F['type_min_ms'] < g[0] * F['type_ms_per_char'] < F['type_max_ms']], float)
R['typing']['median_abs_residual_s'] = round(float(np.median(np.abs(resid))), 4)
R['typing']['outliers_over_0_25_s'] = sorted(round(r, 2) for r in resid if abs(r) > 0.25)
R['typing']['unclamped_lines'] = len(x)
R['typing']['unclamped_slope_ms_per_char_median'] = round(float(np.median(y / x)) * 1000, 2)

# ---- 5. style and timing of each kind of speaker -------------------------------------------------------
NAT, PRB = J('natural_scenes.json'), J('probe_scenes.json')
items = {i['light_index']: i for i in J('light_items.json')}; reps = {r['light_index']: r for r in J('npc_replies.json')['replies']}
def style(lines):
    lines = [l for l in lines if l.strip()]
    n = len(lines)
    return {'lines': n, 'chars_p50': q([len(l) for l in lines], 50), 'words_mean': round(st.mean(len(l.split()) for l in lines), 2),
            'starts_lowercase_pct': round(100 * sum(1 for l in lines if l[0].islower()) / n, 1),
            'no_end_punct_pct': round(100 * sum(1 for l in lines if not re.search(r'[.!?…"\')*]$', l.strip())) / n, 1),
            'question_pct': round(100 * sum(1 for l in lines if '?' in l) / n, 1),
            'addresses_by_name_pct': None}
nat_h = [l['text'] for s in NAT for l in s['lines'] if l['kind'] == 'human']
nat_n = [l['text'] for s in NAT for l in s['lines'] if l['kind'] == 'npc']
prb_h = [l['text'] for s in PRB for l in s['lines'] if l['kind'] == 'human']
prb_n = [l['text'] for s in PRB for l in s['lines'] if l['kind'] == 'npc']
R['style'] = {'developer_players_live': style(nat_h), 'npc_live_earlier_builds': style(nat_n), 'template_players_probe': style(prb_h),
              'npc_zero_script_probe': style(prb_n), 'light_human_replies': style([items[k]['human_reply'] for k in items if reps[k]['npc_reply']]),
              'engine_npc_replies_light': style([reps[k]['npc_reply'] for k in items if reps[k]['npc_reply']])}
for k in R['style']: R['style'][k].pop('addresses_by_name_pct')
# reply latency: X addressed Y, Y answers X within 60 s
def latencies(scenes):
    out = defaultdict(list)
    for s in scenes:
        L = s['lines']
        for i, l in enumerate(L):
            if not l['to']: continue
            for m in L[i + 1:]:
                dt = ts(m['ts']) - ts(l['ts'])
                if dt > 60: break
                if m['from'] == l['to'] and m['to'] == l['from']: out[m['kind']].append(dt); break
                if m['from'] == l['from'] and m['to'] == l['to']: break
    return out
ln, lp = latencies(NAT), latencies(PRB)
R['reply_latency_s'] = {'developer_players_live': {'n': len(ln['human']), 'p50': q(ln['human'], 50), 'p10': q(ln['human'], 10), 'p90': q(ln['human'], 90)},
                        'npc_live_earlier_builds': {'n': len(ln['npc']), 'p50': q(ln['npc'], 50), 'p10': q(ln['npc'], 10), 'p90': q(ln['npc'], 90)},
                        'npc_zero_script_probe': {'n': len(lp['npc']), 'p50': q(lp['npc'], 50), 'p10': q(lp['npc'], 10), 'p90': q(lp['npc'], 90)},
                        'gate_plus_min_typing_s': round(F['gate_settle_ms'] / 1000 + F['type_min_ms'] / 1000, 2)}
fast = F['gate_settle_ms'] / 1000 + F['type_min_ms'] / 1000
R['reply_latency_s']['share_under_gate_plus_min_typing'] = {'developer_players_live': round(sum(1 for v in ln['human'] if v < fast) / len(ln['human']), 4),
    'npc_zero_script_probe': round(sum(1 for v in lp['npc'] if v < fast) / len(lp['npc']), 4),
    'note': 'an NPC can answer faster than the gate when it had already decided to speak to that person before the line arrived'}

# ---- 6. LLM judges ------------------------------------------------------------------------------------------
JD = J('judgments.json')
def verdict(r):
    p = r.get('parsed')
    if p and p.get('verdict') in ('human', 'ai'): return p['verdict'], p.get('confidence')
    m = re.search(r'"verdict"\s*:\s*"(human|ai)"', r.get('raw') or '')
    c = re.search(r'"confidence"\s*:\s*(\d+)', r.get('raw') or '')
    return (m.group(1), int(c.group(1)) if c else None) if m else (None, None)
SHORT = {'google/gemini-2.5-flash-lite': 'gemini-2.5-flash-lite', 'mistralai/mistral-small-3.2-24b-instruct': 'mistral-small-3.2',
         'openai/gpt-4.1-mini': 'gpt-4.1-mini', 'anthropic/claude-sonnet-4.5': 'claude-sonnet-4.5'}
t1 = defaultdict(list)
for r in JD['T1']:
    v, c = verdict(r); t1[r['judge']].append((r['light_index'], r['source'], v, c))
T1 = {}
votes = defaultdict(list)
for jm, rows in t1.items():
    ok = [r for r in rows if r[2]]
    corr = sum(1 for _, s, v, _ in ok if (v == 'human') == (s == 'human'))
    hh = [r for r in ok if r[1] == 'human']; nn = [r for r in ok if r[1] == 'npc']
    scores = [((c or 75) if v == 'human' else 100 - (c or 75)) for _, s, v, c in ok]
    T1[SHORT[jm]] = {**acc_block(corr, len(ok)), 'unanswered': len(rows) - len(ok),
        'recovered_from_truncated_json': sum(1 for r in JD['T1'] if r['judge'] == jm and not r.get('parsed') and verdict(r)[0]),
        'human_replies_called_human_pct': round(100 * sum(1 for r in hh if r[2] == 'human') / len(hh), 1),
        'npc_replies_called_human_pct': round(100 * sum(1 for r in nn if r[2] == 'human') / len(nn), 1),
        'auc': round(roc_auc_score([1 if s == 'human' else 0 for _, s, _, _ in ok], scores), 4)}
    for k, s, v, c in ok: votes[(k, s)].append(v)
maj = [(s, Counter(v).most_common(1)[0][0] if Counter(v)['human'] != Counter(v)['ai'] else 'ai') for (k, s), v in votes.items() if len(v) == 4]
T1['majority_of_4'] = {**acc_block(sum(1 for s, v in maj if (v == 'human') == (s == 'human')), len(maj)), 'tie_rule': 'tie counts as ai',
    'human_replies_called_human_pct': round(100 * sum(1 for s, v in maj if s == 'human' and v == 'human') / sum(1 for s, v in maj if s == 'human'), 1),
    'npc_replies_called_human_pct': round(100 * sum(1 for s, v in maj if s == 'npc' and v == 'human') / sum(1 for s, v in maj if s == 'npc'), 1)}
cues = Counter()
for r in JD['T1']:
    cue = ((r.get('parsed') or {}).get('cue') or '').lower()
    for w, lab in (('formal', 'formal'), ('generic', 'generic'), ('typo', 'typos'), ('grammar', 'grammar'), ('casual', 'casual'), ('informal', 'informal'),
                   ('short', 'short/brief'), ('brief', 'short/brief'), ('emotion', 'emotion'), ('repet', 'repetition'), ('context', 'context fit')):
        if w in cue: cues[lab] += 1
R['judges_T1'] = {'items': len(JD['T1']) // 4, 'pairs': len(JD['T1']) // 8, 'per_judge': T1, 'cue_words_in_explanations': dict(cues.most_common())}
# T2 scenes
t2 = []
for r in JD['T2']:
    sp_ = ((r.get('parsed') or {}).get('speakers') or {})
    for name, truth in r['truth'].items():
        v = (sp_.get(name) or {}).get('verdict')
        t2.append({'judge': SHORT[r['judge']], 'scene': r['scene'], 'kind': r['scene_kind'], 'speaker': name, 'truth': truth, 'verdict': v})
def t2_block(rows):
    rows = [r for r in rows if r['verdict'] in ('human', 'ai')]
    h = [r for r in rows if r['truth'] == 'human']; n = [r for r in rows if r['truth'] == 'npc']
    return {'speaker_judgments': len(rows), **acc_block(sum(1 for r in rows if (r['verdict'] == 'human') == (r['truth'] == 'human')), len(rows)),
            'humans_judged': len(h), 'humans_called_human': sum(1 for r in h if r['verdict'] == 'human'),
            'npcs_judged': len(n), 'npcs_called_human': sum(1 for r in n if r['verdict'] == 'human'),
            'npcs_called_human_pct': round(100 * sum(1 for r in n if r['verdict'] == 'human') / len(n), 1) if n else None}
def probe_block(rows):
    rows = [r for r in rows if r['verdict'] in ('human', 'ai')]
    h = [r for r in rows if r['truth'] == 'human']; n = [r for r in rows if r['truth'] == 'npc']
    return {'template_players_judged': len(h), 'template_players_called_human': sum(1 for r in h if r['verdict'] == 'human'),
            'npcs_judged': len(n), 'npcs_called_human': sum(1 for r in n if r['verdict'] == 'human'),
            'npcs_called_human_pct': round(100 * sum(1 for r in n if r['verdict'] == 'human') / len(n), 1) if n else None}
R['judges_T2'] = {'natural': {'all': t2_block([r for r in t2 if r['kind'] == 'natural']), **{j: t2_block([r for r in t2 if r['kind'] == 'natural' and r['judge'] == j]) for j in SHORT.values()}},
                  'probe': {j: probe_block([r for r in t2 if r['kind'] == 'probe' and (j == 'all' or r['judge'] == j)]) for j in ['all'] + list(SHORT.values())},
                  'scenes': {s['id']: {'lines': len(s['lines']), 'speakers': len({l['from'] for l in s['lines']}), 'human_lines': sum(1 for l in s['lines'] if l['kind'] == 'human')} for s in NAT + PRB},
                  'cues': {f"{SHORT[r['judge']]}/{r['scene']}": (r.get('parsed') or {}).get('cue') for r in JD['T2']}}

# ---- 7. feature classifier on the matched replies ---------------------------------------------------------
CONTR = re.compile(r"\b(dont|im|cant|wont|didnt|thats|ive|youre|isnt|doesnt|lets|whats|ill)\b", re.I)
def feats(reply, ctx_last):
    w = reply.split(); A = set(re.findall(r'\w+', reply.lower())); B = set(re.findall(r'\w+', ctx_last.lower()))
    return [len(reply), len(w), np.mean([len(t) for t in w]) if w else 0, sum(reply.count(c) for c in ',.;:!?') / max(1, len(reply)),
            reply[:1].islower(), bool(re.search(r'\bi\b', reply)), '...' in reply or '..' in reply, '*' in reply, '!' in reply, '?' in reply,
            bool(CONTR.search(reply)), '  ' in reply or ' ,' in reply or ' .' in reply, not re.search(r'[.!?"\')*]$', reply.strip()),
            len(A & B) / max(1, len(A | B))]
FN = ['chars', 'words', 'mean_word_len', 'punct_per_char', 'starts_lowercase', 'lowercase_i', 'ellipsis', 'asterisk', 'exclamation', 'question',
      'contraction_without_apostrophe', 'odd_spacing', 'no_end_punct', 'word_overlap_with_partner_line']
X, Y, G = [], [], []
for k, it in items.items():
    if not reps[k]['npc_reply']: continue
    for txt, lab in ((it['human_reply'], 1), (reps[k]['npc_reply'], 0)):
        X.append(feats(txt, it['context'][-1])); Y.append(lab); G.append(k)
X, Y, G = np.array(X, float), np.array(Y), np.array(G)
pred, prob = np.zeros(len(Y)), np.zeros(len(Y))
for tr, te in GroupKFold(5).split(X, Y, G):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], Y[tr])
    pred[te] = m.predict(X[te]); prob[te] = m.predict_proba(X[te])[:, 1]
full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, Y)
coef = sorted(zip(FN, full[-1].coef_[0]), key=lambda t: -abs(t[1]))
only_len = np.zeros(len(Y))
for tr, te in GroupKFold(5).split(X, Y, G):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr][:, :1], Y[tr]); only_len[te] = m.predict(X[te][:, :1])
R['classifier'] = {'model': 'logistic regression on 14 surface features, 5-fold cross-validation grouped by dialogue', 'n': int(len(Y)),
                   **acc_block(int((pred == Y).sum()), len(Y)), 'auc': round(float(roc_auc_score(Y, prob)), 4),
                   'length_only': acc_block(int((only_len == Y).sum()), len(Y)),
                   'top_coefficients_positive_means_human': [{'feature': f, 'coef': round(float(c), 3)} for f, c in coef[:6]]}

# ---- 8. spend ---------------------------------------------------------------------------------------------------
sb, sa = J('spend_before.json'), J('spend_after.json')
R['spend_usd'] = {'key_usage_before': sb['key_usage_before_usd'], 'key_usage_after': sa['key_usage_after_usd'],
                  'delta_all_experiments': round(sa['key_usage_after_usd'] - sb['key_usage_before_usd'], 6),
                  'judges_billed': round(JD['spent_usd'], 6), 'npc_replies_billed': round(J('npc_replies.json')['cost_usd'], 6),
                  'engine_run_billed': R['engine_run']['llm_cost_usd'], 'cap': 1.50,
                  'note': 'delta read from the key usage counter before and after; includes 5 smoke-test judge calls'}

# ---- charts -----------------------------------------------------------------------------------------------------
def src(fig, text): fig.text(0.01, 0.01, text, fontsize=8.5, color=MID, ha='left', va='bottom')
# chart 1: judge accuracy on matched replies
names = list(SHORT.values()) + ['majority_of_4']
labs = names + ['feature classifier']
vals = [T1[n]['accuracy'] for n in names] + [R['classifier']['accuracy']]
cis = [T1[n]['ci95'] for n in names] + [R['classifier']['ci95']]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
yy = np.arange(len(labs))[::-1]
ax.barh(yy, [v * 100 for v in vals], color=[INK if i < 4 else MID for i in range(len(labs))], height=0.6)
ax.errorbar([v * 100 for v in vals], yy, xerr=[[100 * (v - c[0]) for v, c in zip(vals, cis)], [100 * (c[1] - v) for v, c in zip(vals, cis)]], fmt='none', ecolor=LIGHT, capsize=4, lw=1.5)
ax.axvline(50, color=RED, lw=1.2, ls='--'); ax.text(50.5, yy[0] + 0.45, 'chance', color=RED, fontsize=9)
for v, c, y_ in zip(vals, cis, yy): ax.text(c[1] * 100 + 1.5, y_ - 0.12, f'{v * 100:.1f}%', fontsize=9.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(['majority of 4 judges' if l == 'majority_of_4' else l for l in labs])
ax.set_xlim(0, 100); ax.set_xlabel('accuracy at telling a human reply from an NPC reply (%), with 95% Wilson interval')
ax.set_title(f'Who wrote this reply? Same LIGHT dialogue context, human vs engine NPC ({R["judges_T1"]["items"]} replies)', loc='left', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
src(fig, 'Source: our experiment; human replies from LIGHT test dialogues (Urbanek et al. 2019), NPC replies from the unchanged engine mind (gemini-2.5-flash-lite)')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(CH, 'chart1.png')); plt.close(fig)

# chart 2: how well each observable channel identifies the humans
eo = R['engine_run']
chan = [
    ('relations matrix: empty row', eo['relations_oracle']['correct'] / eo['relations_oracle']['rows'], f"engine run, {eo['relations_oracle']['rows']} rows"),
    ('profile: no outgoing opinions', eo['profile_oracle']['correct'] / eo['profile_oracle']['profiles'], f"engine run, {eo['profile_oracle']['profiles']} profiles"),
    ('name refused at sign-up', (eo['name_oracle']['npc_names_refused'] + eo['name_oracle']['free_player_names_accepted']) / (eo['name_oracle']['npc_names_tried'] + eo['name_oracle']['free_player_names_tried']), 'engine run, 15 names'),
    ('Oakhollow tunic outside player palette', 1 - R['code']['oak_tunic_overlap'] / R['code']['oak_npc_tunics'], 'code, 10 NPC colours'),
    ('median moving speed above 2.75 tiles/s', eo['speed_oracle']['correct'] / eo['speed_oracle']['entities_that_moved'], f"engine run, {eo['speed_oracle']['entities_that_moved']} entities that moved"),
    ('NPC lines on the 90 ms/char typing rule', R['typing']['share_within_0_25_s'], f"{R['typing']['pairs']} NPC lines, within 0.25 s"),
    ('live scenes, text and times, best judge', R['judges_T2']['natural']['claude-sonnet-4.5']['accuracy'], '144 judgments pooled: ' + f"{R['judges_T2']['natural']['all']['accuracy'] * 100:.1f}%"),
    ('matched replies, feature classifier', R['classifier']['accuracy'], 'surface features, cross-validated'),
    ('matched replies, best LLM judge', max(T1[n]['accuracy'] for n in SHORT.values()), 'claude-sonnet-4.5'),
]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
yy = np.arange(len(chan))[::-1]
cols = [RED if v >= 0.95 else AMBER if v >= 0.7 else MID for _, v, _ in chan]
ax.barh(yy, [v * 100 for _, v, _ in chan], color=cols, height=0.6)
for (l, v, s), y_ in zip(chan, yy):
    if v >= 0.9: ax.text(v * 100 - 1, y_ - 0.12, f'{v * 100:.1f}%  ({s})', fontsize=8.5, color='white', ha='right')
    else: ax.text(v * 100 + 1, y_ - 0.12, f'{v * 100:.1f}%  ({s})', fontsize=8.5, color=INK)
ax.axvline(50, color=MID, lw=1, ls='--')
ax.set_yticks(yy); ax.set_yticklabels([c[0] for c in chan]); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('share of cases in which the channel tells human from NPC correctly (%)')
ax.set_title('What gives the humans away: accuracy of each observable channel', loc='left', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
src(fig, 'Source: our measurements on a private Quarthaven copy (shared engine), article-2 probe logs and the judge experiments. Red: at least 95%.')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(CH, 'chart2.png')); plt.close(fig)

# chart 3: NPC typing delay vs line length
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
gx = [g[0] for g in gaps]; gy = [g[1] for g in gaps]
ax.scatter(gx, gy, s=12, color=INK, alpha=0.55, label=f'NPC line, decision to speech ({len(gaps)} lines, server log)')
if writes: ax.scatter([w[0] for w in writes], [w[1] for w in writes], s=34, facecolors='none', edgecolors=RED, label=f'writing icon seen by a player ({len(writes)} lines)')
xs = np.arange(0, max(gx) + 5); ax.plot(xs, [expected(v) for v in xs], color=AMBER, lw=2, label='engine rule: 90 ms per character, clamped to 1.5-9 s')
ax.set_ylim(0, 12)
off = [g for g in gaps if g[1] > 12]
if off: ax.text(max(gx) * 0.55, 11.2, f'{len(off)} lines delayed {min(g[1] for g in off):.0f}-{max(g[1] for g in off):.0f} s are off the scale', fontsize=9, color=MID)
ax.set_xlabel('length of the spoken line (characters)'); ax.set_ylabel('time from decision to speech (s)')
ax.set_title('Every NPC types at exactly the same speed', loc='left', fontsize=12); ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
src(fig, 'Source: zero-script logs of article 2 (24 Oct 2026) and this article\'s engine run (2 Nov 2026); our analysis')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(CH, 'chart3.png')); plt.close(fig)

# chart 4: speed distributions
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
bins = np.arange(0.3, 4.2, 0.1)
ax.hist(sp['npc'], bins=bins, color=MID, alpha=0.85, label=f'NPC mages ({len(sp["npc"])} frame pairs)')
ax.hist(sp['human_seat'], bins=bins, color=RED, alpha=0.75, label=f'player seats ({len(sp["human_seat"])} frame pairs)')
for v, l in ((F['core_speed_npc'], 'NPC walk 2.3'), (F['core_speed_run'], 'NPC run 3.1'), (F['core_speed_human'], 'player 3.2')):
    ax.axvline(v, color=INK, lw=0.8, ls=':'); ax.text(v + 0.02, ax.get_ylim()[1] * 0.92, l, fontsize=8.5, rotation=90, va='top')
ax.set_xlabel('speed between consecutive state frames while moving (tiles per second)'); ax.set_ylabel('number of frame pairs (log scale)'); ax.set_yscale('log')
ax.set_title('Players move faster than walking NPCs', loc='left', fontsize=12); ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
src(fig, 'Source: state frames received by a player client on a private Quarthaven copy, 150 s, 2 Nov 2026; our analysis')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(CH, 'chart4.png')); plt.close(fig)

R['charts'] = {'chart1': 'judge accuracy on matched replies', 'chart2': 'accuracy per observable channel', 'chart3': 'NPC typing delay vs line length', 'chart4': 'speed distributions'}
R['chart2_values'] = [{'channel': l, 'accuracy': round(v, 4), 'basis': s} for l, v, s in chan]
json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ('judges_T1', 'classifier', 'typing', 'reply_latency_s', 'spend_usd')}, indent=1, ensure_ascii=False)[:6000])
