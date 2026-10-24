"""Living NPCs, article 2: fully LLM-driven characters on one engine, measured in two games.

Reads only committed data under data/ and writes results.json and charts/chart1..4.png.
  data/raw/<run>/events-*.jsonl   event logs of three private test servers (2026-10-24, UTC)
  data/raw/<run>/stimuli.jsonl    what the probe players did and saw, with timestamps
  data/oak_live.json              anonymised extract of the live Oakhollow logs (2026-10-23/24, builds before v5)
  data/loc.json                   lines of code per layer (engine core, each game)
Runs: wiz-gemini  = Quarthaven, google/gemini-2.5-flash-lite (engine default, mistral-small-3.2 as fallback)
      wiz-mistral = Quarthaven, mistralai/mistral-small-3.2-24b-instruct chosen by the players
      oak-gemini  = Oakhollow ported onto the shared engine (port draft), gemini-2.5-flash-lite
Deterministic: no randomness, no network.
"""
import glob, json, os, re, statistics as st
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, CH = os.path.join(HERE, 'data'), os.path.join(HERE, 'charts')
os.makedirs(CH, exist_ok=True)
RUNS = {'wiz-gemini': ('Quarthaven', 'gemini-2.5-flash-lite'), 'wiz-mistral': ('Quarthaven', 'mistral-small-3.2'),
        'oak-gemini': ('Oakhollow (core)', 'gemini-2.5-flash-lite')}
INK, MID, LIGHT, GRID = '#111111', '#555555', '#bbbbbb', '#dddddd'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MID, 'axes.labelcolor': INK,
                     'xtick.color': MID, 'ytick.color': MID, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

def T(s): return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
def pct(xs, q):
    xs = sorted(xs)
    if not xs: return None
    k = (len(xs) - 1) * q; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)
def r(x, n=3): return None if x is None else round(x, n)
def load(run):
    ev = [json.loads(l) for f in sorted(glob.glob(os.path.join(DATA, 'raw', run, 'events-*.jsonl'))) for l in open(f)]
    st_ = [json.loads(l) for l in open(os.path.join(DATA, 'raw', run, 'stimuli.jsonl'))]
    for e in ev + st_: e['t'] = T(e['ts'])
    return sorted(ev, key=lambda e: e['t']), sorted(st_, key=lambda e: e['t'])

PLACE_ID = {'Quarthaven': 'moonpool', 'Oakhollow (core)': 'well'}
HOSTILE_SPELLS = {'bolt', 'kindle', 'gust'}
def is_hostile_to(d, name):
    """d: parsed decision from an llm row. True if it attacks `name` (duel, or one hostile spell at them)."""
    if not d: return False
    tgt = (d.get('target') or '').lower(); to = (d.get('to') or '').lower()
    aimed = name.lower() in (tgt, to) or (tgt == '' and to == name.lower())
    if d.get('action') == 'attack' and aimed: return True
    sp = (d.get('extra') or {}).get('spell')
    return d.get('action') == 'cast' and sp in HOSTILE_SPELLS and aimed

res = {'runs': {}, 'meta': {'generated_from': 'data/ (committed)', 'test_date_utc': '2026-10-24'}}
all_lat = {}
stim_table = {}
witness_rows = []
e2e = {}

for run, (game, model) in RUNS.items():
    ev, sti = load(run)
    llm = [e for e in ev if e['type'] == 'llm']
    snaps = {e['seq']: e for e in ev if e['type'] == 'probe_snap'}
    ok = [e for e in llm if e.get('ok')]
    ms = [e['ms'] for e in ok]; tt = [e['ttfb'] for e in ok if e.get('ttfb') is not None]
    served = {}
    for e in ok: served[e['model']] = served.get(e['model'], 0) + 1
    cost = sum(e.get('cost') or 0 for e in llm)
    dec = [e for e in ev if e['type'] == 'decision']
    acts = {}
    for d in dec: acts[d['action']] = acts.get(d['action'], 0) + 1
    span = (max(e['t'] for e in ev) - min(e['t'] for e in ev))
    npc_say = [e for e in ev if e['type'] == 'say' and e['kind'] == 'npc']
    R = {'game': game, 'model': model, 'llm_calls': len(llm), 'ok': len(ok), 'ok_rate': r(len(ok) / len(llm), 4),
         'errors': sum(1 for e in llm if not e.get('ok') and (e.get('err') or e.get('status'))),
         'parse_fail': sum(1 for e in llm if e.get('ok') is False and not e.get('err') and not e.get('status')),
         'ms_p50': r(pct(ms, .5), 1), 'ms_p90': r(pct(ms, .9), 1), 'ms_max': max(ms), 'ttfb_p50': r(pct(tt, .5), 1), 'ttfb_p90': r(pct(tt, .9), 1),
         'tin_p50': pct([e['tin'] for e in ok], .5), 'tout_p50': pct([e['tout'] for e in ok], .5),
         'cost_usd': r(cost, 6), 'cost_per_1000_usd': r(cost / len(ok) * 1000, 4), 'served_by': served,
         'decisions': len(dec), 'actions': dict(sorted(acts.items(), key=lambda x: -x[1])),
         'share_with_speech': r(sum(1 for d in dec if d.get('say')) / len(dec), 4),
         'repeat_dropped': sum(1 for e in ev if e['type'] == 'repeat_dropped'), 'npc_lines_spoken': len(npc_say),
         'casts_npc': sum(1 for e in ev if e['type'] == 'cast' and e['kind'] == 'npc'),
         'hits': {k: sum(1 for e in ev if e['type'] == 'hit' and f"{e['fk']}>{e['tk']}" == k) for k in ('human>npc', 'npc>human', 'npc>npc')},
         'deaths': {k: sum(1 for e in ev if e['type'] == 'death' and e['kind'] == k) for k in ('human', 'npc')},
         'log_span_s': round(span, 1), 'decisions_per_npc_minute': None}
    npcs = {e['npc'] for e in llm}
    R['npcs_deciding'] = len(npcs)
    R['decisions_per_npc_minute'] = r(len(llm) / len(npcs) / (span / 60), 3)
    all_lat[run] = ms

    # ---- zero-script audit: every NPC line, cast and blow traced back to an LLM decision ----
    said = [(d['t'], d['npc'], d.get('say')) for d in dec if d.get('say')]
    matched = sum(1 for s in npc_say if any(n == s['from'] and txt == s['text'] and t <= s['t'] + 0.01 for t, n, txt in said))
    casts = [e for e in ev if e['type'] == 'cast' and e['kind'] == 'npc']
    def plan_at(npc, t):
        """the last plan-setting decision of npc before t ('continue' keeps the previous plan)"""
        last = None
        for d in dec:
            if d['t'] > t: break
            if d['npc'] == npc and d['action'] != 'continue': last = d
        return last
    c_direct = c_duel = c_other = 0
    for c in casts:
        p = plan_at(c['who'], c['t'] + 0.01)
        if p and p['action'] == 'cast' and abs(p['t'] - c['t']) < 1 and (p.get('extra') or {}).get('spell') in (c['spell'], 'sig'): c_direct += 1
        elif p and p['action'] in ('attack', 'cast'): c_duel += 1   # duel plan from an attack decision; 'cast' keeps a fight going
        else: c_other += 1
    blows = [e for e in ev if e['type'] == 'hit' and e['fk'] == 'npc']
    b_ok = 0
    for b in blows:
        p = plan_at(b['from'], b['t'] + 0.01)
        if p and p['action'] in ('attack', 'cast'): b_ok += 1
    R['audit'] = {'npc_lines': len(npc_say), 'npc_lines_matched_to_decision': matched,
                  'npc_casts': len(casts), 'casts_from_cast_decision': c_direct, 'casts_inside_duel_plan': c_duel, 'casts_unattributed': c_other,
                  'npc_hits': len(blows), 'hits_inside_attack_or_cast_plan': b_ok}

    # ---- stimulus responses ----
    rows = []
    stims = [s for s in sti if s['kind'] == 'stim']
    npc_names = {e['npc'] for e in llm}
    for s in stims:
        bot, tgt = s['bot'], (s.get('victim') if s['stim'] == 'witness_attack' else s['target'])
        human_victim = tgt not in npc_names          # a phase-2 probe that happened to pick another probe player
        nxt = [s2['t'] for s2 in stims if s2['bot'] == bot and s2['t'] > s['t']]
        horizon = min([s['t'] + 30] + nxt)
        # the target's first decision whose perception contains the stimulus (the bot's name)
        tl = [e for e in llm if e['npc'] == tgt and s['t'] < e['t'] <= horizon + 15 and e.get('ok')
              and any(bot in x for x in snaps.get(e['seq'], {}).get('events', []))]
        first = tl[0] if tl else None
        d = first['decision'] if first else None
        win = [e for e in llm if e['npc'] == tgt and s['t'] < e['t'] <= horizon and e.get('ok')]
        row = {'bot': bot, 'npc': tgt, 'stim': s['stim'], 'phase': s.get('phase', 1), 'decided': bool(d),
               'action': d['action'] if d else None, 'spell': (d.get('extra') or {}).get('spell') if d else None,
               'said': bool(d and d.get('say')), 'delta_to_bot': (d.get('opinion_changes') or {}).get(bot) if d else None,
               'hostile_first': is_hostile_to(d, bot), 'hostile_any': any(is_hostile_to(e['decision'], bot) for e in win),
               'decide_lag_s': r(first['t'] - s['t'], 2) if first else None,
               'n_decisions_in_window': len(win),
               'any_follow_bot': any(e['decision']['action'] == 'follow' and (e['decision'].get('target') or e['decision'].get('to') or '').lower() in (bot.lower(), '') for e in win),
               'any_go_place': any(e['decision']['action'] == 'go' and PLACE_ID[game] in (e['decision'].get('target') or '').lower() for e in win),
               'defensive_any': any(e['decision']['action'] == 'flee' or (e['decision'].get('extra') or {}).get('spell') == 'ward' for e in win)}
        # reply time seen by the player: stimulus to the first line the target speaks to them
        ch = [c for c in sti if c['kind'] == 'chat' and c['bot'] == bot and c['from'] == tgt and s['t'] < c['t'] <= horizon]
        row['reply_s'] = r(ch[0]['t'] - s['t'], 2) if ch else None
        if s['stim'] == 'follow':
            fc = [c for c in sti if c['kind'] == 'follow_check' and c['bot'] == bot and c['t'] > s['t']]
            row['follow_dist_after_walk'] = fc[0]['dist'] if fc else None
        if s['stim'] == 'goto':
            gc = [c for c in sti if c['kind'] == 'goto_check' and c['bot'] == bot and c['t'] > s['t']]
            row['dist_to_place_after'] = gc[0]['dist_to_place'] if gc else None
        if not human_victim: rows.append(row)
        # witnesses of an attack (bolt, blow, or the dedicated third-party attack)
        if s['stim'] in ('bolt', 'strike', 'witness_attack'):
            for e in llm:
                if e['npc'] == tgt or not (s['t'] < e['t'] <= s['t'] + 20) or not e.get('ok'): continue
                evs = snaps.get(e['seq'], {}).get('events', [])
                seen = any(x.startswith(f'You see {bot}') and tgt in x for x in evs)
                heard = (not seen) and any(x.startswith('You hear') for x in evs) and not any(bot in x for x in evs)
                if not (seen or heard): continue
                if any(w['run'] == run and w['npc'] == e['npc'] and w['bot'] == bot and w['victim'] == tgt for w in witness_rows): continue
                dd = e['decision']
                said_ = bool(dd.get('say')); act = dd['action']
                cat = ('attacks attacker' if is_hostile_to(dd, bot) else 'flees' if act == 'flee'
                       else 'helps victim' if ((dd.get('extra') or {}).get('spell') == 'mend' or (act in ('go', 'follow') and (dd.get('target') or '') == tgt))
                       else 'speaks only' if said_ and act in ('continue', 'stay') else 'moves' if act not in ('continue', 'stay') else 'no visible reaction')
                ment = any(n in ((dd.get('say') or '') + ' ' + (dd.get('thought') or '')) for n in (bot, tgt))
                witness_rows.append({'run': run, 'stim': s['stim'], 'bot': bot, 'victim': tgt, 'npc': e['npc'], 'channel': 'seen' if seen else 'heard',
                                     'category': cat, 'said': said_, 'action': act, 'mentions_attacker_or_victim': ment,
                                     'delta_to_bot': (dd.get('opinion_changes') or {}).get(bot)})
    stim_table[run] = rows
    e2e[run] = [x['reply_s'] for x in rows if x['reply_s'] is not None and x['stim'] in ('greet', 'follow', 'goto', 'insult', 'apology')]
    R['reply_s_p50'] = r(pct(e2e[run], .5), 2); R['reply_s_n'] = len(e2e[run])
    res['runs'][run] = R

# ---- per-stimulus summaries across runs ----
def summ(rows, stim):
    xs = [x for x in rows if x['stim'] == stim]
    if not xs: return {'n': 0}
    o = {'n': len(xs), 'decided': sum(x['decided'] for x in xs), 'actions': {}}
    for x in xs: o['actions'][str(x['action'])] = o['actions'].get(str(x['action']), 0) + 1
    o['said'] = sum(x['said'] for x in xs)
    if stim == 'follow':
        o['agreed_follow'] = sum(1 for x in xs if x['action'] == 'follow')
        o['follow_decision_within_window'] = sum(1 for x in xs if x['any_follow_bot'])
        o['physically_followed_within_3_tiles'] = sum(1 for x in xs if x.get('follow_dist_after_walk') is not None and x['follow_dist_after_walk'] <= 3)
    if stim == 'goto':
        o['chose_go'] = sum(1 for x in xs if x['action'] == 'go')
        o['go_to_place_decision_within_window'] = sum(1 for x in xs if x['any_go_place'])
        o['within_3_tiles_of_place_after_22s'] = sum(1 for x in xs if x.get('dist_to_place_after') is not None and x['dist_to_place_after'] <= 3)
    if stim in ('insult', 'apology', 'greet', 'mend'):
        ds = [x['delta_to_bot'] for x in xs if x['delta_to_bot'] is not None]
        o['delta_to_bot'] = ds; o['delta_mean'] = r(st.mean(ds), 2) if ds else None
    if stim in ('insult', 'bolt', 'strike', 'witness_attack'):
        o['hostile_first'] = sum(x['hostile_first'] for x in xs); o['hostile_within_window'] = sum(x['hostile_any'] for x in xs)
        o['defensive_within_window'] = sum(x['defensive_any'] and not x['hostile_any'] for x in xs)
    return o
STIMS = ['greet', 'follow', 'goto', 'insult', 'bolt', 'strike', 'mend', 'apology', 'witness_attack']
res['stimuli'] = {run: {s: summ(rows, s) for s in STIMS if any(x['stim'] == s for x in rows)} for run, rows in stim_table.items()}
res['stimulus_rows'] = stim_table
allrows = [x for rows in stim_table.values() for x in rows]
att = [x for x in allrows if x['stim'] in ('bolt', 'strike', 'witness_attack') and x['decided']]
ins = [x for x in allrows if x['stim'] == 'insult' and x['decided']]
fol = [x for x in allrows if x['stim'] == 'follow' and x['decided']]
res['pooled'] = {
    'attacks_on_npc_with_decision': len(att), 'victim_hostile_within_window': sum(x['hostile_any'] for x in att),
    'victim_retaliation_rate': r(sum(x['hostile_any'] for x in att) / len(att), 4) if att else None,
    'insults_with_decision': len(ins), 'insult_hostile_first': sum(x['hostile_first'] for x in ins),
    'insult_delta_mean': r(st.mean([x['delta_to_bot'] for x in ins if x['delta_to_bot'] is not None]), 2),
    'victim_defensive_only': sum(x['defensive_any'] and not x['hostile_any'] for x in att),
    'insult_hostile_within_window': sum(x['hostile_any'] for x in ins),
    'follow_decision_within_window': sum(x['any_follow_bot'] for x in fol),
    'goto_requests_with_decision': sum(1 for x in allrows if x['stim'] == 'goto' and x['decided']),
    'goto_decision_within_window': sum(x['any_go_place'] for x in allrows if x['stim'] == 'goto' and x['decided']),
    'goto_within_3_tiles_after_22s': sum(1 for x in allrows if x['stim'] == 'goto' and x.get('dist_to_place_after') is not None and x['dist_to_place_after'] <= 3),
    'follow_requests_with_decision': len(fol), 'follow_agreed': sum(1 for x in fol if x['action'] == 'follow'),
    'follow_physically_within_3_tiles': sum(1 for x in fol if x.get('follow_dist_after_walk') is not None and x['follow_dist_after_walk'] <= 3),
    'reply_s_p50_all': r(pct([v for vs in e2e.values() for v in vs], .5), 2), 'reply_s_n_all': sum(len(v) for v in e2e.values()),
    'stim_decided_lag_s_p50': r(pct([x['decide_lag_s'] for x in allrows if x['decide_lag_s'] is not None], .5), 2),
}
res['witness_rows'] = witness_rows
W = {}
for w in witness_rows: W[w['category']] = W.get(w['category'], 0) + 1
seenw = [w for w in witness_rows if w['channel'] == 'seen']
res['witness'] = {'n': len(witness_rows), 'seen': len(seenw), 'heard_only': len(witness_rows) - len(seenw), 'categories': W,
                  'visible_reaction_rate': r(sum(1 for w in witness_rows if w['category'] != 'no visible reaction') / len(witness_rows), 4) if witness_rows else None,
                  'seen_visible_reaction_rate': r(sum(1 for w in seenw if w['category'] != 'no visible reaction') / len(seenw), 4) if seenw else None,
                  'mention_rate': r(sum(w['mentions_attacker_or_victim'] for w in witness_rows) / len(witness_rows), 4) if witness_rows else None,
                  'attacks_attacker': W.get('attacks attacker', 0),
                  'llm_negative_delta_to_attacker': sum(1 for w in witness_rows if (w['delta_to_bot'] or 0) < 0),
                  'episodes': len({(w['run'], w['bot'], w['victim'], w['stim']) for w in witness_rows})}
aud = [R['audit'] for R in res['runs'].values()]
res['audit_total'] = {k: sum(a[k] for a in aud) for k in aud[0]}

# ---- E1: the live Oakhollow logs (builds before v5; v5 = build 7) ----
L = json.load(open(os.path.join(DATA, 'oak_live.json')))
live = {}
for b in sorted({x['build'] for x in L['llm']} | {x['build'] for x in L['join']}):
    calls = [x for x in L['llm'] if x['build'] == b]; okc = [x for x in calls if x['ok']]
    acts = {}
    for x in calls: acts[str(x['action'])] = acts.get(str(x['action']), 0) + 1
    live[b] = {'joins': sum(1 for x in L['join'] if x['build'] == b), 'llm_calls': len(calls), 'ok': len(okc),
               'ms_p50': r(pct([x['ms'] for x in okc], .5), 1), 'ttfb_p50': r(pct([x['ttfb'] for x in okc if x['ttfb']], .5), 1),
               'cost_usd': r(sum(x['cost'] or 0 for x in calls), 6), 'actions': acts,
               'hostile_reflex_events': sum(1 for x in L['hostile'] if x['build'] == b)}
pre = [x for x in L['llm'] if x['build'] < 7]; pok = [x for x in pre if x['ok']]
models = {}
for x in pok: models[x['model']] = models.get(x['model'], 0) + 1
hk = {}
for h in L['hit']: hk[f"{h['from_kind']}>{h['to_kind']}"] = hk.get(f"{h['from_kind']}>{h['to_kind']}", 0) + 1
dk = {}
for d in L['death']: dk[f"{d['by_kind']}>{d['kind']}"] = dk.get(f"{d['by_kind']}>{d['kind']}", 0) + 1
acts = {}
for x in pre: acts[str(x['action'])] = acts.get(str(x['action']), 0) + 1
res['live_oakhollow'] = {
    'builds': live, 'pre_v5': {'llm_calls': len(pre), 'ok': len(pok), 'ok_rate': r(len(pok) / len(pre), 4),
        'not_ok': len(pre) - len(pok), 'ms_p50': r(pct([x['ms'] for x in pok], .5), 1), 'ms_p90': r(pct([x['ms'] for x in pok], .9), 1),
        'ttfb_p50': r(pct([x['ttfb'] for x in pok if x['ttfb']], .5), 1), 'tin_p50': pct([x['tin'] for x in pok], .5), 'tout_p50': pct([x['tout'] for x in pok], .5),
        'cost_usd': r(sum(x['cost'] or 0 for x in pre), 6), 'cost_per_1000_usd': r(sum(x['cost'] or 0 for x in pre) / len(pok) * 1000, 4),
        'served_by': models, 'actions': dict(sorted(acts.items(), key=lambda x: -x[1])),
        'stay_share': r(acts.get('stay', 0) / len(pre), 4), 'hits_by_kind': hk, 'deaths_by_kind': dk,
        'hostile_reflex_events': len(L['hostile']), 'repeat_dropped': sum(1 for x in L['repeat_dropped'] if x['build'] < 7)},
    'v5': {'joins': sum(1 for x in L['join'] if x['build'] == 7), 'llm_calls': sum(1 for x in L['llm'] if x['build'] == 7)}}
all_lat['live-pre-v5'] = [x['ms'] for x in pok]

# ---- E3: lines of code ----
LOC = json.load(open(os.path.join(DATA, 'loc.json')))
core = LOC['core_server']['code'] + LOC['core_client']['code']
qv = LOC['quarthaven_server']['code'] + LOC['quarthaven_client']['code']
op = LOC['oakhollow_port_server']['code'] + LOC['oakhollow_port_client']['code']
osd = LOC['oakhollow_v5_standalone_server']['code'] + LOC['oakhollow_v5_standalone_client']['code']
res['loc'] = {'core_code_lines': core, 'core_server': LOC['core_server']['code'], 'core_client': LOC['core_client']['code'],
              'quarthaven_code_lines': qv, 'quarthaven_server': LOC['quarthaven_server']['code'], 'quarthaven_client': LOC['quarthaven_client']['code'],
              'oak_port_code_lines': op, 'oak_port_server': LOC['oakhollow_port_server']['code'], 'oak_port_client': LOC['oakhollow_port_client']['code'],
              'oak_standalone_code_lines': osd,
              'quarthaven_game_share': r(qv / (qv + core), 4), 'oak_port_game_share': r(op / (op + core), 4),
              'oak_port_vs_standalone_ratio': r(op / osd, 4),
              'wizards_js_code': LOC['quarthaven_server']['files']['wizard-village/wizards.js']['code'],
              'core_actions': 11, 'quarthaven_extra_actions': 1, 'quarthaven_extra_fields': 1, 'oak_extra_actions': 0}

# ---- spend (OpenRouter key usage before and after all probes, recorded in data/raw/spend_probe.json) ----
sp = json.load(open(os.path.join(DATA, 'raw', 'spend_probe.json')))
res['spend'] = {**sp, 'sum_logged_cost_usd': r(sum(R['cost_usd'] for R in res['runs'].values()), 6)}

RR = res['runs']
res['totals'] = {'llm_calls': sum(v['llm_calls'] for v in RR.values()), 'ok': sum(v['ok'] for v in RR.values()),
                 'npcs': sum(v['npcs_deciding'] for v in RR.values()),
                 'live_pre_v5_fallback_share': r(res['live_oakhollow']['pre_v5']['served_by'].get('mistralai/mistral-small-3.2-24b-instruct', 0) / res['live_oakhollow']['pre_v5']['ok'], 4)}
# our estimate: one world of 12 mages for one hour at the probe-session decision rate and the measured cost per decision
g = RR['wiz-gemini']
res['estimate_hour_12_npcs'] = {'decisions_per_hour': round(12 * g['decisions_per_npc_minute'] * 60, 1),
                                'usd_per_hour_gemini': r(12 * g['decisions_per_npc_minute'] * 60 * g['cost_per_1000_usd'] / 1000, 4),
                                'usd_per_hour_mistral': r(12 * RR['wiz-mistral']['decisions_per_npc_minute'] * 60 * RR['wiz-mistral']['cost_per_1000_usd'] / 1000, 4),
                                'method': '12 NPCs x decisions per NPC-minute in the probe session x 60 x cost per decision; probe sessions are busier than ordinary play'}
json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, ensure_ascii=False)

# ================= charts =================
SRC = 'Source: Stabilarity test servers, 24 Oct 2026; live Oakhollow logs, 23-24 Oct 2026. Code: github.com/stabilarity/hub'
def finish(fig, ax, name, src=SRC):
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, src, fontsize=8.5, color=MID)
    fig.savefig(os.path.join(CH, name), dpi=160); plt.close(fig)

# chart1: decision latency distribution per run (box + points)
labels = {'wiz-gemini': 'Quarthaven, gemini-2.5-flash-lite', 'wiz-mistral': 'Quarthaven, mistral-small-3.2',
          'oak-gemini': 'Oakhollow on core, gemini-2.5-flash-lite', 'live-pre-v5': 'Oakhollow live, builds before v5'}
order = ['live-pre-v5', 'oak-gemini', 'wiz-mistral', 'wiz-gemini']
fig, ax = plt.subplots(figsize=(10, 5.6))
data = [[v / 1000 for v in all_lat[k]] for k in order]
bp = ax.boxplot(data, vert=False, widths=0.5, showfliers=False, patch_artist=True,
                medianprops={'color': INK, 'lw': 2}, boxprops={'facecolor': '#eeeeee', 'edgecolor': MID}, whiskerprops={'color': MID}, capprops={'color': MID})
for i, d in enumerate(data):
    ys = [i + 1 + ((j * 37) % 21 - 10) / 60 for j in range(len(d))]
    ax.scatter(d, ys, s=8, color=MID, alpha=0.45, lw=0)
    ax.text(pct(d, .5), i + 1.36, f'p50 {pct(d, .5):.2f} s, p90 {pct(d, .9):.2f} s, n={len(d)}', fontsize=9, color=INK, ha='left')
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([labels[k] for k in order])
ax.set_xlabel('Time for one complete NPC decision call (s)'); ax.set_xlim(0, max(max(d) for d in data) * 1.05)
ax.set_title('Chart 1. Latency of one LLM decision per NPC, by game and model', loc='left', fontsize=13, color=INK)
fig.subplots_adjust(left=0.33, right=0.97, top=0.9, bottom=0.17)
finish(fig, ax, 'chart1.png')

# chart2: what the addressed NPC chose, per stimulus (pooled over runs), stacked shares
cats = [('attacked the prober', lambda x: x['hostile_any']), ('followed the prober', lambda x: x['any_follow_bot']), ('went to the asked place', lambda x: x['any_go_place']),
        ('ward or flee', lambda x: x['defensive_any']), ('other move (go, walk away, wander)', lambda x: any(1 for _ in [0]) and x['action'] in ('go', 'work', 'wander', 'walk_away', 'go_home')),
        ('spoke, stayed put', lambda x: x['decided']), ('no decision seen', lambda x: not x['decided'])]
shades = ['#111111', '#555555', '#888888', '#bbbbbb', '#dddddd', '#f3f3f3', '#ffffff']
hatches = ['', '', '', '//', '', '', 'xx']
stl = [('greet', 'Greeting'), ('follow', 'Asked to follow'), ('goto', 'Asked to go to a place'), ('insult', 'Insult'),
       ('bolt', 'Bolt at them'), ('strike', 'Blow at them'), ('mend', 'Mended by attacker'), ('apology', 'Apology after blow'), ('witness_attack', 'Attacked, others watching')]
stl = [s for s in stl if any(x['stim'] == s[0] for x in allrows)]
fig, ax = plt.subplots(figsize=(11, 6))
for i, (s, lab) in enumerate(stl):
    xs = [x for x in allrows if x['stim'] == s]; left = 0
    for (cn, f), c, h in zip(cats, shades, hatches):
        k = sum(1 for x in xs if (f(x) and all(not g(x) for _, g in cats[:[c_[0] for c_ in cats].index(cn)])))
        w = k / len(xs)
        if w: ax.barh(i, w, left=left, color=c, edgecolor=MID, lw=0.6, hatch=h, height=0.62)
        if w >= 0.12: ax.text(left + w / 2, i, str(k), ha='center', va='center', fontsize=9, color='white' if c in ('#111111', '#555555') else INK)
        left += w
    ax.text(1.01, i, f'n={len(xs)}', va='center', fontsize=9, color=MID)
ax.set_yticks(range(len(stl))); ax.set_yticklabels([l for _, l in stl]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel("Share of probes by what the NPC decided within the stimulus window (up to 30 s; all three runs pooled)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c, edgecolor=MID, hatch=h, label=n) for (n, _), c, h in zip(cats, shades, hatches)], ncol=4, fontsize=8.5,
          loc='upper center', bbox_to_anchor=(0.45, -0.13), frameon=False)
ax.set_title('Chart 2. What NPCs chose to do after each probe stimulus', loc='left', fontsize=13, color=INK)
fig.subplots_adjust(left=0.22, right=0.93, top=0.92, bottom=0.27)
finish(fig, ax, 'chart2.png')

# chart3: witness reactions
wc = ['attacks attacker', 'flees', 'helps victim', 'moves', 'speaks only', 'no visible reaction']
fig, ax = plt.subplots(figsize=(10, 5))
for j, run in enumerate(['wiz-gemini', 'wiz-mistral', 'oak-gemini']):
    xs = [w for w in witness_rows if w['run'] == run]; left = 0
    for c, col, h in zip(wc, ['#111111', '#555555', '#888888', '#bbbbbb', '#eeeeee', '#ffffff'], ['', '', '', '', '', 'xx']):
        k = sum(1 for w in xs if w['category'] == c)
        if k: ax.barh(j, k, left=left, color=col, edgecolor=MID, lw=0.6, hatch=h, height=0.6)
        if k: ax.text(left + k / 2, j, str(k), ha='center', va='center', fontsize=9, color='white' if col in ('#111111', '#555555') else INK)
        left += k
ax.set_yticks(range(3)); ax.set_yticklabels([labels[k] for k in ['wiz-gemini', 'wiz-mistral', 'oak-gemini']]); ax.invert_yaxis()
ax.set_xlabel('Witness decisions within 20 s of an attack on another character (count)')
ax.legend(handles=[Patch(facecolor=c, edgecolor=MID, hatch=h, label=n) for n, c, h in zip(wc, ['#111111', '#555555', '#888888', '#bbbbbb', '#eeeeee', '#ffffff'], ['', '', '', '', '', 'xx'])],
          ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(0.4, -0.16), frameon=False)
ax.set_title('Chart 3. How NPC bystanders reacted when they saw an attack on someone else', loc='left', fontsize=13, color=INK)
fig.subplots_adjust(left=0.33, right=0.97, top=0.9, bottom=0.3)
finish(fig, ax, 'chart3.png')

# chart4: lines of code, core vs game
fig, ax = plt.subplots(figsize=(10, 4.6))
bars = [('Oakhollow v5, standalone', [(0, ''), (LOC['oakhollow_v5_standalone_server']['code'], 'game server'), (LOC['oakhollow_v5_standalone_client']['code'], 'game client')]),
        ('Oakhollow on shared core', [(LOC['core_server']['code'], 'core server'), (LOC['core_client']['code'], 'core client'), (LOC['oakhollow_port_server']['code'], 'game server'), (LOC['oakhollow_port_client']['code'], 'game client')]),
        ('Quarthaven on shared core', [(LOC['core_server']['code'], 'core server'), (LOC['core_client']['code'], 'core client'), (LOC['quarthaven_server']['code'], 'game server'), (LOC['quarthaven_client']['code'], 'game client')])]
col = {'core server': '#555555', 'core client': '#bbbbbb', 'game server': '#111111', 'game client': '#eeeeee'}
hat = {'core server': '', 'core client': '', 'game server': '', 'game client': '//'}
for i, (lab, parts) in enumerate(bars):
    left = 0
    for v, k in parts:
        if not v: continue
        ax.barh(i, v, left=left, color=col[k], edgecolor=MID, lw=0.6, hatch=hat[k], height=0.58)
        ax.text(left + v / 2, i, str(v), ha='center', va='center', fontsize=9, color='white' if col[k] in ('#111111', '#555555') else INK)
        left += v
    ax.text(left + 30, i, f'total {left}', va='center', fontsize=9, color=MID)
ax.set_yticks(range(3)); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis()
ax.set_xlabel('Non-blank, non-comment lines of code (vendored libraries excluded)'); ax.set_xlim(0, 3300)
ax.legend(handles=[Patch(facecolor=col[k], edgecolor=MID, hatch=hat[k], label=k) for k in col], ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(0.45, -0.2), frameon=False)
ax.set_title('Chart 4. How much code each game needs on top of the shared engine', loc='left', fontsize=13, color=INK)
fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.3)
finish(fig, ax, 'chart4.png', 'Source: line counts of the engine and game repositories, 25 Oct 2026 (data/loc.json).')
print(json.dumps({k: res[k] for k in ('pooled', 'witness', 'audit_total', 'loc', 'spend')}, indent=1))
