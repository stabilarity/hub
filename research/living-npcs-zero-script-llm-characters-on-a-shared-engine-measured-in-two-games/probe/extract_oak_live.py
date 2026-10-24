"""Extracts an anonymised table from the live Oakhollow event logs (2026-10-23/24) into data/oak_live.json.
Kept: per-call latency, tokens, cost, parse result, model, chosen intent/action; hits and deaths by kind (npc/human);
reflex 'hostile' events. Dropped: player names, payer ids, all chat text and thoughts.
Rows are labelled with the server build that was running, from the service start times (UTC)."""
import json, sys, glob
STARTS = ['2026-10-23T17:45:25', '2026-10-23T18:17:28', '2026-10-23T18:26:51', '2026-10-23T19:02:22',
          '2026-10-23T19:03:20', '2026-10-24T01:58:41', '2026-10-24T02:15:42']   # last one = v5 (no scripted behaviour)
def build(ts):
    b = 0
    for i, s in enumerate(STARTS):
        if ts >= s: b = i + 1
    return b
rows = {'llm': [], 'hit': [], 'death': [], 'hostile': [], 'say': [], 'join': [], 'repeat_dropped': []}
for f in sorted(glob.glob('/root/living-village/logs/events-*.jsonl')):
    for l in open(f):
        e = json.loads(l); t = e['type']; b = build(e['ts'])
        if t == 'llm':
            d = e.get('decision') or {}
            rows['llm'].append({'build': b, 'ts': e['ts'], 'npc': e.get('npc'), 'model': e.get('model'), 'ms': e.get('ms'), 'ttfb': e.get('ttfb'),
                'ok': e.get('ok'), 'tin': e.get('tin'), 'tout': e.get('tout'), 'cost': e.get('cost'), 'error': bool(e.get('err')) or e.get('status') not in (None, 200),
                'action': d.get('action') or d.get('intent'), 'says': bool(d.get('say')), 'byok': 'payer' in e and e.get('payer') != 'server'})
        elif t == 'hit':
            rows['hit'].append({'build': b, 'from_kind': e.get('fk'), 'to_kind': e.get('tk'), 'dmg': e.get('dmg'), 'asleep': bool(e.get('asleep'))})
        elif t == 'death':
            rows['death'].append({'build': b, 'kind': e.get('kind'), 'by_kind': e.get('byKind')})
        elif t == 'hostile':
            rows['hostile'].append({'build': b})
        elif t == 'say':
            rows['say'].append({'build': b, 'kind': e.get('kind')})
        elif t == 'join':
            rows['join'].append({'build': b})
        elif t == 'repeat_dropped':
            rows['repeat_dropped'].append({'build': b})
rows['_builds'] = STARTS
json.dump(rows, open(sys.argv[1], 'w'))
for k, v in rows.items(): print(k, len(v))
