# Pairs every NPC line with the decision that produced it, in the article-2 probe logs (zero-script build) and in this
# article's engine run: decision time, line length, time the line was spoken. Writes data/decision_say_pairs.json.
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
A2 = '/root/hub/research/living-npcs-zero-script-llm-characters-on-a-shared-engine-measured-in-two-games/data/raw'
def pairs(evs, run):
    out, pend = [], {}
    for e in sorted(evs, key=lambda e: e['ts']):
        if e['type'] == 'decision' and e.get('say'): pend[e['npc']] = e
        elif e['type'] == 'say' and e.get('kind') == 'npc' and e['from'] in pend and pend[e['from']]['say'] == e['text']:
            d = pend.pop(e['from'])
            out.append({'run': run, 'npc': e['from'], 'chars': len(e['text']), 'decision_ts': d['ts'], 'say_ts': e['ts'], 'react_ms': d.get('react_ms')})
    return out
res = []
for run in ('wiz-gemini', 'wiz-mistral', 'oak-gemini'):
    res += pairs([json.loads(l) for f in glob.glob(f'{A2}/{run}/events-*.jsonl') for l in open(f)], 'a2-' + run)
res += pairs([json.loads(l) for l in open(os.path.join(DATA, 'engine_events.jsonl'))], 'engine-run')
json.dump(res, open(os.path.join(DATA, 'decision_say_pairs.json'), 'w'), indent=0)
print(len(res))
