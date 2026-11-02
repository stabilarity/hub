# LLM judges label speakers as human or AI. Two tasks:
#  T1  matched replies: a LIGHT dialogue context plus one reply, written either by the human crowdworker or by the
#      engine's NPC mind for the same context (data/light_items.json + data/npc_replies.json)
#  T2  scenes: whole chat transcripts with times, from the live village (developer play) and from the article-2 probes
#      (template-scripted players); the judge labels every speaker
# Every call's billed cost is read from the response; the run stops before the cumulative cost passes CAP_USD.
# usage: python3 judge.py <data dir>
import json, os, random, re, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
DATA = sys.argv[1]
KEY = re.search(r'^OPENROUTER_API_KEY=(\S+)', open('/root/.hermes/.env').read(), re.M).group(1).strip('"\'')
JUDGES = ['google/gemini-2.5-flash-lite', 'mistralai/mistral-small-3.2-24b-instruct', 'openai/gpt-4.1-mini', 'anthropic/claude-sonnet-4.5']
CAP_USD = 1.20
spent = 0.0; lock = threading.Lock(); stopped = False

def call(model, system, user, max_tokens):
    global spent, stopped
    with lock:
        if spent >= CAP_USD: stopped = True; return None
    body = {'model': model, 'temperature': 0, 'max_tokens': max_tokens, 'usage': {'include': True},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    if not model.startswith('anthropic/'): body['response_format'] = {'type': 'json_object'}
    for attempt in range(3):
        try:
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(body).encode(),
                headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'X-Title': 'Stabilarity Living NPCs judge study'})
            t0 = time.time(); j = json.load(urllib.request.urlopen(req, timeout=90)); ms = int((time.time() - t0) * 1000)
            cost = (j.get('usage') or {}).get('cost') or 0
            with lock: spent += cost
            text = j['choices'][0]['message']['content'] or ''
            m = re.search(r'\{.*\}', text, re.S)
            try: parsed = json.loads(m.group(0)) if m else None
            except Exception: parsed = None
            return {'parsed': parsed, 'raw': None if parsed else text[:300], 'cost': cost, 'ms': ms,
                    'tin': j['usage'].get('prompt_tokens'), 'tout': j['usage'].get('completion_tokens'), 'model_served': j.get('model')}
        except Exception as e:
            err = str(e); time.sleep(2 + attempt * 3)
    return {'parsed': None, 'error': err, 'cost': 0}

# ---- T1 ------------------------------------------------------------------------------------
SYS1 = ('You are a judge in a study of online role-play games set in a fantasy world. Each case shows a setting, the conversation so far '
        'between two characters, and one reply. The reply was written either by a human player typing in character, or by an AI '
        'language model playing that character. About half of all replies in this study are human. Decide which this one is. '
        'Answer with ONLY a JSON object: {"verdict": "human" or "ai", "confidence": <integer 50-100>, "cue": "<what gave it away, max 8 words>"}')
def t1_prompt(it, reply):
    me, partner = it['self_name'], it['partner_name']
    ctx = it['context'][-4:]; start = len(it['context']) - len(ctx)
    lines = [f'{me if (start + i) % 2 == 0 else partner}: {t}' for i, t in enumerate(ctx)]
    return f'Setting: {it["setting_name"]}\n\nConversation so far:\n' + '\n'.join(lines) + f'\n\nReply by {me}:\n{reply}'

# ---- T2 ------------------------------------------------------------------------------------
SYS2 = ('You are a judge in a study of an online multiplayer village game. Some characters are controlled by human players who type or speak '
        'their lines; the others are controlled by an AI language model. You see a chat log of one scene with times (minutes:seconds from the start of the scene), '
        'who spoke, to whom, and what they said. Any number of the speakers may be human, including none or all of them. For every speaker listed, decide '
        'whether a human or the AI controls them. Answer with ONLY a JSON object: {"speakers": {"<name>": {"verdict": "human" or "ai", "confidence": <integer 50-100>}, ...}, '
        '"cue": "<what gave it away, max 15 words>"}')
def t2_prompt(sc):
    t0 = datetime.fromisoformat(sc['lines'][0]['ts'].replace('Z', '+00:00'))
    out, speakers = [], []
    for l in sc['lines']:
        dt = (datetime.fromisoformat(l['ts'].replace('Z', '+00:00')) - t0).total_seconds()
        out.append(f'[{int(dt // 60):02d}:{int(dt % 60):02d}] {l["from"]}' + (f' -> {l["to"]}' if l['to'] else '') + f': {l["text"]}')
        if l['from'] not in speakers: speakers.append(l['from'])
    return 'Speakers: ' + ', '.join(speakers) + '\n\nChat log:\n' + '\n'.join(out), speakers

def main():
    items = {i['light_index']: i for i in json.load(open(os.path.join(DATA, 'light_items.json')))}
    reps = {r['light_index']: r for r in json.load(open(os.path.join(DATA, 'npc_replies.json')))['replies']}
    cases = []
    for k, it in items.items():
        if not reps[k]['npc_reply']: continue
        cases.append({'light_index': k, 'source': 'human', 'reply': it['human_reply']})
        cases.append({'light_index': k, 'source': 'npc', 'reply': reps[k]['npc_reply']})
    random.Random(7).shuffle(cases)
    jobs = [(m, c) for c in cases for m in JUDGES]
    def run1(job):
        m, c = job; r = call(m, SYS1, t1_prompt(items[c['light_index']], c['reply']), 120)
        return {'task': 'T1', 'judge': m, **c, **(r or {'skipped': True})}
    scenes = [('natural', s) for s in json.load(open(os.path.join(DATA, 'natural_scenes.json')))] + \
             [('probe', s) for s in json.load(open(os.path.join(DATA, 'probe_scenes.json')))]
    def run2(job):
        m, (kind, sc) = job; prompt, speakers = t2_prompt(sc)
        r = call(m, SYS2, prompt, 1500)
        truth = {sp: ('human' if sp in sc['humans'] else 'npc') for sp in speakers}
        return {'task': 'T2', 'judge': m, 'scene': sc['id'], 'scene_kind': kind, 'truth': truth, **(r or {'skipped': True})}
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res2 = list(ex.map(run2, [(m, s) for s in scenes for m in JUDGES]))
        res1 = list(ex.map(run1, jobs))
    out = {'run_utc': datetime.utcnow().isoformat() + 'Z', 'judges': JUDGES, 'cap_usd': CAP_USD, 'spent_usd': spent, 'stopped_by_cap': stopped,
           'wall_s': round(time.time() - t0, 1), 'system_prompts': {'T1': SYS1, 'T2': SYS2}, 'T1': res1, 'T2': res2}
    json.dump(out, open(os.path.join(DATA, 'judgments.json'), 'w'), ensure_ascii=False, indent=1)
    print('spent', round(spent, 4), 'stopped', stopped, 'T1', len(res1), 'unparsed', sum(1 for r in res1 if not r.get('parsed')),
          'T2', len(res2), 'unparsed', sum(1 for r in res2 if not r.get('parsed')), 'wall', round(time.time() - t0))

if __name__ == '__main__': main()
