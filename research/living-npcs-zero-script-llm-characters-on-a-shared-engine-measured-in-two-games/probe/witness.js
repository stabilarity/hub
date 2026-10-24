// Phase-2 probe: a fresh player walks up to an NPC that has other NPCs around it and attacks it without a word
// (a bolt in Quarthaven, two blows in Oakhollow), then stands still and records what happens.
// The victim is the rank-th NPC by number of other NPCs within sight (9 tiles) of it, so parallel bots pick different victims.
// usage: node witness.js <port> <wiz|oak> <botName> <rank> <outFile> [model]
'use strict';
const WebSocket = require('/root/ai-npc-core/node_modules/ws');
const fs = require('fs');
const key = fs.readFileSync('/root/.hermes/.env', 'utf8').match(/^OPENROUTER_API_KEY=(\S+)/m)[1].replace(/["']/g, '');
const [port, game, botName, rank, outFile, model] = process.argv.slice(2);
const WIZ = game === 'wiz';
const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
const out = o => { if (ws.readyState === 1) ws.send(JSON.stringify(o)); };
const rec = o => fs.appendFileSync(outFile, JSON.stringify({ ts: new Date().toISOString(), bot: botName, phase: 2, ...o }) + '\n');
const sleep = ms => new Promise(r => setTimeout(r, ms));
let me = null, mine = null, dead = false;
const ents = new Map();
const D = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
ws.on('message', raw => {
  const m = JSON.parse(raw);
  if (m.t === 'choose') {
    out({ t: 'key', key }); if (model) out({ t: 'model', model });
    const look = WIZ ? { element: 'water', hair: m.looks.hair[0] } : { color: m.looks.color[0], hair: m.looks.hair[0] };
    out({ t: 'join', name: botName, build: 'f', lang: 'en', ...look });
  } else if (m.t === 'you') { me = m; rec({ kind: 'joined' }); }
  else if (m.t === 's' && me) { const seen = new Set(); for (const e of m.e) { if (e.id === me.id) mine = e; else { ents.set(e.n, { ...e, seenAt: Date.now() }); seen.add(e.n); } } for (const [n, e] of ents) if (!seen.has(n)) ents.delete(n); }
  else if (m.t === 'chat' && m.id !== me?.id) rec({ kind: 'chat', from: m.name, to: m.to, text: m.text, faint: m.faint });
  else if (m.t === 'died') { dead = true; rec({ kind: 'bot_died', by: m.by }); }
});
(async () => {
  while (!me || !mine) await sleep(200);
  await sleep(1500);
  const npcsNow = () => [...ents.values()].filter(e => !e.d);
  const ranked = npcsNow().map(v => ({ v, k: npcsNow().filter(o => o !== v && D(o, v) <= 9).length })).sort((a, b) => b.k - a.k);
  const pickV = ranked[+rank] || ranked[0];
  if (!pickV) { rec({ kind: 'skip', why: 'nobody outdoors' }); process.exit(0); }
  const name = pickV.v.n; let d = null; const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    const v = ents.get(name); if (!v) { await sleep(800); continue; }
    d = D(mine, v); if (d <= (WIZ ? 4 : 1.2)) break;
    out({ t: 'moveTo', x: v.x + (mine.x < v.x ? -1 : 1), y: v.y + 0.4 }); await sleep(800);
  }
  const v = ents.get(name);
  if (!v || d > (WIZ ? 4 : 1.2)) { rec({ kind: 'skip', why: 'victim not reached', victim: name }); process.exit(0); }
  const watchers = npcsNow().filter(e => e.n !== name && (D(e, mine) <= 9 || D(e, v) <= 9)).map(e => e.n);
  if (WIZ) out({ t: 'cast', spell: 'bolt', target: v.id }); else { out({ t: 'attack', target: v.id }); setTimeout(() => out({ t: 'attack', target: v.id }), 1000); }
  rec({ kind: 'stim', stim: 'witness_attack', victim: name, dist: +d.toFixed(2), watchers_seen_by_bot: watchers });
  for (let i = 0; i < 3 && !dead; i++) { await sleep(10000); if (mine) out({ t: 'moveTo', x: mine.x, y: mine.y }); }
  rec({ kind: 'done', dead }); ws.close(); setTimeout(() => process.exit(0), 300);
})();
setTimeout(() => process.exit(0), 90000);
