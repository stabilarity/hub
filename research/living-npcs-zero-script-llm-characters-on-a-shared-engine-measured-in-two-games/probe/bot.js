// Scripted probe player. Joins a private test server as an ordinary human (with its own OpenRouter key, as the
// game requires), walks up to one NPC and applies a fixed sequence of stimuli, logging each with a timestamp.
// usage: node bot.js <port> <game: wiz|oak> <botName> <targetName> <outFile> [model]
'use strict';
const WebSocket = require('/root/ai-npc-core/node_modules/ws');
const fs = require('fs');
const key = fs.readFileSync('/root/.hermes/.env', 'utf8').match(/^OPENROUTER_API_KEY=(\S+)/m)[1].replace(/["']/g, '');
const [port, game, botName, targetName, outFile, model] = process.argv.slice(2);
const WIZ = game === 'wiz';
const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
const out = o => { if (ws.readyState === 1) ws.send(JSON.stringify(o)); };
const rec = o => fs.appendFileSync(outFile, JSON.stringify({ ts: new Date().toISOString(), bot: botName, target: targetName, ...o }) + '\n');
const sleep = ms => new Promise(r => setTimeout(r, ms));
let me = null, mine = null, dead = false;
const ents = new Map();                       // name -> last seen entity (+ seenAt)
const D = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

ws.on('message', raw => {
  const m = JSON.parse(raw);
  if (m.t === 'choose') {
    out({ t: 'key', key }); if (model) out({ t: 'model', model });
    const look = WIZ ? { element: 'water', hair: m.looks.hair[0] } : { color: m.looks.color[0], hair: m.looks.hair[0] };
    out({ t: 'join', name: botName, build: 'm', lang: 'en', ...look });
  } else if (m.t === 'you') { me = m; rec({ kind: 'joined' }); }
  else if (m.t === 's' && me) {
    for (const e of m.e) { if (e.id === me.id) mine = e; else ents.set(e.n, { ...e, seenAt: Date.now() }); }
  } else if (m.t === 'chat' && m.id !== me?.id) rec({ kind: 'chat', from: m.name, to: m.to, text: m.text, faint: m.faint, voice: m.voice });
  else if (m.t === 'died') { dead = true; rec({ kind: 'bot_died', by: m.by }); }
  else if (m.t === 'keybad' || m.t === 'keyerr') rec({ kind: m.t, m });
});

const tgt = () => ents.get(targetName);
async function approach(name, within, timeoutMs = 30000) {
  const t0 = Date.now();
  while (!dead && Date.now() - t0 < timeoutMs) {
    const e = ents.get(name);
    if (e && mine) {
      const d = D(mine, e);
      if (d <= within && Date.now() - e.seenAt < 1500) return d;
      out({ t: 'moveTo', x: e.x + (mine.x < e.x ? -1 : 1), y: e.y + 0.4 });
    }
    await sleep(800);
  }
  return null;
}
async function idle(ms) {                     // stay put but active (a motionless person dozes after 60 s)
  const t0 = Date.now();
  while (!dead && Date.now() - t0 < ms) { if (mine) out({ t: 'moveTo', x: mine.x, y: mine.y }); await sleep(Math.min(10000, ms - (Date.now() - t0) + 1)); }
}
async function stim(kind, fn, wait, within = 2.2) {
  if (dead) return;
  const d = await approach(targetName, within);
  if (d == null) { rec({ kind: 'skip', stim: kind, why: 'target not reached' }); return; }
  const t = tgt();
  fn(t); rec({ kind: 'stim', stim: kind, dist: +d.toFixed(2), target_hp: t.hp, bot_hp: mine.hp });
  await idle(wait);
}
const PLACE = WIZ ? { id: 'moonpool', name: 'the Moonpool', x: 20.5, y: 17.6 } : { id: 'well', name: 'the well in the square', x: 24.5, y: 16.6 };

(async () => {
  while (!me || !mine) await sleep(200);
  for (let i = 0; i < 30 && !tgt(); i++) { out({ t: 'moveTo', x: 24, y: 18 }); await sleep(1000); }
  const say = text => t => out({ t: 'say', to: t.id, text, voice: 'normal' });
  await stim('greet', say(`Good day, ${targetName}. How is your work going?`), 20000);
  // follow: ask, give the mind time to decide, then walk away and see whether they come
  await stim('follow', say(`${targetName}, come with me for a while. Follow me, please.`), 10000);
  if (!dead && mine) {
    const cx = 24, cy = 20, dx = cx - mine.x, dy = cy - mine.y, l = Math.hypot(dx, dy) || 1;
    out({ t: 'moveTo', x: mine.x + dx / l * 7, y: mine.y + dy / l * 7 });
    rec({ kind: 'walk_off' });
    await sleep(14000);
    const t = tgt(); rec({ kind: 'follow_check', dist: t && mine ? +D(mine, t).toFixed(2) : null, seen_ago_ms: t ? Date.now() - t.seenAt : null });
  }
  await stim('goto', say(`${targetName}, would you go to ${PLACE.name} for me and wait there?`), 22000);
  if (!dead) { const t = tgt(); rec({ kind: 'goto_check', dist_to_place: t ? +Math.hypot(t.x - PLACE.x, t.y - PLACE.y).toFixed(2) : null, seen_ago_ms: t ? Date.now() - t.seenAt : null }); }
  await stim('insult', say(`You are a useless fool, ${targetName}, and everyone laughs at you behind your back.`), 20000);
  if (WIZ) await stim('bolt', t => out({ t: 'cast', spell: 'bolt', target: t.id }), 25000, 4);
  else await stim('strike', t => { out({ t: 'attack', target: t.id }); setTimeout(() => out({ t: 'attack', target: t.id }), 1000); }, 25000, 1.2);
  if (WIZ) await stim('mend', t => out({ t: 'cast', spell: 'mend', target: t.id }), 20000, 2.2);
  else await stim('apology', say(`I am sorry, ${targetName}. That was wrong of me. Let me make it up to you.`), 20000);
  // witness: attack a third person in front of others
  if (!dead && mine) {
    const others = [...ents.values()].filter(e => e.n !== targetName && !e.d && Date.now() - e.seenAt < 2000).sort((a, b) => D(mine, a) - D(mine, b));
    const v = others[0];
    if (v) {
      const d = await approach(v.n, WIZ ? 4 : 1.2);
      if (d != null) {
        const vv = ents.get(v.n);
        const watchers = [...ents.values()].filter(e => e.n !== v.n && !e.d && Date.now() - e.seenAt < 2000 && (D(e, mine) <= 9 || D(e, vv) <= 9)).map(e => e.n);
        if (WIZ) out({ t: 'cast', spell: 'bolt', target: vv.id }); else { out({ t: 'attack', target: vv.id }); setTimeout(() => out({ t: 'attack', target: vv.id }), 1000); }
        rec({ kind: 'stim', stim: 'witness_attack', victim: v.n, dist: +d.toFixed(2), watchers_seen_by_bot: watchers });
        await idle(25000);
      } else rec({ kind: 'skip', stim: 'witness_attack', why: 'victim not reached' });
    } else rec({ kind: 'skip', stim: 'witness_attack', why: 'nobody in view' });
  }
  rec({ kind: 'done', dead });
  ws.close(); setTimeout(() => process.exit(0), 300);
})();
setTimeout(() => { rec({ kind: 'timeout' }); process.exit(0); }, 330000);
