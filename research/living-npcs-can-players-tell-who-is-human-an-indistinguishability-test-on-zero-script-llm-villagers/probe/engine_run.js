// Leak measurement on a private copy of Quarthaven (shared engine, unchanged code), seen from the player's side.
// Two scripted player seats join with a key, walk about (one with direction keys, one by clicking) and greet the nearest
// mage now and then. Seat A records every state frame it receives (who is where, which activity icon), every chat line,
// the relations matrix and each profile, exactly as a browser client would get them. At the end a third connection
// tries to join under every mage's name and some player names, to see which names the server refuses.
// usage: NPC_LOG_DIR=<scratch> node engine_run.js <port> <seconds> <out.json>
'use strict';
const fs = require('fs');
const WebSocket = require('/root/ai-npc-core/node_modules/ws');
const { startServer } = require('/root/ai-npc-core/server');
const game = require('/root/wizard-village/wizards.js');
const [PORT, SECS, OUT] = [+process.argv[2], +process.argv[3], process.argv[4]];
const key = fs.readFileSync('/root/.hermes/.env', 'utf8').match(/^OPENROUTER_API_KEY=(\S+)/m)[1].replace(/["']/g, '');
startServer({ game, root: '/root/wizard-village', port: PORT, prefix: '', publicId: 'quarthaven' });
const T0 = Date.now(), now = () => Date.now() - T0;
const rec = { frames: [], chats: [], relations: [], profiles: [], names: [], seats: {} };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const LINES = ['hello there', 'what are you up to?', 'is the Veil still closed?', 'anyone seen the archmage?', 'nice weather today', 'where is the library?'];

function seat(name, record, mode) {
  const ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`);
  const st = { ws, me: null, mine: null, ents: new Map(), send: o => ws.readyState === 1 && ws.send(JSON.stringify(o)) };
  ws.on('message', raw => {
    const m = JSON.parse(raw);
    if (m.t === 'choose') { st.send({ t: 'key', key }); st.send({ t: 'join', name, build: 'm', lang: 'en', element: 'water', hair: m.looks.hair[0] }); }
    else if (m.t === 'you') { st.me = m; rec.seats[name] = m.id; }
    else if (m.t === 's') {
      for (const e of m.e) { if (st.me && e.id === st.me.id) st.mine = e; st.ents.set(e.id, e); }
      if (record) rec.frames.push({ t: now(), e: m.e.map(e => [e.id, e.n, e.x, e.y, e.a || '', e.d, e.hp]) });
    }
    else if (record && m.t === 'chat') rec.chats.push({ t: now(), id: m.id, name: m.name, to: m.to, text: m.text, faint: m.faint });
    else if (record && m.t === 'relations') rec.relations.push({ t: now(), names: m.names, ids: m.ids, nullRows: m.matrix.map(r => r === null) });
    else if (record && m.t === 'profile') rec.profiles.push({ t: now(), id: m.id, name: m.name, hasOutgoing: !!m.outgoing, incoming: (m.incoming || []).length, about: m.about });
  });
  st.loop = async () => {
    while (!st.mine) await sleep(200);
    let k = 0;
    while (now() < SECS * 1000 - 3000) {
      const others = [...st.ents.values()].filter(e => e.id !== st.me.id && !e.d && !Object.values(rec.seats).includes(e.id));
      const near = others.sort((a, b) => Math.hypot(a.x - st.mine.x, a.y - st.mine.y) - Math.hypot(b.x - st.mine.x, b.y - st.mine.y))[0];
      if (near) {
        const dx = near.x - st.mine.x, dy = near.y - st.mine.y, d = Math.hypot(dx, dy);
        if (d > 2.5) {
          if (mode === 'keys') {             // 8-way keyboard movement, as the grid keys send it
            const sx = Math.abs(dx) > 0.7 ? Math.sign(dx) : 0, sy = Math.abs(dy) > 0.7 ? Math.sign(dy) : 0, l = Math.hypot(sx, sy) || 1;
            st.send({ t: 'input', dx: sx / l, dy: sy / l }); await sleep(1500); st.send({ t: 'input', dx: 0, dy: 0 });
          } else { st.send({ t: 'moveTo', x: near.x + 1, y: near.y + 0.5 }); await sleep(2500); }
        } else if (k++ % 3 === 0) { st.send({ t: 'say', to: near.id, text: LINES[(k + name.length) % LINES.length], voice: 'normal' }); await sleep(9000); }
        else { st.send({ t: 'moveTo', x: st.mine.x + (Math.random() * 10 - 5), y: st.mine.y + (Math.random() * 6 - 3) }); await sleep(4000); }
      } else await sleep(1000);
    }
  };
  return st;
}

(async () => {
  await sleep(800);
  const A = seat('Caelum', true, 'keys'), B = seat('Hesper', false, 'click');
  A.loop(); B.loop();
  // relations every 15 s, profiles of everyone once, as a curious player would open the panels
  for (let t = 20; t < SECS - 5; t += 15) setTimeout(() => A.send({ t: 'relations' }), t * 1000);
  setTimeout(async () => { for (const id of [...A.ents.keys()]) { A.send({ t: 'profile', id }); await sleep(150); } }, (SECS - 25) * 1000);
  await sleep((SECS - 4) * 1000);
  // name oracle: which names may a newcomer take?
  const tryNames = [...game.personas.map(p => p.name), 'Merrin', 'Quill', 'Rowena', 'Caelum'];
  for (const n of tryNames) {
    const r = await new Promise(res => {
      const ws = new WebSocket(`ws://127.0.0.1:${PORT}/ws`); let done = false;
      ws.on('message', raw => { const m = JSON.parse(raw);
        if (m.t === 'choose') ws.send(JSON.stringify({ t: 'join', name: n, build: 'm', lang: 'en', element: 'fire', hair: m.looks.hair[0] }));
        else if (!done && (m.t === 'you' || m.t === 'badname')) { done = true; ws.close(); res(m.t); } });
      setTimeout(() => { if (!done) { ws.close(); res('timeout'); } }, 3000);
    });
    rec.names.push({ name: n, npc: game.personas.some(p => p.name === n), in_use_by_seat: n === 'Caelum', result: r });
    await sleep(100);
  }
  rec.npc_names = game.personas.map(p => p.name);
  rec.duration_s = SECS;
  fs.writeFileSync(OUT, JSON.stringify(rec));
  console.log('frames', rec.frames.length, 'chats', rec.chats.length, 'relations', rec.relations.length, 'profiles', rec.profiles.length);
  process.exit(0);
})();
