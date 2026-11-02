// For each LIGHT item, asks the unchanged NPC mind of the shared engine (/root/ai-npc-core/brain.js: same system rules,
// same prompt layout, same model, temperature and JSON schema as in the games) what the character says next.
// The LIGHT setting becomes the shared lore, the crowdworker's persona becomes the NPC's persona, earlier turns become
// memories and the partner's last line becomes the event that triggers the decision.
// usage: node gen_npc_replies.js <data/light_items.json> <out.json>
'use strict';
const fs = require('fs');
const brain = require('/root/ai-npc-core/brain');
const key = fs.readFileSync('/root/.hermes/.env', 'utf8').match(/^OPENROUTER_API_KEY=(\S+)/m)[1].replace(/["']/g, '');
const [inFile, outFile] = process.argv.slice(2);
const items = JSON.parse(fs.readFileSync(inFile, 'utf8'));
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
const game = {
  prompt: { world: 'You live in a medieval fantasy realm of castles, villages, forests and dungeons, among people and creatures of every kind.', doingExample: 'mending a net' },
  workText: () => 'whatever your station in life brings',
};
let cost = 0;
async function one(it) {
  const me = cap(it.self_name), partner = cap(it.partner_name);
  const memory = [], said = [];
  it.context.forEach((line, i) => {
    const mine = i % 2 === 0;
    memory.push(`[day 1 08:${String(i).padStart(2, '0')}] ${mine ? `you said to ${partner}` : `${partner} says to you`}: "${line}"`);
    if (mine) said.push(line);
  });
  const last = it.context[it.context.length - 1];
  const npc = { persona: { name: me, role: it.self_name, traits: it.self_persona, threshold: -40 }, bonds: {}, opinions: { [partner]: 0 },
    memory: memory.slice(0, -1), said, concern: '' };
  const snap = { day: 1, clock: '08:10', period: 'morning', place: it.setting_name, hp: 100, self: '', plan: `talking with ${partner}`, places: [],
    visible: [{ name: partner, look: `a ${it.partner_name}`, dist: 2, ear: 'in earshot', state: 'unhurt', doing: '', lang: 'English', opinion: 0 }],
    events: [`${partner} says to you: "${last}"`], noticed: [], lang: 'English', lore: `${it.setting_name}. ${it.setting}` };
  for (let attempt = 0; attempt < 2; attempt++) {
    let row = null;
    const d = await brain.decide(game, npc, snap, ev => { if (ev.type === 'llm') row = ev; }, { id: 'exp', key, model: brain.DEFAULT_MODEL });
    if (row && row.cost) cost += row.cost;
    if (d && !d.fallback && d.say) return { light_index: it.light_index, npc_reply: d.say, action: d.action, attempts: attempt + 1, model: row?.model, ms: row?.ms, cost: row?.cost };
  }
  return { light_index: it.light_index, npc_reply: '', attempts: 2 };
}
(async () => {
  const out = [];
  for (let i = 0; i < items.length; i += 6) out.push(...await Promise.all(items.slice(i, i + 6).map(one)));
  fs.writeFileSync(outFile, JSON.stringify({ generated_utc: new Date().toISOString(), cost_usd: cost, replies: out }, null, 1));
  console.log('done', out.length, 'empty', out.filter(r => !r.npc_reply).length, 'cost', cost.toFixed(5));
})();
