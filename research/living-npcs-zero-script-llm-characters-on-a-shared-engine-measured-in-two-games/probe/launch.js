// Starts a private test copy of a game on the shared engine, unchanged, with one addition for measurement:
// before every NPC decision it logs the full perception (events, noticed, visible) with a sequence number,
// and tags the matching 'llm' log row with the same number. Nothing about behaviour is altered.
// usage: NPC_LOG_DIR=<dir> PORT=<port> node launch.js <game-module-path> <publicId>
'use strict';
const brain = require('/root/ai-npc-core/brain');
const orig = brain.decide;
let seq = 0;
brain.decide = (game, npc, snap, log, payer) => {
  const s = ++seq;
  log({ type: 'probe_snap', seq: s, npc: npc.persona.name, trigger: snap.trigger, events: snap.events, noticed: snap.noticed,
    visible: snap.visible.map(v => ({ name: v.name, dist: v.dist, opinion: v.opinion })), hp: snap.hp });
  return orig(game, npc, snap, ev => log(ev.type === 'llm' ? { ...ev, seq: s } : ev), payer);
};
const path = require('path');
const gameFile = path.resolve(process.argv[2]);
const { startServer } = require('/root/ai-npc-core/server');
startServer({ game: require(gameFile), root: path.dirname(gameFile), port: +process.env.PORT, prefix: '', publicId: process.argv[3] });
