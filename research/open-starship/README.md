# Open Starship — Open Source Two-Stage Reusable Rocket Research

**Repository:** [stabilarity/hub](https://github.com/stabilarity/hub/tree/master/research/open-starship)  
**Simulation:** `simulation.html` (Three.js, browser-based, no server required)

---

## Overview

Open Starship is a research series documenting the engineering, economics, and policy of fully reusable two-stage rocket systems — modeled after SpaceX Starship but designed as an open scientific reference implementation.

**Architecture:**
- **Stage 1 (Super Heavy):** ~70m tall booster, 33 Raptor engines, returns to launch site after separation
- **Stage 2 (Starship):** ~50m upper stage, 6 Raptor engines (vacuum), designed for LEO refueling, Moon, and Mars missions

---

## 3D Simulation

**`simulation.html`** — Browser-based Three.js simulation of the full mission profile:

- Earth + atmosphere + clouds
- Moon with orbital path
- Two-stage rocket (Super Heavy + Starship)
- Full mission animation: liftoff → stage sep → boostback → landing → TLI → lunar orbit → return
- Keyboard controls: `Space` play/pause · `R` reset · `V` toggle follow camera

**To run:** Open `simulation.html` in any modern browser (Chrome/Firefox/Edge). No build step, no server.

---

## Mission Profile

| Event | Time | Description |
|-------|------|-------------|
| Liftoff | T+0 | Full thrust, both stages |
| MECO | T+10s | Stage separation |
| Boostback | T+14s | Booster flies back to launch site |
| Re-entry burn | T+22s | Booster slows for re-entry |
| Landing | T+28s | Booster lands at launch site |
| Circularization | T+38s | Ship orbits Earth |
| TLI | T+55s | Trans-Lunar Injection |
| Lunar orbit | T+85s | Ship captured by Moon gravity |
| Landing | T+95s | Ship lands on Moon surface |
| Return | T+115s | Trans-Earth Injection |
| Splashdown | T+175s | Ship splashes down in ocean |

---

## Research Articles

Articles will be published at [hub.stabilarity.com/series/open-starship/](https://hub.stabilarity.com/series/open-starship/)

---

## References

- SpaceX Starship (public documentation)
- NASA Artemis Program
- Federal Aviation Administration (FAA) commercial launch licenses
- FCC experimental rocket licenses (public records)

---

*This is a scientific reference implementation for research purposes. Not affiliated with SpaceX or NASA.*
