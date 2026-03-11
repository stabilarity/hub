# Open Humanoid — Master Specification Schema v0.1

This schema is referenced by every article in the series. Each subsystem spec follows this format.

## Subsystem Spec Template
```yaml
subsystem: [name]
version: 0.1
status: research | specified | validated | simulated
dependencies: [list of other subsystems]
constraints:
  mass_budget_kg: [value]
  power_budget_w: [value]
  volume_mm: [L x W x H]
  cost_usd: [target BOM cost]
performance_targets:
  [metric]: [value with unit]
open_challenges:
  - [challenge 1]
  - [challenge 2]
references:
  - [DOI or arXiv]
```

## Global Robot Constraints
- Total mass: ≤ 80kg
- Height: 160–180cm
- Battery life: > 60 minutes under normal operation
- Operating temperature: 0–40°C
- IP rating: IP54 minimum
- Emergency stop: hardware + software, <100ms response
- Communication: onboard WiFi6 + Bluetooth 5.2, optional 5G
