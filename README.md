# ParticleSim — Living Patterns (Python)

A fast, friendly "particle life" / living patterns sandbox written in Python with Pygame + NumPy.

Features:
- Agents with energy, age, reproduction, mutation
- Per-type attraction/repulsion rule matrix (regenerate live)
- Food field that regrows and scent fields that diffuse
- Terrain slows/accelerates movement
- Spatial grid neighbor search for thousands of agents
- Pause/step, screenshots, GIF export
- Presets

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
python -m particlesim --preset 1
```

Keys:
- **P** pause/resume, **S** step one frame
- **Space** regenerate rules
- **R** reset world
- **F** screenshot (PNG in `assets/`)
- **G** toggle trails
- **H** help overlay
- **1..5** load presets

Export a short GIF:
```bash
python -m particlesim --preset 3 --frames 1200 --record out.gif
```

If your machine struggles, try fewer particles with `--n 2000` or turn off trails with **G**.
