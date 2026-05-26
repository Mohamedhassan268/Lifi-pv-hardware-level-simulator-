# LiFi / PV Simulator — React Frontend (Phase 1)

Vite + React + TypeScript + Tailwind + Framer Motion + Zustand. Talks to the
FastAPI backend in `../backend/`.

## Quick start (dev)

In two terminals from the repo root:

```bash
# Terminal 1 — backend (port 8000)
./venv/Scripts/python.exe -m uvicorn backend.app:app --reload --port 8000

# Terminal 2 — frontend (port 5173)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. You should see:

- Hero with TX LED → light cone → PV cell, looping beam, staggered copy
- TopBar with preset selector populated from `GET /api/presets`
- Channel canvas + 4 sliders (distance, TX angle, RX tilt, PV area)
- Link Budget card whose values update live as you drag sliders (calls
  `POST /api/link-budget`)

## Verification checklist

- `curl http://localhost:8000/api/presets` returns 6 names
- Open DevTools → Network → filter `link-budget`; dragging the distance
  slider fires one POST per debounce tick (~80 ms)
- Open DevTools → React DevTools → "Highlight updates"; dragging a slider
  re-renders **only** `ChannelCanvas` and `LinkBudgetTable`, not the Hero
- Reduce-motion (Windows Settings → Accessibility → Visual effects →
  Animation effects = Off) freezes the looping beam but not the static layout

## What's NOT in Phase 1

Engine, Results, Workbench, and sweeps land in later phases. The "Run
simulation" button in the TopBar is a placeholder until Phase 2.
