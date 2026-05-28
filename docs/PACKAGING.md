# Packaging — single-file `.exe`

The React UI, FastAPI backend, and Python simulation core ship as a single
Tauri-packaged desktop app. Final artifact on Windows:

```
LiFi PV Simulator_0.1.0_x64_en-US.msi   (~80–120 MB, self-contained)
```

## One-time setup

### 1. Rust toolchain

```powershell
winget install Rustlang.Rustup
# then open a new terminal:
rustup default stable
```

### 2. Tauri prerequisites (Windows only — WebView2 runtime)

WebView2 is pre-installed on Windows 11. On Windows 10, install from
<https://developer.microsoft.com/en-us/microsoft-edge/webview2>.

### 3. Frontend deps (one-time)

```powershell
cd frontend
$env:NODE_OPTIONS = "--use-system-ca"  # only if you hit the TLS cert issue
npm install
```

### 4. App icons (one-time)

```powershell
cd frontend
npx @tauri-apps/cli icon path\to\your\logo.png
# writes src-tauri/icons/ — see src-tauri/icons/README.md
```

## Each build

### A. Build the Python sidecar

```powershell
# From repo root, with venv created and dependencies installed:
.\scripts\build-sidecar.ps1
```

This:

1. Installs `pyinstaller` into your venv
2. Runs `pyinstaller backend\lifi-backend.spec --clean`
3. Outputs `dist\lifi-backend.exe`
4. Copies it to `frontend\src-tauri\binaries\lifi-backend-<target-triple>.exe`
   where Tauri expects sidecar binaries

### B. Build the Tauri installer

```powershell
cd frontend
npm run tauri:build
```

This:

1. Runs `npm run build` (Vite → `frontend/dist/`)
2. Compiles the Rust shell
3. Bundles the sidecar + frontend + WebView2 hook into an MSI installer
4. Outputs to `frontend/src-tauri/target/release/bundle/msi/`

## Dev workflow (no packaging)

For day-to-day iteration you don't need to rebuild the sidecar — run the
backend and frontend separately. The frontend detects it's not in Tauri and
talks to `http://localhost:8000`:

```powershell
# Terminal 1
.\venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

Open <http://localhost:5173>.

## Dev workflow inside Tauri

If you want the full Tauri shell during dev (window chrome, sidecar
lifecycle, native menus) without bundling an MSI:

```powershell
# Build the sidecar once
.\scripts\build-sidecar.ps1

# Run Tauri in dev mode — it spawns the sidecar AND the Vite dev server
cd frontend
npm run tauri:dev
```

## Verification checklist for the packaged build

After installing the MSI, launch the app from the Start menu:

- The boot splash shows briefly while Tauri waits for `/health` → 200 OK
- Setup view loads with the kadirvelu2021 preset's link budget populated
- Open Task Manager → Details — you should see `LiFi PV Simulator.exe`
  AND `lifi-backend.exe` running side-by-side
- Click **Run simulation** — the Engine route streams 3 step events
- Click **Sweeps** → **Run thermal sweep** — points stream live
- Close the window — both processes terminate within ~1 second
- Re-open — uses a fresh random port (no conflict if the previous run
  didn't clean up)
- Disconnect from the network entirely — everything still works (proves
  the sidecar is fully self-contained, no internet round-trips)

## Architecture (what runs where)

```
LiFi PV Simulator.exe              ← Tauri Rust shell
  ├── webview (Chromium)           ← serves the bundled Vite build from disk
  │     React + Plotly + Framer Motion + Zustand
  │     calls http://127.0.0.1:<port> for REST,
  │            ws://127.0.0.1:<port>  for WebSockets
  │
  └── lifi-backend.exe             ← PyInstaller-bundled child process
        FastAPI + uvicorn on a random local port
        wraps cosim/, components/, physics/, etc. unmodified
```

The legacy PyQt6 GUI (`python cli.py gui`) is **untouched** — it still works
as a fallback or development reference.
