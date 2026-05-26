# App icons

Tauri needs platform icons (PNG, ICO, ICNS). The repo doesn't ship binary
artwork — generate the set once with the Tauri CLI before your first
`tauri build`:

```powershell
# From frontend/, after running `npm install` once so tauri CLI exists:
npx @tauri-apps/cli icon path/to/source.png
```

The source PNG should be at least 1024×1024 with transparency. The CLI
writes all required files into this directory:

- `32x32.png`
- `128x128.png`
- `128x128@2x.png`
- `icon.ico` (Windows)
- `icon.icns` (macOS)

If you want to skip icons entirely while iterating, swap the `bundle.icon`
array in `tauri.conf.json` for `[]` — but the bundler (`tauri build`) will
fall back to Tauri's default icon, which is fine for a private build but
not for a public release.
