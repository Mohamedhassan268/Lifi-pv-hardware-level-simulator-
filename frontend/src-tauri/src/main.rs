// Tauri shell for OptiSim (LiFi/PV simulator).
//
// Lifecycle (matches the migration plan §4):
//   1. On setup, pick a free local port via portpicker.
//   2. Spawn the bundled `lifi-backend` sidecar with `--port <N>`.
//   3. Poll `GET http://127.0.0.1:<N>/health` until 200 OK (≤5s).
//   4. Inject `window.__BACKEND_PORT = N` into the webview; the React app
//      (frontend/src/api/client.ts + ws.ts) reads it.
//   5. On window close, kill the child process group.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Duration;

use tauri::api::process::{Command, CommandChild, CommandEvent};
use tauri::{Manager, RunEvent};

struct Sidecar(Mutex<Option<CommandChild>>);

#[tauri::command]
fn backend_port(state: tauri::State<u16>) -> u16 {
    *state
}

fn main() {
    let port = portpicker::pick_unused_port().expect("no free port available");

    tauri::Builder::default()
        .manage(port)
        .manage(Sidecar(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![backend_port])
        .setup(move |app| {
            // ---- Spawn the bundled Python backend ----
            let (mut rx, child) = Command::new_sidecar("lifi-backend")
                .expect("failed to locate lifi-backend sidecar")
                .args(["--port", &port.to_string()])
                .spawn()
                .expect("failed to spawn lifi-backend");

            app.state::<Sidecar>().0.lock().unwrap().replace(child);

            // Log stdout/stderr from the backend so they appear in the Tauri
            // dev console — useful when something fails to import.
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => println!("[backend] {line}"),
                        CommandEvent::Stderr(line) => eprintln!("[backend] {line}"),
                        CommandEvent::Error(err) => eprintln!("[backend!] {err}"),
                        CommandEvent::Terminated(payload) => {
                            eprintln!("[backend] exited code={:?}", payload.code);
                            break;
                        }
                        _ => {}
                    }
                }
            });

            // ---- Wait for /health, then inject the port into the webview ----
            let window = app.get_window("main").expect("main window missing");
            tauri::async_runtime::spawn(async move {
                let url = format!("http://127.0.0.1:{port}/health");
                let mut delay_ms = 100u64;
                for _ in 0..30 {
                    if reqwest::get(&url).await.map(|r| r.status().is_success()).unwrap_or(false) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 3 / 2).min(800);
                }

                let inject = format!("window.__BACKEND_PORT = {port};");
                if let Err(e) = window.eval(&inject) {
                    eprintln!("failed to inject __BACKEND_PORT: {e}");
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error building Tauri app")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
                // Best-effort kill the sidecar so we don't leak a Python process.
                if let Some(state) = app_handle.try_state::<Sidecar>() {
                    if let Some(child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
