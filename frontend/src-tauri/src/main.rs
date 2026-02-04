#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use tauri::{
    api::process::Command,
    CustomMenuItem,
    GlobalShortcutManager,
    Manager,
    SystemTray,
    SystemTrayEvent,
    SystemTrayMenu,
    WindowEvent,
};

#[derive(Clone)]
struct BackendState {
    port_file: PathBuf,
}

#[tauri::command]
fn open_dashboard(window: tauri::Window, url: String) -> Result<(), String> {
    tauri::api::shell::open(&window.shell_scope(), url, None).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_backend_url(state: tauri::State<BackendState>) -> Result<String, String> {
    let port_file = &state.port_file;
    for _ in 0..120 {
        if let Some(port) = read_port_file(port_file) {
            return Ok(format!("http://127.0.0.1:{}", port));
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err("Backend did not start in time.".to_string())
}

fn toggle_window(window: &tauri::Window) {
    let is_visible = window.is_visible().unwrap_or(false);
    if is_visible {
        let _ = window.hide();
    } else {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn read_port_file(path: &Path) -> Option<u16> {
    let text = fs::read_to_string(path).ok()?;
    let raw = text.trim();
    if raw.is_empty() {
        return None;
    }
    raw.parse::<u16>().ok()
}

fn spawn_backend(port_file: &Path) {
    if let Some(parent) = port_file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::remove_file(port_file);
    let Ok(command) = Command::new_sidecar("GravexStudio") else {
        eprintln!("Backend sidecar not found. Start the backend manually.");
        return;
    };
    let envs = HashMap::from([
        ("VOICEBOT_OPEN_BROWSER".to_string(), "0".to_string()),
        ("VOICEBOT_HOST".to_string(), "127.0.0.1".to_string()),
        ("VOICEBOT_PORT".to_string(), "8000".to_string()),
        (
            "VOICEBOT_PORT_FILE".to_string(),
            port_file.to_string_lossy().to_string(),
        ),
    ]);
    let _ = command.envs(envs).spawn();
}

fn main() {
    let show = CustomMenuItem::new("show".to_string(), "Show Widget");
    let hide = CustomMenuItem::new("hide".to_string(), "Hide Widget");
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let tray_menu = SystemTrayMenu::new().add_item(show).add_item(hide).add_item(quit);
    let tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(tray)
        .invoke_handler(tauri::generate_handler![open_dashboard, get_backend_url])
        .setup(|app| {
            let app_handle = app.handle();
            let window = app_handle.get_window("widget");
            if let Some(window) = window {
                let _ = window.set_always_on_top(true);
            }

            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            }

            let port_file = app_handle
                .path_resolver()
                .app_data_dir()
                .unwrap_or_else(|| std::env::temp_dir())
                .join("backend_port.txt");

            spawn_backend(&port_file);
            app.manage(BackendState { port_file });

            let app_handle = app.handle();
            let _ = app
                .global_shortcut_manager()
                .register("Option+Command+Space", move || {
                    if let Some(window) = app_handle.get_window("widget") {
                        toggle_window(&window);
                    }
                });

            Ok(())
        })
        .on_system_tray_event(|app, event| {
            if let SystemTrayEvent::MenuItemClick { id, .. } = event {
                if let Some(window) = app.get_window("widget") {
                    match id.as_str() {
                        "show" => {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                        "hide" => {
                            let _ = window.hide();
                        }
                        "quit" => {
                            std::process::exit(0);
                        }
                        _ => {}
                    }
                }
            }
        })
        .on_window_event(|event| {
            if let WindowEvent::CloseRequested { api, .. } = event.event() {
                api.prevent_close();
                let _ = event.window().hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
