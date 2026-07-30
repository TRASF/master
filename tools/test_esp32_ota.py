"""Local ESP32 OTA Update Server & Test Tool.

Hosts model_full_int8.tflite and config_ota.json on a local HTTP server
for testing ESP32 Over-The-Air updates over Wi-Fi.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import socketserver
from pathlib import Path

DEFAULT_PORT = 8080


def get_local_ip() -> str:
    """Find local IP address on active Wi-Fi network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_ota_manifest(export_dir: Path, local_ip: str, port: int, version: int = 2) -> Path:
    """Create ota_manifest.json pointing to local server URLs."""
    manifest_path = export_dir / "ota_manifest.json"
    manifest_data = {
        "version": version,
        "firmware_url": "",
        "model_url": f"http://{local_ip}:{port}/model_full_int8.tflite",
        "config_url": f"http://{local_ip}:{port}/config_ota.json",
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


def run_ota_test_server(export_dir: str | Path = "output/quantized_export", port: int = DEFAULT_PORT) -> None:
    export_path = Path(export_dir).resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export directory not found at {export_path}. Run export pipeline first.")

    local_ip = get_local_ip()
    manifest_path = generate_ota_manifest(export_path, local_ip, port)

    print("============================================================")
    print("        ESP32 LOCAL OTA TEST SERVER READY                   ")
    print("============================================================")
    print(f"Local Server IP   : http://{local_ip}:{port}")
    print(f"OTA Manifest URL  : http://{local_ip}:{port}/ota_manifest.json")
    print(f"TFLite Model URL  : http://{local_ip}:{port}/model_full_int8.tflite")
    print(f"Config JSON URL   : http://{local_ip}:{port}/config_ota.json")
    print("------------------------------------------------------------")
    print(f"Configure in deployment/tflitemicro/main/config.h:")
    print(f'#define OTA_MANIFEST_URL "http://{local_ip}:{port}/ota_manifest.json"')
    print("============================================================")
    print(f"Serving files from {export_path}... (Press Ctrl+C to stop)")

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(export_path), **kwargs)

    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nOTA Server stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local ESP32 OTA Test Server")
    parser.add_argument("--export-dir", default="output/quantized_export", help="Path to exported model files")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 8080)")
    args = parser.parse_args()

    run_ota_test_server(args.export_dir, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
