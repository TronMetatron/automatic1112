#!/usr/bin/env python3
"""
Ollama Server Manager for HunyuanImage-3.0

Start/stop Ollama server and manage models (list, install, remove).
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import requests
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import shutil

OLLAMA_URL = "http://localhost:11434"
# Dynamically find ollama, fallback to some reasonable defaults if not in PATH
OLLAMA_BIN = shutil.which("ollama") or "/usr/local/bin/ollama"
# Place PID file in the same directory as this script rather than a hardcoded path
PID_FILE = Path(__file__).parent / ".ollama_server.pid"

# Note: ~/.ollama is symlinked to /media/james/BigDrive/AI/models/ollama
# This keeps models on BigDrive (home drive is small)

# Popular models for image prompt generation
RECOMMENDED_MODELS = {
    "qwen2.5:7b-instruct": {
        "size": "4.7GB",
        "description": "Fast, good for quick prompt enhancement",
        "recommended_for": "Default enhancement"
    },
    "qwen2.5:14b-instruct": {
        "size": "9GB",
        "description": "Better quality, moderate speed",
        "recommended_for": "Quality enhancement"
    },
    "llama3.2:3b": {
        "size": "2GB",
        "description": "Very fast, lightweight",
        "recommended_for": "Quick drafts"
    },
    "mistral:7b-instruct": {
        "size": "4.1GB",
        "description": "Good creative writing",
        "recommended_for": "Creative prompts"
    },
    "gemma2:9b": {
        "size": "5.4GB",
        "description": "Google's efficient model",
        "recommended_for": "Balanced performance"
    },
    "phi3:medium": {
        "size": "7.9GB",
        "description": "Microsoft's capable model",
        "recommended_for": "Detailed descriptions"
    },
}


class OllamaManager:
    """Manage Ollama server and models"""

    def __init__(self, ollama_bin: str = OLLAMA_BIN, base_url: str = OLLAMA_URL):
        self.ollama_bin = ollama_bin
        self.base_url = base_url.rstrip('/')

    def is_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False

    def get_status(self) -> Dict:
        """Get detailed server status"""
        status = {
            "running": False,
            "pid": None,
            "models": [],
            "version": None,
            "url": self.base_url
        }

        # Check if running
        if self.is_running():
            status["running"] = True
            try:
                # Get models
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status["models"] = [
                        {
                            "name": m["name"],
                            "size": self._format_size(m.get("size", 0)),
                            "modified": m.get("modified_at", "")[:10]
                        }
                        for m in data.get("models", [])
                    ]
            except Exception as e:
                logger.debug(f"Error getting models: {e}")

        # Check PID file
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                # Check if process exists
                os.kill(pid, 0)
                status["pid"] = pid
            except (ValueError, ProcessLookupError):
                pass

        # Get version
        try:
            result = subprocess.run(
                [self.ollama_bin, "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                status["version"] = result.stdout.strip().split()[-1]
        except:
            pass

        return status

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    def start(self, wait: bool = True, timeout: int = 30, gpu_index: Optional[int] = None, gpu_indices: Optional[List[int]] = None) -> Tuple[bool, str]:
        """Start the Ollama server.

        Args:
            wait: Wait for server to be ready
            timeout: Timeout in seconds
            gpu_index: Single GPU index to use (legacy, sets CUDA_VISIBLE_DEVICES)
            gpu_indices: List of GPU indices to use (preferred over gpu_index)
        """
        if self.is_running():
            return True, "Ollama server is already running"

        try:
            # Set up environment with GPU selection
            env = os.environ.copy()

            # Determine GPU selection (gpu_indices takes precedence)
            if gpu_indices is not None:
                # Filter out GPU 1 (Blackwell) - always exclude it for memory safety
                safe_gpus = [i for i in gpu_indices if i != 1]
                if not safe_gpus:
                    safe_gpus = [0]  # Fallback to GPU 0
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, safe_gpus))
                logger.info(f"Starting Ollama on GPU(s) {safe_gpus}")
            elif gpu_index is not None:
                # Legacy single GPU mode
                if gpu_index == 1:
                    gpu_index = 0  # Redirect Blackwell to GPU 0
                    logger.warning("GPU 1 (Blackwell) requested but excluded, using GPU 0")
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                logger.info(f"Starting Ollama on GPU {gpu_index}")
            else:
                # Default to GPUs 0 and 2, EXCLUDING GPU 1 (Blackwell) which is reserved for image generation
                env["CUDA_VISIBLE_DEVICES"] = "0,2"
                logger.info("Starting Ollama on GPUs 0,2 (Blackwell reserved for image gen)")

            # Start server in background
            process = subprocess.Popen(
                [self.ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env
            )

            # Save PID
            PID_FILE.write_text(str(process.pid))

            if wait:
                # Wait for server to be ready
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_running():
                        return True, f"Ollama server started (PID: {process.pid})"
                    time.sleep(0.5)
                return False, "Server started but not responding"

            return True, f"Ollama server starting (PID: {process.pid})"

        except FileNotFoundError:
            return False, f"Ollama binary not found at {self.ollama_bin}"
        except Exception as e:
            return False, f"Error starting server: {e}"

    def stop(self) -> Tuple[bool, str]:
        """Stop the Ollama server"""
        if not self.is_running():
            # Clean up PID file if exists
            if PID_FILE.exists():
                PID_FILE.unlink()
            return True, "Ollama server is not running"

        try:
            # Try to get PID from file
            pid = None
            if PID_FILE.exists():
                try:
                    pid = int(PID_FILE.read_text().strip())
                except:
                    pass

            # Find ollama processes
            result = subprocess.run(
                ["pgrep", "-f", "ollama serve"],
                capture_output=True, text=True
            )
            pids = [int(p) for p in result.stdout.strip().split() if p]

            if pid and pid not in pids:
                pids.append(pid)

            # Kill processes
            for p in pids:
                try:
                    os.kill(p, signal.SIGTERM)
                except ProcessLookupError:
                    pass

            # Wait for shutdown
            time.sleep(1)

            # Force kill if still running
            if self.is_running():
                for p in pids:
                    try:
                        os.kill(p, signal.SIGKILL)
                    except:
                        pass
                time.sleep(0.5)

            # Clean up PID file
            if PID_FILE.exists():
                PID_FILE.unlink()

            if not self.is_running():
                return True, "Ollama server stopped"
            else:
                return False, "Failed to stop server"

        except Exception as e:
            return False, f"Error stopping server: {e}"

    def restart(self, gpu_index: Optional[int] = None, gpu_indices: Optional[List[int]] = None) -> Tuple[bool, str]:
        """Restart the Ollama server.

        Args:
            gpu_index: Single GPU index to use (legacy, sets CUDA_VISIBLE_DEVICES)
            gpu_indices: List of GPU indices to use (preferred over gpu_index)
        """
        self.stop()
        time.sleep(1)
        return self.start(gpu_index=gpu_index, gpu_indices=gpu_indices)

    def list_models(self) -> List[Dict]:
        """List installed models"""
        if not self.is_running():
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "name": m["name"],
                        "size": self._format_size(m.get("size", 0)),
                        "modified": m.get("modified_at", "")[:10],
                        "family": m.get("details", {}).get("family", ""),
                        "parameters": m.get("details", {}).get("parameter_size", "")
                    }
                    for m in data.get("models", [])
                ]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []

    def pull_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """Pull/install a model"""
        if not self.is_running():
            return False, "Ollama server is not running"

        try:
            # Use streaming to show progress
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=3600  # 1 hour for large models
            )

            if response.status_code != 200:
                return False, f"Error: {response.status_code}"

            last_status = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")

                    if progress_callback:
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total > 0:
                            pct = (completed / total) * 100
                            progress_callback(status, pct)
                        else:
                            progress_callback(status, -1)

                    last_status = status

                    if data.get("error"):
                        return False, data["error"]

            return True, f"Model '{model_name}' installed successfully"

        except requests.exceptions.Timeout:
            return False, "Timeout while downloading model"
        except Exception as e:
            return False, f"Error pulling model: {e}"

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """Delete/remove a model"""
        if not self.is_running():
            return False, "Ollama server is not running"

        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name},
                timeout=30
            )

            if response.status_code == 200:
                return True, f"Model '{model_name}' deleted"
            else:
                return False, f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            return False, f"Error deleting model: {e}"

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed info about a model"""
        if not self.is_running():
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None


def format_status(status: Dict) -> str:
    """Format status for display"""
    lines = []
    lines.append(f"Ollama Server Status")
    lines.append("=" * 40)
    lines.append(f"Running: {'Yes' if status['running'] else 'No'}")

    if status['version']:
        lines.append(f"Version: {status['version']}")
    if status['pid']:
        lines.append(f"PID: {status['pid']}")
    lines.append(f"URL: {status['url']}")

    if status['models']:
        lines.append(f"\nInstalled Models ({len(status['models'])}):")
        lines.append("-" * 40)
        for m in status['models']:
            lines.append(f"  {m['name']:<30} {m['size']:>8}")
    else:
        lines.append("\nNo models installed")

    return "\n".join(lines)


def main():
    """CLI interface for Ollama management"""
    parser = argparse.ArgumentParser(description="Ollama Server Manager")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Status
    subparsers.add_parser('status', help='Show server status')

    # Start
    start_parser = subparsers.add_parser('start', help='Start Ollama server')
    start_parser.add_argument('--no-wait', action='store_true', help="Don't wait for server to be ready")

    # Stop
    subparsers.add_parser('stop', help='Stop Ollama server')

    # Restart
    subparsers.add_parser('restart', help='Restart Ollama server')

    # List models
    subparsers.add_parser('list', help='List installed models')

    # Pull/install model
    pull_parser = subparsers.add_parser('pull', help='Pull/install a model')
    pull_parser.add_argument('model', help='Model name (e.g., llama3.2:3b)')

    # Delete model
    delete_parser = subparsers.add_parser('delete', help='Delete a model')
    delete_parser.add_argument('model', help='Model name to delete')

    # Show recommended models
    subparsers.add_parser('recommended', help='Show recommended models for prompt generation')

    # Info about a model
    info_parser = subparsers.add_parser('info', help='Show info about a model')
    info_parser.add_argument('model', help='Model name')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = OllamaManager()

    if args.command == 'status':
        status = manager.get_status()
        print(format_status(status))

    elif args.command == 'start':
        print("Starting Ollama server...")
        success, msg = manager.start(wait=not args.no_wait)
        print(msg)
        if success:
            status = manager.get_status()
            if status['models']:
                print(f"\n{len(status['models'])} model(s) available")

    elif args.command == 'stop':
        print("Stopping Ollama server...")
        success, msg = manager.stop()
        print(msg)

    elif args.command == 'restart':
        print("Restarting Ollama server...")
        success, msg = manager.restart()
        print(msg)

    elif args.command == 'list':
        models = manager.list_models()
        if not models:
            if not manager.is_running():
                print("Ollama server is not running")
            else:
                print("No models installed")
            return

        print(f"\nInstalled Models ({len(models)}):")
        print("-" * 60)
        print(f"{'Name':<35} {'Size':>10} {'Modified':>12}")
        print("-" * 60)
        for m in models:
            print(f"{m['name']:<35} {m['size']:>10} {m['modified']:>12}")

    elif args.command == 'pull':
        if not manager.is_running():
            print("Starting Ollama server first...")
            success, msg = manager.start()
            if not success:
                print(msg)
                return

        print(f"Pulling model: {args.model}")
        print("This may take a while for large models...\n")

        def progress(status, pct):
            if pct >= 0:
                print(f"\r{status}: {pct:.1f}%", end="", flush=True)
            else:
                print(f"\r{status}...", end="", flush=True)

        success, msg = manager.pull_model(args.model, progress_callback=progress)
        print(f"\n{msg}")

    elif args.command == 'delete':
        confirm = input(f"Delete model '{args.model}'? (y/N): ")
        if confirm.lower() == 'y':
            success, msg = manager.delete_model(args.model)
            print(msg)
        else:
            print("Cancelled")

    elif args.command == 'recommended':
        print("\nRecommended Models for Image Prompt Generation:")
        print("=" * 60)
        for name, info in RECOMMENDED_MODELS.items():
            print(f"\n{name}")
            print(f"  Size: {info['size']}")
            print(f"  Description: {info['description']}")
            print(f"  Best for: {info['recommended_for']}")
        print("\nInstall with: python ollama_manager.py pull <model-name>")

    elif args.command == 'info':
        info = manager.get_model_info(args.model)
        if info:
            print(f"\nModel: {args.model}")
            print("-" * 40)
            if 'modelfile' in info:
                print(info['modelfile'][:500] + "..." if len(info.get('modelfile', '')) > 500 else info.get('modelfile', ''))
            if 'details' in info:
                for k, v in info['details'].items():
                    print(f"{k}: {v}")
        else:
            print(f"Model '{args.model}' not found or server not running")


if __name__ == "__main__":
    main()
