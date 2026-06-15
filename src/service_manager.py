"""
Service Manager for Ollama
Handles starting and monitoring Ollama service
"""
import subprocess
import time
import sys
import os
import requests
from typing import Tuple

class OllamaServiceManager:
    """Manage Ollama service lifecycle"""
    
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_HEALTH_ENDPOINT = f"{OLLAMA_URL}/api/tags"
    MAX_RETRIES = 30  # 30 * 2 seconds = 60 seconds timeout
    RETRY_DELAY = 2  # seconds
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.process = None
        self.is_running = False
        # Use bundled portable Ollama if available
        self.ollama_exe = self._get_ollama_path()
    
    def _get_ollama_path(self) -> str:
        """Get path to Ollama executable, preferring bundled portable version"""
        # Check for portable bundled version first
        import sys
        from pathlib import Path
        
        # If running as installed app
        script_dir = Path(sys.argv[0]).parent if hasattr(sys, 'argv') else Path.cwd()
        portable_path = script_dir / "ollama-portable" / "ollama.exe"
        if portable_path.exists():
            return str(portable_path)
        
        # Check parent directory (launcher.py location)
        parent_portable = script_dir.parent / "ollama-portable" / "ollama.exe"
        if parent_portable.exists():
            return str(parent_portable)
        
        # Fall back to system PATH
        return "ollama"
    
    def log(self, message: str, level: str = "info"):
        """Log messages with level"""
        if not self.verbose:
            return
        
        prefix = {
            "info": "[INFO]",
            "success": "[✓]",
            "error": "[✗]",
            "warning": "[⚠]"
        }.get(level, "[*]")
        
        print(f"{prefix} {message}")
    
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                [self.ollama_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            self.log(f"Error checking Ollama installation: {e}", "error")
            return False
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(self.OLLAMA_HEALTH_ENDPOINT, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start_ollama(self) -> Tuple[bool, str]:
        """Start Ollama service in background"""
        
        if not self.check_ollama_installed():
            error_msg = (
                "Ollama is not installed. "
                "Please download from https://ollama.com/download"
            )
            self.log(error_msg, "error")
            return False, error_msg
        
        self.log("Checking if Ollama is already running...", "info")
        
        # Check if already running
        if self.is_ollama_running():
            self.log("Ollama is already running", "success")
            self.is_running = True
            return True, "Ollama already running"
        
        self.log("Starting Ollama service...", "info")
        
        try:
            # Set environment variables for optimal performance
            env = os.environ.copy()
            env["OLLAMA_NUM_PARALLEL"] = "1"  # Single parallel request
            env["OLLAMA_MAX_LOADED_MODELS"] = "1"  # Only one model in memory
            env["OLLAMA_LOAD_TIMEOUT"] = "10m"  # Increase load timeout
            
            # Start Ollama in background (Windows)
            if sys.platform == "win32":
                # Use CREATE_NO_WINDOW to hide console, CREATE_NEW_PROCESS_GROUP for isolation
                self.process = subprocess.Popen(
                    [self.ollama_exe, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                    start_new_session=False
                )
            else:
                self.process = subprocess.Popen(
                    [self.ollama_exe, "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    preexec_fn=os.setsid
                )
            
            # Wait for Ollama to be ready
            return self.wait_for_ollama_ready()
        
        except Exception as e:
            error_msg = f"Error starting Ollama: {str(e)}"
            self.log(error_msg, "error")
            return False, error_msg
    
    def wait_for_ollama_ready(self) -> Tuple[bool, str]:
        """Wait for Ollama to be ready to accept requests"""
        self.log("Waiting for Ollama to be ready...", "info")
        
        for attempt in range(self.MAX_RETRIES):
            if self.is_ollama_running():
                self.log(
                    f"Ollama is ready (attempt {attempt + 1}/{self.MAX_RETRIES})",
                    "success"
                )
                self.is_running = True
                return True, "Ollama started successfully"
            
            if attempt % 5 == 0:
                self.log(
                    f"Waiting for Ollama... ({attempt + 1}/{self.MAX_RETRIES})",
                    "info"
                )
            
            time.sleep(self.RETRY_DELAY)
        
        error_msg = (
            f"Ollama did not start within {self.MAX_RETRIES * self.RETRY_DELAY} seconds. "
            f"Please check installation or run 'ollama serve' manually."
        )
        self.log(error_msg, "error")
        return False, error_msg
    
    def ensure_model_available(self, model_name: str = "llama3.1:8b") -> Tuple[bool, str]:
        """Ensure LLM model is available"""
        if not self.is_running:
            return False, "Ollama is not running"
        
        self.log(f"Checking if model '{model_name}' is available...", "info")
        
        try:
            result = subprocess.run(
                [self.ollama_exe, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if model_name in result.stdout:
                self.log(f"Model '{model_name}' is available", "success")
                return True, f"Model {model_name} available"
            
            # Model not available, pull it
            self.log(
                f"Model '{model_name}' not found. Downloading... (this may take 5-10 minutes)",
                "info"
            )
            
            result = subprocess.run(
                [self.ollama_exe, "pull", model_name],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )
            
            if result.returncode == 0:
                self.log(f"Model '{model_name}' downloaded successfully", "success")
                return True, f"Model {model_name} downloaded"
            else:
                error_msg = f"Failed to download model: {result.stderr}"
                self.log(error_msg, "error")
                return False, error_msg
        
        except subprocess.TimeoutExpired:
            error_msg = f"Model download timeout. Please try again later."
            self.log(error_msg, "error")
            return False, error_msg
        except Exception as e:
            error_msg = f"Error checking model: {str(e)}"
            self.log(error_msg, "error")
            return False, error_msg
    
    def stop_ollama(self):
        """Stop Ollama service"""
        if self.process:
            try:
                if sys.platform == "win32":
                    self.process.terminate()
                else:
                    os.killpg(os.getpgid(self.process.pid), 9)
                self.is_running = False
                self.log("Ollama stopped", "info")
            except Exception as e:
                self.log(f"Error stopping Ollama: {e}", "error")

if __name__ == "__main__":
    manager = OllamaServiceManager(verbose=True)
    success, msg = manager.start_ollama()
    print(f"\nResult: {msg}")
    
    if success:
        success, msg = manager.ensure_model_available()
        print(f"Model check: {msg}")
    
    sys.exit(0 if success else 1)
