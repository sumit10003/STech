"""
S-TECH Main Launcher - Handles Ollama startup, warmup, and Streamlit launch
This file coordinates the complete application startup sequence
"""
import os
import sys
import subprocess
import shutil
import time
import ctypes
import urllib.request
import urllib.error
from pathlib import Path

# Windows console window constants
SW_HIDE = 0
SW_MINIMIZE = 6

def get_app_root():
    """Get application root directory for both dev and packaged layouts."""
    script_dir = Path(__file__).resolve().parent

    # Development layout: main.py lives at project root.
    if (script_dir / "src").exists() and (script_dir / "Data").exists():
        return script_dir

    # Packaged layout: main.py may live in app/ under root.
    if (script_dir.parent / "Data").exists() and (script_dir.parent / "Models").exists():
        return script_dir.parent

    # Safe fallback keeps behavior predictable.
    return script_dir

def print_step(step_num, msg):
    """Print formatted step message"""
    print(f"\n[Step {step_num}] {msg}")
    print("-" * 50)

def minimize_console():
    """Minimize the console window"""
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, SW_MINIMIZE)
    except Exception:
        pass

def is_ollama_running():
    """Check if Ollama service is running"""
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except urllib.error.URLError as e:
        # Connection refused - Ollama not responding yet
        return False
    except Exception as e:
        # Other errors - assume not running
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    print_step(1, "Checking Ollama Installation")
    ollama_path = shutil.which("ollama")
    if ollama_path:
        print(f"[OK] Ollama found at: {ollama_path}")
        return True
    else:
        print("[FAIL] Ollama not found in PATH")
        print("\nOffline mode requires a preinstalled local Ollama runtime.")
        print("Please ensure Ollama is installed and available in PATH.")
        return False

def start_ollama_service():
    """Start Ollama service if not running"""
    print_step(2, "Starting Ollama Service")
    
    # Set Ollama environment for optimal single-user usage
    os.environ["OLLAMA_NUM_PARALLEL"] = "1"
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    os.environ["OLLAMA_LOAD_TIMEOUT"] = "10m"
    
    if is_ollama_running():
        print("[OK] Ollama is already running")
        return True
    
    print("Starting Ollama service...")
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = SW_HIDE
        
        # Try to start Ollama - capture output for debugging
        try:
            process = subprocess.Popen(
                ["ollama", "serve"],
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            print(f"[FAIL] Could not start Ollama process: {e}")
            print("Possible causes:")
            print("  1. Ollama is not installed")
            print("  2. Ollama is not in your system PATH")
            print("  3. You don't have permission to run it")
            return False
        
        # Wait for service to start (up to 60 seconds - increased from 30)
        for i in range(60):
            time.sleep(1)
            if is_ollama_running():
                print(f"[OK] Ollama started successfully (took {i+1}s)")
                return True
            
            # Check if process died
            poll_status = process.poll()
            if poll_status is not None:
                # Process exited with error
                stdout, stderr = process.communicate()
                print(f"[FAIL] Ollama process exited with code {poll_status}")
                if stderr:
                    print(f"Error output: {stderr[:500]}")
                return False
            
            if i % 10 == 0:  # Print every 10 seconds
                print(f"  Waiting... ({i+1}/60)")
        
        print("[FAIL] Ollama did not start in time (60 second timeout)")
        print("Trying to terminate process...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        return False
        
    except Exception as e:
        print(f"[FAIL] Failed to start Ollama: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_available():
    """Check if llama3.1:8b model is available"""
    print("Checking for llama3.1:8b model...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"[WARN] ollama list returned error code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
        
        if "llama3.1:8b" in result.stdout:
            print("[OK] llama3.1:8b model is available")
            return True
        else:
            print("[FAIL] llama3.1:8b model not found")
            print("\nAvailable models:")
            if result.stdout:
                print(result.stdout)
            else:
                print("  (no models listed)")
            print("\nOffline mode does not auto-download models.")
            print("Please import llama3.1:8b into local Ollama before launch.")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] ollama list command timed out (10 seconds)")
        return False
    except Exception as e:
        print(f"[FAIL] Failed to check models: {e}")
        return False

def prewarm_model():
    """Pre-warm the model with a simple test query"""
    print_step(3, "Pre-warming Model")
    print("Loading llama3.1:8b into memory...")
    print("This may take 30-60 seconds on first run...")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b", "test"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] Model loaded in {elapsed:.1f} seconds")
            return True
        else:
            print(f"[WARN] Model warmup returned non-zero, but continuing...")
            return True
            
    except subprocess.TimeoutExpired:
        print("[WARN] Model warmup timed out, but continuing...")
        return True
    except Exception as e:
        print(f"[WARN] Model warmup failed: {e}, but continuing...")
        return True

def launch_streamlit():
    """Launch Streamlit application"""
    print_step(4, "Launching Application")
    
    app_root = get_app_root()
    bundled_python = app_root / "python" / "python.exe"
    python_exe = bundled_python if bundled_python.exists() else Path(sys.executable)
    
    # Determine which app to run (default: user_app.py, or admin_app.py if --admin)
    app_name = "admin_app.py" if "--admin" in sys.argv else "user_app.py"
    app_file = Path(__file__).parent / app_name
    
    # Set environment variables
    os.environ["STECH_ROOT"] = str(app_root)
    os.environ["STECH_MODELS"] = str(app_root / "Models")
    os.environ["STECH_DATA"] = str(app_root / "Data")
    
    cmd = [
        str(python_exe),
        "-m", "streamlit", "run",
        str(app_file),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    
    print(f"Starting Streamlit server ({app_name})...")
    print(f"Using Python: {python_exe}")
    
    # Don't hide console - let user see output
    process = subprocess.Popen(
        cmd,
        cwd=str(app_file.parent),
    )
    
    print("\n" + "=" * 50)
    print("Application is ready!")
    print("=" * 50)
    print("\nOpening browser...")
    print("\nDO NOT CLOSE THIS WINDOW - it keeps the app running")
    print("To stop: Close the browser window or press Ctrl+C here")
    
    time.sleep(3)
    minimize_console()
    
    import webbrowser
    webbrowser.open("http://localhost:8501")
    
    # Wait for Streamlit process
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        process.terminate()

def main():
    """Main application launcher"""
    print("\n" + "=" * 50)
    print("  S-TECH RAG CHATBOT LAUNCHER")
    print("=" * 50)
    
    # Step 1: Check Ollama installation
    if not check_ollama_installed():
        print("\nERROR: AI Engine is not installed")
        input("\nPress Enter to exit...")
        return 1
    
    # Step 2: Start Ollama service
    if not start_ollama_service():
        print("\nERROR: Failed to start Ollama service")
        input("\nPress Enter to exit...")
        return 1
    
    # Check model before warmup
    if not check_model_available():
        print("\nERROR: AI model not found")
        print("Please install the AI model first.")
        input("\nPress Enter to exit...")
        return 1
    
    # Step 3: Pre-warm the model
    prewarm_model()
    
    # Step 4: Launch Streamlit
    launch_streamlit()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
