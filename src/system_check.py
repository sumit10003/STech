"""
System Requirements Checker
Validates PC specifications before running the RAG application
"""
import os
import sys
import platform
import psutil
import subprocess
from typing import Dict, List, Tuple

# Import config for paths
try:
    from src import config
except ImportError:
    # If running standalone, set a default
    config = None

class SystemChecker:
    """Check system requirements for RAG application"""
    
    # Minimum requirements
    MIN_PYTHON_VERSION = (3, 9)
    MIN_RAM_GB = 8  # Minimum 8GB RAM
    MIN_DISK_GB = 30  # Minimum 30GB free disk space
    MIN_CPU_CORES = 2
    
    # Recommended specs
    RECOMMENDED_RAM_GB = 16
    RECOMMENDED_DISK_GB = 100
    RECOMMENDED_CPU_CORES = 4
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
    
    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements"""
        version = sys.version_info[:2]
        if version < self.MIN_PYTHON_VERSION:
            self.issues.append(
                f"Python {version[0]}.{version[1]} detected. "
                f"Minimum required: Python {self.MIN_PYTHON_VERSION[0]}.{self.MIN_PYTHON_VERSION[1]}"
            )
            return False
        self.info.append(f"✓ Python {version[0]}.{version[1]}")
        return True
    
    def check_operating_system(self) -> bool:
        """Check if OS is Windows"""
        os_name = platform.system()
        if os_name != "Windows":
            self.warnings.append(
                f"OS detected: {os_name}. "
                f"This application is optimized for Windows 10/11."
            )
        self.info.append(f"✓ OS: {os_name}")
        return os_name == "Windows"
    
    def check_ram(self) -> Tuple[bool, float]:
        """Check available RAM"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        self.info.append(f"✓ Total RAM: {ram_gb:.1f}GB")
        self.info.append(f"✓ Available RAM: {available_gb:.1f}GB")
        
        if ram_gb < self.MIN_RAM_GB:
            self.issues.append(
                f"Insufficient RAM: {ram_gb:.1f}GB. Minimum required: {self.MIN_RAM_GB}GB"
            )
            return False, ram_gb
        
        if ram_gb < self.RECOMMENDED_RAM_GB:
            self.warnings.append(
                f"RAM below recommended: {ram_gb:.1f}GB. "
                f"Recommended: {self.RECOMMENDED_RAM_GB}GB (system may run slower)"
            )
        
        return True, ram_gb
    
    def check_disk_space(self) -> Tuple[bool, float]:
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        self.info.append(f"✓ Total Disk: {disk_total_gb:.1f}GB")
        self.info.append(f"✓ Free Disk: {disk_free_gb:.1f}GB")
        
        if disk_free_gb < self.MIN_DISK_GB:
            self.issues.append(
                f"Insufficient disk space: {disk_free_gb:.1f}GB free. "
                f"Minimum required: {self.MIN_DISK_GB}GB"
            )
            return False, disk_free_gb
        
        if disk_free_gb < self.RECOMMENDED_DISK_GB:
            self.warnings.append(
                f"Disk space below recommended: {disk_free_gb:.1f}GB free. "
                f"Recommended: {self.RECOMMENDED_DISK_GB}GB"
            )
        
        return True, disk_free_gb
    
    def check_cpu(self) -> bool:
        """Check CPU cores"""
        cpu_count = psutil.cpu_count(logical=True)
        self.info.append(f"✓ CPU Cores: {cpu_count}")
        
        if cpu_count < self.MIN_CPU_CORES:
            self.warnings.append(
                f"CPU cores: {cpu_count}. Recommended: {self.RECOMMENDED_CPU_CORES} "
                f"(system may run slower)"
            )
        
        return True
    
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.info.append(f"✓ Ollama installed: {result.stdout.strip()}")
                return True
        except Exception:
            pass
        
        self.issues.append(
            "Ollama not found in PATH. Offline mode requires a preinstalled local Ollama runtime."
        )
        return False
    
    def check_ollama_model(self, model_name: str = "llama3.1:8b") -> bool:
        """Check if Ollama model is downloaded"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if model_name in result.stdout:
                self.info.append(f"✓ Ollama model available: {model_name}")
                return True
        except Exception:
            pass
        
        self.warnings.append(
            f"Ollama model '{model_name}' not downloaded yet. "
            "Offline mode will not download models automatically. "
            "Please import the model locally before launching."
        )
        return False
    
    def check_embedding_model(self) -> bool:
        """Check if embedding model exists locally"""
        if config:
            embedding_path = config.EMBEDDING_MODEL
        else:
            embedding_path = os.path.join("Models", "all-mpnet-base-v2")
        
        if os.path.exists(embedding_path):
            self.info.append(f"✓ Embedding model found: {embedding_path}")
            return True
        
        self.warnings.append(
            f"Embedding model not found at {embedding_path}. "
            "Ensure model is bundled with the installation."
        )
        return False
    
    def run_all_checks(self) -> Dict:
        """Run all system checks and return comprehensive report"""
        print("\n" + "="*60)
        print("SYSTEM REQUIREMENTS CHECK")
        print("="*60 + "\n")
        
        # Run checks
        python_ok = self.check_python_version()
        os_ok = self.check_operating_system()
        ram_ok, ram_size = self.check_ram()
        disk_ok, disk_size = self.check_disk_space()
        cpu_ok = self.check_cpu()
        ollama_ok = self.check_ollama_installed()
        ollama_model_ok = self.check_ollama_model() if ollama_ok else False
        embedding_ok = self.check_embedding_model()
        
        # Print info messages
        print("System Information:")
        for info in self.info:
            print(f"  {info}")
        
        # Print warnings
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        # Print issues
        if self.issues:
            print("\nCritical Issues (blocking):")
            for issue in self.issues:
                print(f"  ✗ {issue}")
        
        # Overall status
        all_critical_ok = python_ok and os_ok and ram_ok and disk_ok and ollama_ok
        
        print("\n" + "="*60)
        if all_critical_ok:
            print("✓ SYSTEM CHECK PASSED - Ready to run application")
        else:
            print("✗ SYSTEM CHECK FAILED - Please resolve critical issues above")
        print("="*60 + "\n")
        
        return {
            "python_ok": python_ok,
            "os_ok": os_ok,
            "ram_ok": ram_ok,
            "disk_ok": disk_ok,
            "cpu_ok": cpu_ok,
            "ollama_installed": ollama_ok,
            "ollama_model_available": ollama_model_ok,
            "embedding_model_available": embedding_ok,
            "all_critical_ok": all_critical_ok,
            "ram_size_gb": ram_size,
            "disk_free_gb": disk_size,
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info
        }

if __name__ == "__main__":
    checker = SystemChecker()
    result = checker.run_all_checks()
    sys.exit(0 if result["all_critical_ok"] else 1)
