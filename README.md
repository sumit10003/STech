# S-TECH - Equipment Documentation AI Assistant

Offline RAG (Retrieval-Augmented Generation) chatbot for technical documentation.

## Prerequisites (Install Manually)

Install these components BEFORE running STech_Setup.exe:

### Required
1. **Visual C++ Redistributable** - `VC_redist.x64.exe`
   - Required for Python applications to run
   - Install and restart PC if prompted

2. **Ollama** - Download from https://ollama.com/download
   - AI engine for the LLM
   - After installation, open PowerShell and run:
     ```
     ollama pull llama3.1:8b
     ```
   - Wait for the model to download (~4GB)

### Optional (GPU Acceleration)
3. **NVIDIA Driver** - For 3-5x faster AI responses
   - Only if you have an NVIDIA GPU

4. **CUDA Toolkit 11.8** - Required for GPU acceleration
   - Only if you have an NVIDIA GPU and installed NVIDIA driver

## Installation

1. Install all prerequisites above
2. Run `STech_Setup.exe`
3. Choose your installation folder
4. Complete the installation

## Usage

1. **Launch**: Double-click S-TECH from Start Menu or Desktop
2. **Wait**: Ollama will start and load the model (30-60 seconds on first run)
3. **Admin Panel**: Upload your technical documents (PDF, TXT, DOCX, etc.)
4. **Build Vector Store**: Click "Build Vector Store" in Admin panel
5. **Query**: Switch to User mode and ask questions about your documents

## Folder Structure

```
S-TECH/
├── STech.bat           # Main launcher
├── main.py             # Application launcher script
├── admin_app.py        # Streamlit admin/user interface
├── user_app.py         # Standalone user interface
├── src/                # Python source code
├── Models/             # Embedding model (all-MiniLM-L6-v2)
├── Data/
│   ├── admin_data/     # Upload your documents here
│   └── vector_store/   # Generated vector database
└── python_env/         # Embedded Python (if bundled)
```

## System Requirements

- Windows 10/11 (64-bit)
- 16 GB RAM minimum (32 GB recommended)
- 30 GB free disk space
- Intel/AMD 64-bit processor

## Troubleshooting

### "Ollama not found"
- Ensure Ollama is installed and added to PATH
- Try running `ollama --version` in PowerShell

### "Model not found"
- Run `ollama pull llama3.1:8b` in PowerShell

### Slow responses
- First query takes 30-60 seconds (model loading)
- Subsequent queries are faster
- For faster responses, install NVIDIA driver + CUDA

### Vector store errors
- Ensure you have documents in Data/admin_data/
- Click "Build Vector Store" in Admin panel

## Developer

Developed by Sumit Gupta
