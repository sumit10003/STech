; S-TECH Installation Script for Inno Setup 6
; Creates a Windows installer with folder selection dialog
;
; Prerequisites (install manually BEFORE running this installer):
;   1. Visual C++ Redistributable (VC_redist.x64.exe)
;   2. NVIDIA Driver (optional, for GPU acceleration)
;   3. CUDA Toolkit 11.8 (optional, for GPU acceleration)
;   4. Ollama (OllamaSetup.exe)
;   5. Ollama model: ollama pull llama3.1:8b
;
; Build command:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" STech_Setup.iss

#define MyAppName "S-TECH"
#define MyAppVersion "1.0"
#define MyAppPublisher "S-TECH"
#define MyAppURL "https://s-tech.local"
#define MyAppExeName "STech.exe"

[Setup]
; Unique app identifier
AppId={{8F9D5E2A-1B3C-4D6E-9F8A-7C5B3E1D2A4F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}

; Default install location (user can change)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; Allow user to choose directory
DisableDirPage=no
UsePreviousAppDir=yes

; Output settings
OutputDir=.
OutputBaseFilename=STech_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

; Require admin for Program Files, but allow user folder too
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Icon for installer and uninstaller
SetupIconFile=assets\stech.ico
UninstallDisplayIcon={app}\assets\stech.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main launcher
Source: "STech.exe"; DestDir: "{app}"; Flags: ignoreversion

; Main application files
Source: "main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "admin_app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "user_app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "STech.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Source code
Source: "src\*"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs createallsubdirs

; Embedding model (all-MiniLM-L6-v2)
Source: "Models\*"; DestDir: "{app}\Models"; Flags: ignoreversion recursesubdirs createallsubdirs

; Data folders (create empty structure)
Source: "Data\admin_data\*"; DestDir: "{app}\Data\admin_data"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "Data\vector_store\*"; DestDir: "{app}\Data\vector_store"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Assets (icon, etc.)
Source: "assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

; Python environment (if bundled)
Source: "python_env\*"; DestDir: "{app}\python_env"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Dirs]
; Ensure Data folders exist
Name: "{app}\Data\admin_data"
Name: "{app}\Data\vector_store"

[Icons]
; Start Menu shortcuts with icon
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\assets\stech.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

; Desktop shortcut with icon
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\assets\stech.ico"; Tasks: desktopicon

[Run]
; Launch app after installation (optional)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent shellexec

[Code]
// Check if Ollama is installed
function IsOllamaInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/c where ollama', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

// Show warning if prerequisites are missing
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpWelcome then
  begin
    if not IsOllamaInstalled() then
    begin
      MsgBox('WARNING: Ollama is not installed!' + #13#10 + #13#10 +
             'Prerequisites required BEFORE installing S-TECH:' + #13#10 +
             '1. Visual C++ Redistributable (VC_redist.x64.exe)' + #13#10 +
             '2. Ollama (https://ollama.com/download)' + #13#10 +
             '3. Ollama model: ollama pull llama3.1:8b' + #13#10 + #13#10 +
             'Optional (for faster GPU processing):' + #13#10 +
             '4. NVIDIA Driver' + #13#10 +
             '5. CUDA Toolkit 11.8' + #13#10 + #13#10 +
             'Please install prerequisites first, then run this installer again.',
             mbError, MB_OK);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox('S-TECH Installation Complete!' + #13#10 + #13#10 +
           'To use the application:' + #13#10 +
           '1. Launch S-TECH from the Start Menu or Desktop' + #13#10 +
           '2. Wait for Ollama to start and load the model' + #13#10 +
           '3. Use the Admin panel to upload documents' + #13#10 +
           '4. Build the vector store to enable Q&A',
           mbInformation, MB_OK);
  end;
end;
