; Open this file in Inno Setup 7 and choose Build > Compile (Ctrl+F9).
; First run: python scripts/build_pyinstaller.py (on Windows).
; The bundle is read from dist\HyperView; installers are written to dist\installer.
; Change MyAppVersion for each release. Keep AppId unchanged for upgrades.
#define MyAppName "HyperView"
#define MyAppVersion "1.0.0"
#define MyAppExeName "HyperView.exe"
#ifndef MyAppSource
#define MyAppSource "dist\HyperView"
#endif
; Optional compiler override: /DMyAppSource="C:\path\to\HyperView"
#define MyAppIcon MyAppSource + "\_internal\ui\assets\hyperview.ico"

[Setup]
AppId={{BA7B56C1-6E33-42C3-935B-BB94CEB4AF16}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
; Install for the current Windows user without requesting administrator rights.
DefaultDirName={localappdata}\Programs\{#MyAppName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
WizardStyle=modern
SetupIconFile={#MyAppIcon}
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=dist\installer
OutputBaseFilename=HyperView-Setup-{#MyAppVersion}-windows-amd64
Compression=lzma2/fast
SolidCompression=yes
CloseApplications=yes
RestartApplications=no
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; The entire --onedir bundle is required, including _internal and its subfolders.
Source: "{#MyAppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; Register both formats for this user. Windows keeps control of UserChoice.
Root: HKCU; Subkey: "Software\Classes\.bil\OpenWithProgids"; ValueType: string; ValueName: "HyperView.BIL"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKCU; Subkey: "Software\Classes\.hdr\OpenWithProgids"; ValueType: string; ValueName: "HyperView.HDR"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKCU; Subkey: "Software\Classes\HyperView.BIL"; ValueType: string; ValueData: "HyperView BIL image"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\HyperView.BIL\DefaultIcon"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"",0"
Root: HKCU; Subkey: "Software\Classes\HyperView.BIL\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" --image ""%1"""
Root: HKCU; Subkey: "Software\Classes\HyperView.HDR"; ValueType: string; ValueData: "HyperView HDR header"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\HyperView.HDR\DefaultIcon"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"",0"
Root: HKCU; Subkey: "Software\Classes\HyperView.HDR\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" --image ""%1"""
Root: HKCU; Subkey: "Software\HyperView\Capabilities"; ValueType: string; ValueName: "ApplicationName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\HyperView\Capabilities"; ValueType: string; ValueName: "ApplicationDescription"; ValueData: "View hyperspectral BIL images and their HDR headers."
Root: HKCU; Subkey: "Software\HyperView\Capabilities"; ValueType: string; ValueName: "ApplicationIcon"; ValueData: """{app}\{#MyAppExeName}"",0"
Root: HKCU; Subkey: "Software\HyperView\Capabilities\FileAssociations"; ValueType: string; ValueName: ".bil"; ValueData: "HyperView.BIL"
Root: HKCU; Subkey: "Software\HyperView\Capabilities\FileAssociations"; ValueType: string; ValueName: ".hdr"; ValueData: "HyperView.HDR"
Root: HKCU; Subkey: "Software\RegisteredApplications"; ValueType: string; ValueName: "HyperView"; ValueData: "Software\HyperView\Capabilities"; Flags: uninsdeletevalue

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent unchecked

Filename: "ms-settings:defaultapps?registeredAppUser=HyperView"; Description: "Choose HyperView as the default for .BIL and .HDR files"; Flags: shellexec postinstall skipifsilent unchecked
