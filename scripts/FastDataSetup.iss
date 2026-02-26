; ---------- FastDataSetup.iss ----------
; Path to your built EXE (matches main.spec output)
#define MyAppExePath "..\\dist\\FastData-win-x64\\FastData.exe"
#define MyAppExeDir ExtractFileDir(MyAppExePath)

; Read fields from EXE's version info
#define MyAppName GetStringFileInfo(MyAppExePath, "ProductName")
#define MyAppVersion GetStringFileInfo(MyAppExePath, "ProductVersion")
#define MyAppPublisher GetStringFileInfo(MyAppExePath, "CompanyName")
#define MyAppExeName ExtractFileName(MyAppExePath)  ; <-- fixed here

#pragma message "Building " + MyAppName + " version " + MyAppVersion

; Get rid of any old libs always
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Setup]
AppId={{DE7306C9-3E87-4399-B349-5B9671A259F3}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Shows in file properties of the installer (File version/Product version)
VersionInfoVersion={#MyAppVersion}
; Per-machine vs per-user friendly default dir
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=no
OutputDir=..\dist
; Include version in the installer filename
OutputBaseFilename=FastData-Installer-{#MyAppVersion}
SetupIconFile=..\resources\icons\fastdata_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os

; Dual-mode: show choice and do NOT remember previous selection
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
UsePreviousPrivileges=no

; (optional) silence per-user warning since we guard with checks
UsedUserAreasWarning=no

UninstallDisplayIcon={app}\{#MyAppExeName}
; LicenseFile is optional:
; LicenseFile=LICENSE.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#MyAppExeDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion overwritereadonly restartreplace

[Icons]
; Start Menu shortcut (always)
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

; Desktop shortcut for ALL USERS (only when installing per-machine)
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon; Check: IsAdminInstallMode

; Desktop shortcut for CURRENT USER (when installing per-user)
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon; Check: not IsAdminInstallMode

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
; ---------- end ----------
