; Inno Setup script for the xdna1-llm installer. build-installer.ps1 runs it on a portable build:
;   ISCC /DBundle=<dist\xdna1-llm> /DAppVersion=<x.y.z> xdna1-llm.iss
; Installs for the current user only (no admin prompt) into %LOCALAPPDATA%\Programs\xdna1-llm.

#ifndef Bundle
  #define Bundle "..\dist\xdna1-llm"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{D8DA61F5-54D6-47B0-A20B-4813D82AE72E}
AppName=xdna1 chat
AppVersion={#AppVersion}
AppVerName=xdna1 chat {#AppVersion}
AppPublisher=xdna1-llm contributors
AppPublisherURL=https://github.com/sathish8839/xdna1-llm
AppSupportURL=https://github.com/sathish8839/xdna1-llm/issues
DefaultDirName={localappdata}\Programs\xdna1-llm
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
OutputDir=..\dist
OutputBaseFilename=xdna1-llm-setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\xdna1-chat.exe
UninstallDisplayName=xdna1 chat

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#Bundle}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[InstallDelete]
; an upgrade replaces the engine's Python and the kernels wholesale, so no stale files are left behind
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\kernels"

[Icons]
Name: "{autoprograms}\xdna1 chat"; Filename: "{app}\xdna1-chat.exe"
Name: "{autodesktop}\xdna1 chat"; Filename: "{app}\xdna1-chat.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\xdna1-chat.exe"; Description: "Start xdna1 chat"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  { AMD's NPU driver installs xrt_coreutil.dll, which the engine loads to reach the NPU. }
  if not FileExists(ExpandConstant('{sys}\xrt_coreutil.dll')) then
    Result := SuppressibleMsgBox(
      'The AMD NPU driver does not seem to be installed.' + #13#10#13#10 +
      'xdna1 chat runs on the NPU of Ryzen 7040, 8040 and 200 series processors and needs AMD''s NPU driver ' +
      '(Device Manager > Neural processors). Install anyway?',
      mbConfirmation, MB_YESNO, IDNO) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep <> usPostUninstall then Exit;
  Data := ExpandConstant('{localappdata}\xdna1-llm');
  if DirExists(Data) and not UninstallSilent() then
    if MsgBox('Also delete the downloaded models, the weight cache and the settings (' + Data + ')?',
              mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(Data, True, True, True);
end;
