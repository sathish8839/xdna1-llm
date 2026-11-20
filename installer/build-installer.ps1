<#
.SYNOPSIS
Builds dist\xdna1-llm-setup-<version>.exe: the portable build (build-portable.ps1) wrapped by Inno Setup.

.DESCRIPTION
Needs Inno Setup 6 (ISCC.exe on PATH, in its default install folder, or in $env:ISCC) and everything
build-portable.ps1 needs. -Version defaults to the app's <Version> in app\Xdna1Chat.csproj.
#>
param(
    [string]$Version,
    [string]$Kernels = "$PSScriptRoot\..\kernels\build",
    [string]$Pyxrt = "$PSScriptRoot\..\engine\.venv\Lib\site-packages\pyxrt.pyd"
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..").Path

if (-not $Version) {
    $Version = ([xml](Get-Content "$root\app\Xdna1Chat.csproj")).Project.PropertyGroup.Version | Select-Object -First 1
}
$iscc = @($env:ISCC, (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 not found: install it (winget install JRSoftware.InnoSetup) or set $env:ISCC' }

& "$PSScriptRoot\build-portable.ps1" -Out "$root\dist\xdna1-llm" -Kernels $Kernels -Pyxrt $Pyxrt
Write-Host "== installer $Version"
& $iscc /Q "/DBundle=$root\dist\xdna1-llm" "/DAppVersion=$Version" "$PSScriptRoot\xdna1-llm.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
$exe = Get-Item "$root\dist\xdna1-llm-setup-$Version.exe"
Write-Host ("== done: {0} ({1:N0} MB)" -f $exe.FullName, ($exe.Length / 1MB))
