<#
.SYNOPSIS
Assembles a portable xdna1-llm folder: the chat app, an embedded Python with the engine, and the kernels.

.DESCRIPTION
Layout of -Out:
  xdna1-chat.exe ...   the app, published self-contained (no .NET or Windows App SDK runtime needed)
  python\              the embeddable CPython with the engine, its dependencies and pyxrt in Lib\site-packages
  kernels\             the NPU kernels (xclbins and instruction streams)
The only machine requirement left is the NPU driver, which provides xrt_coreutil.dll for pyxrt.
Needs the .NET 8 SDK and uv. Run from anywhere; paths default to this checkout.
#>
param(
    [string]$Out = "$PSScriptRoot\..\dist\xdna1-llm",
    [string]$Kernels = "$PSScriptRoot\..\kernels\build",
    [string]$Pyxrt = "$PSScriptRoot\..\engine\.venv\Lib\site-packages\pyxrt.pyd"
)
$ErrorActionPreference = 'Stop'

$PythonVersion = '3.13.17'
$PythonSha256 = '9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38'

$root = (Resolve-Path "$PSScriptRoot\..").Path
$cache = Join-Path $root 'dist\cache'

function Invoke-Checked([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}

foreach ($p in @($Kernels, $Pyxrt)) {
    if (-not (Test-Path $p)) { throw "not found: $p" }
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out, $cache | Out-Null
$Out = (Resolve-Path $Out).Path

Write-Host '== app'
Invoke-Checked dotnet @('publish', "$root\app", '-c', 'Release', '-r', 'win-x64', '--self-contained', 'true',
    '-p:DebugType=None', '-o', $Out, '--nologo', '-v', 'q')

Write-Host "== python $PythonVersion"
$zip = Join-Path $cache "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $zip)) {
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    Invoke-WebRequest -UseBasicParsing $url -OutFile "$zip.part"
    Move-Item "$zip.part" $zip
}
if ((Get-FileHash -Algorithm SHA256 $zip).Hash -ne $PythonSha256) {
    Remove-Item $zip
    throw "checksum mismatch for $zip"
}
$py = Join-Path $Out 'python'
Expand-Archive $zip $py
$site = Join-Path $py 'Lib\site-packages'
New-Item -ItemType Directory -Force $site | Out-Null
# The ._pth file fixes sys.path (and ignores PYTHONPATH and user site-packages), so the bundle stays isolated.
$pth = Get-ChildItem $py -Filter 'python*._pth' | Select-Object -First 1
Set-Content -Encoding ascii $pth.FullName "$($pth.BaseName).zip`r`n.`r`nLib\site-packages`r`n"

Write-Host '== engine'
$wheels = Join-Path $cache 'wheels'
if (Test-Path $wheels) { Remove-Item -Recurse -Force $wheels }
Invoke-Checked uv @('build', '--wheel', '--quiet', '-o', $wheels, "$root\engine")
$wheel = (Get-ChildItem $wheels -Filter '*.whl' | Select-Object -First 1).FullName
Invoke-Checked uv @('pip', 'install', '--quiet', '--target', $site, '--python-version', '3.13',
    '--python-platform', 'x86_64-pc-windows-msvc', '--only-binary', ':all:', $wheel)
Copy-Item $Pyxrt $site
Remove-Item -Recurse -Force (Join-Path $site 'bin') -ErrorAction SilentlyContinue  # console-script shims
Invoke-Checked "$py\python.exe" @('-m', 'compileall', '-q', '-j', '0', $site)

Write-Host '== kernels'
$kdir = New-Item -ItemType Directory -Force (Join-Path $Out 'kernels')
Get-ChildItem $Kernels -File | Where-Object { $_.Extension -in '.xclbin', '.bin' -and $_.Name -notlike 'smoke_*' } |
    Copy-Item -Destination $kdir

Write-Host '== notices'
Copy-Item "$root\LICENSE", "$root\NOTICE" $Out
$nuget = if ($env:NUGET_PACKAGES) { $env:NUGET_PACKAGES } else { "$env:USERPROFILE\.nuget\packages" }
Invoke-Checked "$py\python.exe" @("$PSScriptRoot\notices.py", $Out, "$root\app\obj\project.assets.json", $nuget)

$mb = [math]::Round((Get-ChildItem -Recurse -File $Out | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "== done: $Out ($mb MB)"
