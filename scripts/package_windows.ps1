Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
  throw "This script is intended for Windows."
}

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $RootDir ".build/venv-windows"
$PythonBin = $env:PYTHON_BIN

function Test-Python {
  param(
    [string]$Command,
    [string[]]$Args = @()
  )
  try {
    $process = Start-Process -FilePath $Command `
      -ArgumentList ($Args + @("-c", "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)")) `
      -NoNewWindow -Wait -PassThru
    return $process.ExitCode -eq 0
  } catch {
    return $false
  }
}

$PythonCommand = $null
$PythonArgs = @()

if ($PythonBin) {
  $PythonCommand = $PythonBin
}

if (-not $PythonCommand) {
  $candidates = @(
    @{ Cmd = "py"; Args = @("-3.11") },
    @{ Cmd = "py"; Args = @("-3.10") },
    @{ Cmd = "python"; Args = @() },
    @{ Cmd = "python3"; Args = @() }
  )

  foreach ($candidate in $candidates) {
    if (Test-Python -Command $candidate.Cmd -Args $candidate.Args) {
      $PythonCommand = $candidate.Cmd
      $PythonArgs = $candidate.Args
      break
    }
  }
}

if (-not $PythonCommand) {
  throw "Python 3.10+ is required. Set PYTHON_BIN to a compatible interpreter."
}

Write-Host "Using Python: $PythonCommand $($PythonArgs -join ' ')"

& (Join-Path $RootDir "scripts/build_ui.ps1")

New-Item -ItemType Directory -Force (Join-Path $RootDir ".build") | Out-Null
if (Test-Path $VenvDir) {
  Remove-Item -Recurse -Force $VenvDir
}

& $PythonCommand @($PythonArgs + @("-m", "venv", $VenvDir))

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install "$RootDir[web,packaging]"

$UiData = "$RootDir\voicebot\web\ui;voicebot/web/ui"
$StaticData = "$RootDir\voicebot\web\static;voicebot/web/static"

& $VenvPython -m PyInstaller --noconfirm --clean `
  --name "GravexStudio" `
  --onefile `
  --windowed `
  --collect-all "voicebot.web" `
  --collect-all "tiktoken" `
  --copy-metadata "tiktoken" `
  --add-data $UiData `
  --add-data $StaticData `
  (Join-Path $RootDir "voicebot/launcher.py")

Write-Host "Built Windows app: dist/GravexStudio.exe"
