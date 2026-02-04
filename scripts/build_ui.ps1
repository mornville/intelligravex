Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RootDir "frontend"
$OutDir = Join-Path $RootDir "voicebot/web/ui"

function Test-Command {
  param([string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-Node {
  if (Test-Command "winget") {
    winget install -e --id OpenJS.NodeJS.LTS
    return
  }
  if (Test-Command "choco") {
    choco install nodejs-lts -y
    return
  }
  throw "npm not found. Install Node.js LTS and re-run this script."
}

if (-not (Test-Path $FrontendDir)) {
  throw "frontend/ not found."
}

if (-not (Test-Command "npm")) {
  Write-Host "npm not found. Attempting to install Node.js LTS..."
  Install-Node
  if (-not (Test-Command "npm")) {
    throw "npm still not found. Restart PowerShell and re-run this script."
  }
}

Write-Host "Building Studio UI..."
Push-Location $FrontendDir
try {
  if (Test-Path (Join-Path $FrontendDir "package-lock.json")) {
    npm ci
  } else {
    npm install
  }
  npm run build
} finally {
  Pop-Location
}

Write-Host "Syncing UI build to $OutDir"
if (Test-Path $OutDir) {
  Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $FrontendDir "dist/*") $OutDir

Write-Host "Done."
