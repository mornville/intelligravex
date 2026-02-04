Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendDir = Join-Path $RootDir "frontend"
$OutDir = Join-Path $RootDir "voicebot/web/ui"

if (-not (Test-Path $FrontendDir)) {
  throw "frontend/ not found."
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
