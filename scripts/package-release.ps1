#!/usr/bin/env pwsh

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("windows")]
    [string]$Target = "windows"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $RepoRoot

function Die {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host ""
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Get-PackageVersion {
    $cargoToml = Join-Path $RepoRoot "Cargo.toml"

    if (!(Test-Path $cargoToml)) {
        Die "Cargo.toml not found at $cargoToml"
    }

    $inPackage = $false

    foreach ($line in Get-Content $cargoToml) {
        if ($line -match '^\[package\]') {
            $inPackage = $true
            continue
        }

        if ($inPackage -and $line -match '^\[') {
            break
        }

        if ($inPackage -and $line -match '^\s*version\s*=\s*"(.+)"') {
            return $Matches[1]
        }
    }

    Die "Could not determine package version from Cargo.toml"
}

function Write-Sha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$File
    )

    if (!(Test-Path $File)) {
        Die "Cannot write checksum because file does not exist: $File"
    }

    $hash = Get-FileHash $File -Algorithm SHA256
    $fileName = [System.IO.Path]::GetFileName($File)
    "$($hash.Hash.ToLower())  $fileName" | Out-File "$File.sha256" -Encoding ascii
}

$Version = Get-PackageVersion

$BinaryPath = Join-Path $RepoRoot "target\x86_64-pc-windows-msvc\release\werk.exe"
$StagingDir = Join-Path $RepoRoot "target\package\windows"
$ReleaseDir = Join-Path $RepoRoot "releases"
$Artifact = Join-Path $ReleaseDir "werk1112-v$Version-windows-x86_64.zip"

Write-Host ""
Write-Host "==> Building Windows release artifact"
Write-Host "    Note: release artifacts are universal runtime-router binaries."
Write-Host "    Running: scripts/build-windows.ps1"

& (Join-Path $ScriptDir "build-windows.ps1")

if ($LASTEXITCODE -ne 0) {
    Die "scripts/build-windows.ps1 failed"
}

if (!(Test-Path $BinaryPath)) {
    Die "expected build output not found: $BinaryPath"
}

Remove-Item $StagingDir -Recurse -Force -ErrorAction Ignore
New-Item $StagingDir -ItemType Directory -Force | Out-Null
New-Item $ReleaseDir -ItemType Directory -Force | Out-Null

Copy-Item $BinaryPath (Join-Path $StagingDir "werk.exe") -Force
Copy-Item (Join-Path $RepoRoot "README.md") (Join-Path $StagingDir "README.md") -Force

Remove-Item $Artifact -Force -ErrorAction Ignore
Remove-Item "$Artifact.sha256" -Force -ErrorAction Ignore

Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $Artifact -Force

Write-Sha256 $Artifact

Write-Host ""
Write-Host "Created:"
Write-Host "  $Artifact"
Write-Host "  $Artifact.sha256"
