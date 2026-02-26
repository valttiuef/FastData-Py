# PowerShell script for Windows - FastData Release
Write-Host "🚀 FastData Windows Installer Release" -ForegroundColor Blue

# Error handling
$ErrorActionPreference = "Stop"

# Read appmeta.json
if (-not (Test-Path "appmeta.json")) {
    Write-Host "❌ appmeta.json not found!" -ForegroundColor Red
    exit 1
}

$config = Get-Content "appmeta.json" | ConvertFrom-Json
$version = $config.version
$appName = $config.app_name
$repoUrl = $config.repo_url
$repoPath = $repoUrl -replace '^https://github\.com/', '' -replace '/$', ''

Write-Host "📦 Version: $version" -ForegroundColor Yellow
Write-Host "📦 App: $appName" -ForegroundColor Yellow

# Check if GitHub CLI is installed
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "❌ GitHub CLI (gh) is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "`n📥 Install it using:" -ForegroundColor Yellow
    Write-Host "   winget install --id GitHub.cli" -ForegroundColor Cyan
    Write-Host "`nAfter installing, CLOSE AND REOPEN VSCode completely." -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
    
    $install = Read-Host "`nInstall now using winget? (y/n)"
    if ($install -eq 'y') {
        try {
            winget install --id GitHub.cli -e
            Write-Host "✅ GitHub CLI installed! Please CLOSE AND REOPEN VSCode completely." -ForegroundColor Green
            exit 0
        } catch {
            Write-Host "❌ Installation failed. Please install manually from: https://cli.github.com/" -ForegroundColor Red
            exit 1
        }
    }
    exit 1
}

# Check GitHub CLI authentication
Write-Host "`n🔑 Checking GitHub authentication..." -ForegroundColor Blue
$authStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Not authenticated with GitHub CLI" -ForegroundColor Red
    Write-Host "   Running login flow..." -ForegroundColor Yellow
    gh auth login
    
    # Check again after login
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Authentication failed. Please run 'gh auth login' manually." -ForegroundColor Red
        exit 1
    }
}
Write-Host "✅ GitHub CLI authenticated" -ForegroundColor Green

# Find EXE in dist folder
$distFolder = ".\dist"
if (-not (Test-Path $distFolder)) {
    Write-Host "❌ Dist folder not found!" -ForegroundColor Red
    exit 1
}

$exeFile = Get-ChildItem -Path $distFolder -Filter "*.exe" | Select-Object -First 1

if (-not $exeFile) {
    Write-Host "❌ No .exe file found in dist folder!" -ForegroundColor Red
    Write-Host "   Contents of dist folder:" -ForegroundColor Yellow
    Get-ChildItem -Path $distFolder | ForEach-Object { Write-Host "   - $($_.Name)" }
    exit 1
}

Write-Host "✅ Found installer: $($exeFile.Name)" -ForegroundColor Green

# Create releases folder
$releaseFolder = ".\releases"
if (Test-Path $releaseFolder) {
    Remove-Item "$releaseFolder\*" -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Force -Path $releaseFolder | Out-Null
}

# Copy with versioned name
$versionedName = "$appName-Setup-$version.exe"
Copy-Item $exeFile.FullName "$releaseFolder\$versionedName" -Force

Write-Host "✅ Copied to: $releaseFolder\$versionedName" -ForegroundColor Green

# Generate checksum
$hash = Get-FileHash "$releaseFolder\$versionedName" -Algorithm SHA256
Write-Host "✅ SHA256 checksum calculated" -ForegroundColor Green

# Show summary
Write-Host "`n📋 Release Summary:" -ForegroundColor Yellow
Write-Host "  Version: v$version"
Write-Host "  File: $versionedName"
$fileInfo = Get-Item "$releaseFolder\$versionedName"
$fileSizeMB = [math]::Round($fileInfo.Length/1MB, 2)
Write-Host "  Size: $fileSizeMB MB"
Write-Host "  SHA256: $($hash.Hash.Substring(0, 16))..." -ForegroundColor Gray

$confirm = Read-Host "`nCreate GitHub release? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "❌ Release cancelled" -ForegroundColor Red
    exit 1
}

# FLAG to track if we need to recreate tag/release
$forceRecreate = $false

# Check if tag already exists and handle it
Write-Host "`n🏷️  Checking git tag v$version..." -ForegroundColor Blue
$tagExists = git tag -l "v$version"
if ($tagExists) {
    Write-Host "⚠️  Tag v$version already exists" -ForegroundColor Yellow
    $deleteTag = Read-Host "   Delete existing tag and recreate? (y/n)"
    if ($deleteTag -eq 'y') {
        git tag -d "v$version"
        git push origin --delete "v$version" 2>$null
        Write-Host "✅ Deleted existing tag" -ForegroundColor Green
        # Continue with creating new tag
    } else {
        Write-Host "❌ Release cancelled - tag already exists" -ForegroundColor Red
        exit 1
    }
}

# Create git tag (if it doesn't exist or we deleted it)
Write-Host "`n🏷️  Creating git tag v$version..." -ForegroundColor Blue
git tag -a "v$version" -m "Release v$version"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create git tag" -ForegroundColor Red
    exit 1
}

git push origin "v$version"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to push git tag" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Git tag created and pushed" -ForegroundColor Green

# Check if release already exists
Write-Host "`n⬆️  Checking if release already exists..." -ForegroundColor Blue
$releaseExists = $false
try {
    $releaseCheck = gh release view "v$version" --json name 2>$null
    if ($releaseCheck) {
        $releaseExists = $true
    }
} catch {
    $releaseExists = $false
}

if ($releaseExists) {
    Write-Host "⚠️  Release v$version already exists" -ForegroundColor Yellow
    $deleteRelease = Read-Host "   Delete existing release and recreate? (y/n)"
    if ($deleteRelease -eq 'y') {
        Write-Host "   Deleting existing release..." -ForegroundColor Gray
        gh release delete "v$version" --yes 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Deleted existing release" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Could not delete release, continuing anyway..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Release cancelled" -ForegroundColor Red
        exit 1
    }
}

# Create GitHub release - DO THIS ONLY ONCE!
Write-Host "`n⬆️  Creating new GitHub release v$version..." -ForegroundColor Blue

# First, create the release as a draft
Write-Host "   Creating release draft..." -ForegroundColor Gray
$releaseCreate = gh release create "v$version" `
    --title "FastData v$version" `
    --draft `
    --repo "$repoPath" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create release draft" -ForegroundColor Red
    Write-Host "   Error: $releaseCreate" -ForegroundColor Red
    exit 1
}

# Small pause to let GitHub process
Start-Sleep -Seconds 2

# Upload the assets
Write-Host "   Uploading installer..." -ForegroundColor Gray
gh release upload "v$version" "$releaseFolder\$versionedName" --repo "$repoPath" --clobber
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to upload installer" -ForegroundColor Red
    exit 1
}

$downloadUrl = "https://github.com/$repoPath/releases/download/v$version/$([System.Uri]::EscapeDataString($versionedName))"

# Create detailed release notes with CORRECT asset URL
$releaseNotes = @"
## FastData v$version

### Windows Installer

[Download FastData Setup v$version]($downloadUrl)

**File:** $versionedName  
**Size:** $fileSizeMB MB  
**Release Tag:** v$version

### What's New in v$version
- Initial release of FastData
- Process data analysis and modeling tool
- Windows installer package

### System Requirements
- **OS:** Windows 10 or Windows 11 (64-bit)
- **RAM:** 4GB minimum
- **Storage:** 500MB free disk space
- **Architecture:** x64

### Installation Instructions
1. Click the download link above to get the installer
2. Run the downloaded $versionedName file
3. Follow the setup wizard
4. Launch FastData from the desktop shortcut

### License
MIT License - Copyright (c) Valtteri Tiitta

---

[View all releases](https://github.com/$repoPath/releases)
"@

# Update the release with the correct notes and publish it
Write-Host "   Publishing release with notes..." -ForegroundColor Gray
gh release edit "v$version" `
    --notes "$releaseNotes" `
    --repo "$repoPath" `
    --draft=$false

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅✅✅ Release v$version created successfully! ✅✅✅" -ForegroundColor Green
    $releaseUrl = "https://github.com/$repoPath/releases/tag/v$version"
    Write-Host "🔗 $releaseUrl" -ForegroundColor Blue
    
    # Verify the release was created with assets
    Start-Sleep -Seconds 2
    $verifyRelease = gh release view "v$version" --json assets --repo "$repoPath" 2>$null | ConvertFrom-Json
    if ($verifyRelease.assets.Count -gt 0) {
        Write-Host "✅ Verified: $($verifyRelease.assets.Count) asset(s) uploaded" -ForegroundColor Green
        Write-Host "   Download URL: $($verifyRelease.assets[0].browser_download_url)" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Warning: Release created but no assets found. Check manually." -ForegroundColor Yellow
    }
    
    $openBrowser = Read-Host "`nOpen release page in browser? (y/n)"
    if ($openBrowser -eq 'y') {
        Start-Process $releaseUrl
    }
    
    Write-Host "`n🎉 Done! Don't forget to update the version in appmeta.json for next release." -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to update release notes" -ForegroundColor Red
    exit 1
}
