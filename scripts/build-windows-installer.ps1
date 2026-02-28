$ISCC = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

# @ai(gpt-5, codex, refactor, 2026-02-28)
function Get-ChangelogSection {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version,
        [Parameter(Mandatory = $true)]
        [string]$ChangelogPath
    )

    if (-not (Test-Path $ChangelogPath)) {
        return $null
    }

    $content = Get-Content -Path $ChangelogPath -Raw
    $escapedVersion = [regex]::Escape($Version)
    $pattern = '(?ms)^##\s+\[' + $escapedVersion + '\]\s+-\s+.+?\r?\n([\s\S]*?)(?=^##\s+\[|\z)'
    $match = [regex]::Match($content, $pattern)

    if (-not $match.Success) {
        return $null
    }

    $section = $match.Groups[1].Value.Trim()
    if ([string]::IsNullOrWhiteSpace($section)) {
        return $null
    }

    return $section
}

# @ai(gpt-5, codex, refactor, 2026-02-28)
function Convert-ChangelogToInstallerText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version,
        [Parameter(Mandatory = $false)]
        [string]$SectionMarkdown
    )

    if ([string]::IsNullOrWhiteSpace($SectionMarkdown)) {
        return @(
            "FastData v$Version - What's New",
            "",
            "No change log notes were found for this version."
        ) -join [Environment]::NewLine
    }

    $lines = $SectionMarkdown -split "`r?`n"
    $output = [System.Collections.Generic.List[string]]::new()
    $output.Add("FastData v$Version - What's New")
    $output.Add('')

    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            if ($output.Count -gt 0 -and $output[$output.Count - 1] -ne '') {
                $output.Add('')
            }
            continue
        }

        if ($trimmed -match '^###\s+(.+)$') {
            if ($output.Count -gt 0 -and $output[$output.Count - 1] -ne '') {
                $output.Add('')
            }
            $output.Add($Matches[1].Trim() + ':')
            continue
        }

        if ($trimmed -match '^-\s+(.+)$') {
            $output.Add('- ' + $Matches[1].Trim())
            continue
        }

        $output.Add($trimmed)
    }

    while ($output.Count -gt 0 -and $output[$output.Count - 1] -eq '') {
        $output.RemoveAt($output.Count - 1)
    }

    return $output -join [Environment]::NewLine
}

if (-not (Test-Path $ISCC)) {
    Write-Error "ISCC not found at '$ISCC'."
    exit 1
}

$appMetaPath = ".\appmeta.json"
if (-not (Test-Path $appMetaPath)) {
    Write-Error "appmeta.json not found at '$appMetaPath'."
    exit 1
}

$config = Get-Content -Path $appMetaPath | ConvertFrom-Json
$version = [string]$config.version
if ([string]::IsNullOrWhiteSpace($version)) {
    Write-Error "Version is missing from appmeta.json."
    exit 1
}

$changelogSection = Get-ChangelogSection -Version $version -ChangelogPath ".\CHANGELOG.md"
$installerNotes = Convert-ChangelogToInstallerText -Version $version -SectionMarkdown $changelogSection
$changelogOutputPath = ".\scripts\installer-changelog.txt"
Set-Content -Path $changelogOutputPath -Value $installerNotes -Encoding utf8
Write-Host "Installer changelog prepared at $changelogOutputPath" -ForegroundColor Green

& $ISCC .\scripts\FastDataSetup.iss
Write-Host "`nOutput should be in .\dist\FastData-Installer.exe"
