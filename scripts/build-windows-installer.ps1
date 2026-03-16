$ISCC = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

function Get-FullChangelog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ChangelogPath
    )

    if (-not (Test-Path $ChangelogPath)) {
        return $null
    }

    $content = Get-Content -Path $ChangelogPath -Raw
    if ([string]::IsNullOrWhiteSpace($content)) {
        return $null
    }

    # Keep only actual version sections that start with:
    # ## [1.2.3] - 2026-02-28
    $matches = [regex]::Matches(
        $content,
        '(?ms)^##\s+\[(.+?)\]\s+-\s+(.+?)\r?\n([\s\S]*?)(?=^##\s+\[|\z)'
    )

    if ($matches.Count -eq 0) {
        return $null
    }

    $sections = [System.Collections.Generic.List[string]]::new()

    foreach ($match in $matches) {
        $version = $match.Groups[1].Value.Trim()
        $date = $match.Groups[2].Value.Trim()
        $body = $match.Groups[3].Value.Trim()

        if (-not [string]::IsNullOrWhiteSpace($body)) {
            $sections.Add("## [$version] - $date")
            $sections.Add($body)
            $sections.Add("")
        }
    }

    while ($sections.Count -gt 0 -and [string]::IsNullOrWhiteSpace($sections[$sections.Count - 1])) {
        $sections.RemoveAt($sections.Count - 1)
    }

    return $sections -join [Environment]::NewLine
}

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
            "No change log notes were found."
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

        # Version heading: ## [1.2.3] - 2026-02-28
        if ($trimmed -match '^##\s+\[(.+?)\]\s+-\s+(.+)$') {
            if ($output.Count -gt 0 -and $output[$output.Count - 1] -ne '') {
                $output.Add('')
            }
            $output.Add("Version $($Matches[1].Trim()) - $($Matches[2].Trim())")
            continue
        }

        # Section heading: ### Added
        if ($trimmed -match '^###\s+(.+)$') {
            if ($output.Count -gt 0 -and $output[$output.Count - 1] -ne '') {
                $output.Add('')
            }
            $output.Add($Matches[1].Trim() + ':')
            continue
        }

        # Bullet point
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

$fullChangelog = Get-FullChangelog -ChangelogPath ".\CHANGELOG.md"
$installerNotes = Convert-ChangelogToInstallerText -Version $version -SectionMarkdown $fullChangelog

$changelogOutputPath = ".\scripts\installer-changelog.txt"
Set-Content -Path $changelogOutputPath -Value $installerNotes -Encoding utf8

Write-Host "Installer changelog prepared at $changelogOutputPath" -ForegroundColor Green

& $ISCC .\scripts\FastDataSetup.iss
Write-Host "`nOutput should be in .\dist\FastData-Installer.exe"