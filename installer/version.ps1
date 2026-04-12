param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('get','set')]
    [string]$Action,

    [string]$NewVersion
)

$repoRoot = Join-Path $PSScriptRoot '..' | Resolve-Path -Relative
$versionFile = Join-Path $repoRoot 'VERSION'

if (-not (Test-Path $versionFile)) {
    Write-Error "VERSION file not found: $versionFile"
    exit 1
}

function Get-Version {
    Get-Content -Path $versionFile -Raw | ForEach-Object { $_.Trim() }
}

function Update-CargoVersions {
    param([string]$version)

    Get-ChildItem -Path (Join-Path $repoRoot 'crates') -Filter Cargo.toml -Recurse | ForEach-Object {
        $lines = Get-Content -Path $_.FullName
        $section = ''
        $updated = $false

        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            if ($line -match '^[ \t]*\[(.+?)\][ \t]*$') {
                $section = $matches[1]
            }

            if (($section -eq 'package' -or $section -eq 'package.metadata.bundle') -and $line -match '^[ \t]*version[ \t]*=[ \t]*".*"[ \t]*$') {
                $lines[$i] = "version = \"$version\""
                $updated = $true
            }
        }

        if ($updated) {
            Set-Content -Path $_.FullName -Value $lines
        }
    }
}

if ($Action -eq 'get') {
    Get-Version
    exit 0
}

if ($Action -eq 'set') {
    if (-not $NewVersion) {
        Write-Error 'Usage: version.ps1 set <new-version>'
        exit 1
    }

    $NewVersionTrimmed = $NewVersion.Trim()
    Set-Content -Path $versionFile -Value $NewVersionTrimmed
    Update-CargoVersions -version $NewVersionTrimmed
    Write-Output "Version updated to $NewVersionTrimmed"
    exit 0
}
