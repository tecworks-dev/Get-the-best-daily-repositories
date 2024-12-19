# Define the root path
$rootPath = 'C:\Code\Modules'
$outputFile = Join-Path -Path $rootPath -ChildPath 'CombinedModules.ps1'

# Initialize the output file
New-Item -Path $outputFile -ItemType File -Force

# Function to combine files from a directory and its subdirectories
function Combine-AllPS1Files {
    param (
        [string]$directory
    )
    Get-ChildItem -Path $directory -Filter *.ps1 -Recurse | ForEach-Object {
        Get-Content -Path $_.FullName | Add-Content -Path $outputFile
        Add-Content -Path $outputFile -Value "`n"  # Add a new line for separation
    }
}

# Combine all PS1 files in the root path and its subdirectories
Combine-AllPS1Files -directory $rootPath

Write-Host "All PS1 files have been combined into $outputFile"
