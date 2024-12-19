# Method 1: Using Get-Content and ConvertFrom-Json
function Process-Json-GetContent {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $jsonContent = Get-Content -Path $JsonFilePath -Raw
    $jsonData = $jsonContent | ConvertFrom-Json

    # Simulate processing
    $jsonData.Count

    $stopwatch.Stop()
    Write-Host "Method 1: Get-Content and ConvertFrom-Json took $($stopwatch.Elapsed.TotalMilliseconds) ms"
}

# Usage
Process-Json-GetContent -JsonFilePath "C:\log.json"