# Method 2: Using System.IO.StreamReader and ConvertFrom-Json
function Process-Json-StreamReader {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $streamReader = [System.IO.StreamReader]::new($JsonFilePath)
    $jsonContent = $streamReader.ReadToEnd()
    $streamReader.Close()
    $jsonData = $jsonContent | ConvertFrom-Json

    # Simulate processing
    $jsonData.Count

    $stopwatch.Stop()
    Write-Host "Method 2: StreamReader and ConvertFrom-Json took $($stopwatch.Elapsed.TotalMilliseconds) ms"
}

# Usage
Process-Json-StreamReader -JsonFilePath "C:\log.json"