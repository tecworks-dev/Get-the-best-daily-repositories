# Load System.Text.Json assembly
Add-Type -AssemblyName System.Text.Json

# Method 3: Using System.Text.Json.JsonDocument
function Process-Json-JsonDocument {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $fileStream = [System.IO.File]::OpenRead($JsonFilePath)
    $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

    # Simulate processing
    $elementCount = $jsonDoc.RootElement.GetArrayLength()

    $fileStream.Close()

    $stopwatch.Stop()
    Write-Host "Method 3: System.Text.Json.JsonDocument took $($stopwatch.Elapsed.TotalMilliseconds) ms"
}

# Usage
Process-Json-JsonDocument -JsonFilePath "C:\log.json"