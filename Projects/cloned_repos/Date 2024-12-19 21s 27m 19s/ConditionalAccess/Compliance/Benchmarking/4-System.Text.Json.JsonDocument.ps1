# Load System.Text.Json assembly
Add-Type -AssemblyName System.Text.Json

# Method 4: Using System.Text.Json.JsonDocument with an Efficient File Open
function Process-Json-JsonDocument {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    # Open the file using FileStream with buffering and SequentialScan
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

    # Simulate processing
    $elementCount = $jsonDoc.RootElement.GetArrayLength()

    $fileStream.Close()

    $stopwatch.Stop()
    Write-Host "Method 4: System.Text.Json.JsonDocument with system.io.filestream took $($stopwatch.Elapsed.TotalMilliseconds) ms"
}

# Usage
Process-Json-JsonDocument -JsonFilePath "C:\log.json"