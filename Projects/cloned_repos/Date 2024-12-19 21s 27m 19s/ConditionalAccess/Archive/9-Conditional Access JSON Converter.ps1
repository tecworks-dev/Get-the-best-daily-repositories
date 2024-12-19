# Function to process a single JSON file
function Convert-CAJson {
    param (
        [Parameter(Mandatory)]
        [string]$FilePath
    )

    try {
        # Read the original JSON content
        $originalJson = Get-Content -Path $FilePath -Raw | ConvertFrom-Json

        # Create the new JSON structure
        $newJson = @{
            '@odata.context' = 'https://graph.microsoft.com/beta/$metadata#policies/conditionalAccessPolicies'
            'value' = @($originalJson)
        }

        # Create output file path
        $directory = [System.IO.Path]::GetDirectoryName($FilePath)
        $filename = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
        $outputPath = Join-Path -Path $directory -ChildPath "$filename-formatted.json"

        # Convert and save the new JSON
        $newJson | ConvertTo-Json -Depth 20 | Set-Content -Path $outputPath

        # Return success info
        [PSCustomObject]@{
            OriginalFile = $FilePath
            NewFile = $outputPath
            Status = 'Success'
            Error = $null
        }
    }
    catch {
        # Return error info
        [PSCustomObject]@{
            OriginalFile = $FilePath
            NewFile = $null
            Status = 'Failed'
            Error = $_.Exception.Message
        }
    }
}

# Function to process all files in a directory
function Convert-CADirectory {
    param (
        [Parameter(Mandatory)]
        [string]$DirectoryPath,
        
        [string]$OutputPath = "conversion_results.html"
    )

    # Validate directory exists
    if (-not (Test-Path -Path $DirectoryPath)) {
        Write-Error "Directory not found: $DirectoryPath"
        return
    }

    # Process all JSON files
    $results = Get-ChildItem -Path $DirectoryPath -Filter "*.json" | ForEach-Object {
        Write-Host "Processing $($_.FullName)..."
        Convert-CAJson -FilePath $_.FullName
    }

    # Create HTML report
    $htmlResults = @"
<!DOCTYPE html>
<html>
<head>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .success { color: green; }
        .failed { color: red; }
    </style>
</head>
<body>
    <h1>Conditional Access JSON Conversion Results</h1>
    <table>
        <tr>
            <th>Original File</th>
            <th>New File</th>
            <th>Status</th>
            <th>Error</th>
        </tr>
        $(
            $results | ForEach-Object {
                $statusClass = if ($_.Status -eq 'Success') { 'success' } else { 'failed' }
                "<tr>
                    <td>$($_.OriginalFile)</td>
                    <td>$($_.NewFile)</td>
                    <td class='$statusClass'>$($_.Status)</td>
                    <td>$($_.Error)</td>
                </tr>"
            }
        )
    </table>
</body>
</html>
"@

    # Save HTML report
    $htmlResults | Set-Content -Path $OutputPath

    # Display summary
    $successCount = ($results | Where-Object Status -eq 'Success').Count
    $failCount = ($results | Where-Object Status -eq 'Failed').Count
    
    Write-Host "`nConversion Summary:"
    Write-Host "----------------"
    Write-Host "Successful conversions: $successCount"
    Write-Host "Failed conversions: $failCount"
    Write-Host "Report saved to: $OutputPath"

    # Open the report in default browser
    Start-Process $OutputPath
}

# Example usage:
# Convert-CADirectory -DirectoryPath "C:\code\CA\KVS\cabaseline202409\ConditionalAccess"
Convert-CADirectory -DirectoryPath "C:\CaaC\SandBox\Dec102024-v2\AO\ConditionalAccess"