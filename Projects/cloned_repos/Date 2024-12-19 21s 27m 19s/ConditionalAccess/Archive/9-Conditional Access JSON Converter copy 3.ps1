#Requires -Modules PSWriteHTML

function Convert-CADirectoryToSingleJson {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$DirectoryPath,
        
        [Parameter()]
        [string]$OutputPath = "conversion_results.html"
    )

    if (-not (Test-Path -Path $DirectoryPath)) {
        Write-Error "Directory not found: $DirectoryPath"
        return
    }

    $allPolicies = [System.Collections.ArrayList]::new()
    $results = [System.Collections.ArrayList]::new()
    $jsonFiles = @(Get-ChildItem -Path $DirectoryPath -Filter "*.json")

    foreach ($file in $jsonFiles) {
        try {
            $policy = Get-Content -Path $file.FullName -Raw | ConvertFrom-Json
            $null = $allPolicies.Add($policy)
            $null = $results.Add([PSCustomObject]@{
                    FileName   = $file.Name
                    Status     = "Success"
                    Error      = ""
                    PolicyName = $policy.displayName
                })
        }
        catch {
            $null = $results.Add([PSCustomObject]@{
                    FileName   = $file.Name
                    Status     = "Failed"
                    Error      = $_.Exception.Message
                    PolicyName = ""
                })
        }
    }

    $combinedJson = @{
        '@odata.context' = 'https://graph.microsoft.com/beta/$metadata#policies/conditionalAccessPolicies'
        'value'          = $allPolicies.ToArray()
    }

    $combinedJson | ConvertTo-Json -Depth 20 | Set-Clipboard

    $successCount = ($results | Where-Object Status -eq "Success").Count
    $failureCount = ($results | Where-Object Status -eq "Failed").Count

    $metadata = @{
        GeneratedBy   = $env:USERNAME
        GeneratedOn   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        TotalPolicies = $results.Count
        SuccessCount  = $successCount
        FailureCount  = $failureCount
    }

    # Generate HTML report
    New-HTML -Title "Conditional Access Policy Conversion Report" -FilePath $OutputPath -ShowHTML {
        New-HTMLSection -HeaderText "Conversion Summary" {
            New-HTMLPanel {
                New-HTMLText -Text @"
            <h3>Report Details</h3>
            <ul>
                <li>Generated By: $($metadata.GeneratedBy)</li>
                <li>Generated On: $($metadata.GeneratedOn)</li>
                <li>Total Policies Processed: $($metadata.TotalPolicies)</li>
                <li>Successful Conversions: $($metadata.SuccessCount)</li>
                <li>Failed Conversions: $($metadata.FailureCount)</li>
            </ul>
"@
            }
        }
    
        New-HTMLSection -HeaderText "Policy Conversion Results" {
            New-HTMLTable -DataTable $results -ScrollX `
                -Buttons @('copyHtml5', 'excelHtml5', 'csvHtml5') `
                -SearchBuilder {
                New-TableCondition -Name 'Status' -ComparisonType string -Operator eq -Value 'Failed' -BackgroundColor Salmon -Color Black
                New-TableCondition -Name 'Status' -ComparisonType string -Operator eq -Value 'Success' -BackgroundColor LightGreen -Color Black
            }
        }
    }

    Write-Host "`nConversion Summary:" -ForegroundColor Cyan
    Write-Host "----------------"
    Write-Host "Successful conversions: $successCount" -ForegroundColor Green
    Write-Host "Failed conversions: $failureCount" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "Green" })
    Write-Host "`nCombined JSON has been copied to clipboard" -ForegroundColor Green
    Write-Host "Report saved to: $OutputPath" -ForegroundColor Green
}


# Example usage:
Convert-CADirectoryToSingleJson -DirectoryPath "C:\code\CA\KVS\cabaseline202409\ConditionalAccess"