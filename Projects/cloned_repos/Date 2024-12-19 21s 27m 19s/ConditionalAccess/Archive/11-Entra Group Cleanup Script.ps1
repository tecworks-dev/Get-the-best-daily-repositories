# Requires -Modules Microsoft.Graph.Groups, PSWriteHTML

function Get-RecentEntraGroups {
    param (
        [int]$HoursBack = 1
    )
    
    $groups = [System.Collections.Generic.List[PSCustomObject]]::new()
    # Get timestamp in ISO 8601 format without milliseconds
    $filterDate = (Get-Date).AddHours(-$HoursBack).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:00Z")
    Write-Verbose "Filtering groups created after: $filterDate"
    
    # URL encode the filter value and build the query
    $encodedDate = [System.Web.HttpUtility]::UrlEncode($filterDate)
    $baseUri = "https://graph.microsoft.com/v1.0/groups"
    $select = "`$select=id,displayName,description,createdDateTime"
    $filter = "`$filter=createdDateTime ge $encodedDate"
    $orderBy = "`$orderby=createdDateTime desc"
    $uri = "$baseUri?$filter&$select&$orderBy"
    
    do {
        try {
            Write-Verbose "Executing request: $uri"
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{
                "Prefer" = "odata.maxpagesize=999"
                "ConsistencyLevel" = "eventual"
            }
            
            if ($response.value) {
                $response.value | ForEach-Object { [void]$groups.Add($_) }
            }
            
            $uri = $response.'@odata.nextLink'
        }
        catch {
            Write-Error "Failed to retrieve groups: $_"
            Write-Verbose "Full error details: $($_.Exception.Message)"
            if ($_.ErrorDetails) {
                Write-Verbose "Error details: $($_.ErrorDetails)"
            }
            return $null
        }
    } while ($uri)
    
    Write-Output $groups
}

function Remove-EntraGroupsSafely {
    param (
        [Parameter(Mandatory)]
        [object[]]$Groups
    )
    
    # Prepare report data
    $reportData = $Groups | Select-Object DisplayName, Description, @{
        Name = 'CreatedDateTime'
        Expression = { [DateTime]$_.CreatedDateTime }
    }

    # Generate HTML report
    $htmlParams = @{
        FilePath = ".\GroupDeletionReport_$(Get-Date -Format 'yyyyMMdd_HHmmss').html"
        Title    = "Entra Groups Pending Deletion"
        TableData = $reportData
    }
    
    New-HTML @htmlParams

    # Export CSV for backup
    $csvPath = ".\GroupDeletionBackup_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
    $reportData | Export-Csv -Path $csvPath -NoTypeInformation

    # Display summary in console
    Write-Host "`nGroups to be deleted:" -ForegroundColor Yellow
    $reportData | Format-Table -AutoSize

    # Ask for confirmation
    $confirmation = Read-Host "`nDo you want to proceed with deleting these groups? (Y/N)"
    
    if ($confirmation -eq 'Y') {
        $deleted = [System.Collections.ArrayList]::new()
        $failed = [System.Collections.ArrayList]::new()

        foreach ($group in $Groups) {
            try {
                $deleteUri = "https://graph.microsoft.com/v1.0/groups/$($group.id)"
                Invoke-MgGraphRequest -Uri $deleteUri -Method DELETE
                [void]$deleted.Add($group)
                Write-Host "Deleted group: $($group.DisplayName)" -ForegroundColor Green
            }
            catch {
                [void]$failed.Add(@{
                    Group = $group.DisplayName
                    Error = $_.Exception.Message
                })
                Write-Host "Failed to delete group $($group.DisplayName): $_" -ForegroundColor Red
            }
        }

        # Generate deletion results report
        $resultsReport = @{
            FilePath = ".\DeletionResults_$(Get-Date -Format 'yyyyMMdd_HHmmss').html"
            Title    = "Group Deletion Results"
        }

        New-HTML @resultsReport -Content {
            New-HTMLSection -HeaderText "Successfully Deleted Groups" {
                New-HTMLTable -DataTable $deleted
            }
            if ($failed.Count -gt 0) {
                New-HTMLSection -HeaderText "Failed Deletions" {
                    New-HTMLTable -DataTable $failed
                }
            }
        }
    }
    else {
        Write-Host "Operation cancelled by user." -ForegroundColor Yellow
    }
}

function Start-EntraGroupCleanup {
    [CmdletBinding()]
    param()
    
    # Connect to Microsoft Graph if not already connected
    try {
        $context = Get-MgContext -ErrorAction Stop
        Write-Host "Connected to Microsoft Graph as: $($context.Account)" -ForegroundColor Green
    }
    catch {
        Write-Host "Please connect to Microsoft Graph first using Connect-MgGraph -Scopes 'Group.ReadWrite.All'" -ForegroundColor Yellow
        return
    }

    # Add System.Web for URL encoding
    Add-Type -AssemblyName System.Web

    $recentGroups = Get-RecentEntraGroups
    
    if ($null -eq $recentGroups) {
        Write-Host "No groups found or error occurred." -ForegroundColor Yellow
        return
    }
    
    if ($recentGroups.Count -eq 0) {
        Write-Host "No groups found created in the past hour." -ForegroundColor Green
        return
    }
    
    Remove-EntraGroupsSafely -Groups $recentGroups
}


Start-EntraGroupCleanup -Verbose