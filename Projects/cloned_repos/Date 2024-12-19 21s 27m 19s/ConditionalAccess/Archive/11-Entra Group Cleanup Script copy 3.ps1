# Requires -Modules Microsoft.Graph.Groups, PSWriteHTML

function Get-RecentEntraGroups {
    [CmdletBinding()]
    param (
        [int]$HoursBack = 1
    )
    
    $groups = [System.Collections.Generic.List[PSCustomObject]]::new()
    $filterDate = (Get-Date).AddHours(-$HoursBack).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-Verbose "Filtering groups created after: $filterDate"
    
    # Build query components
    $baseUri = "https://graph.microsoft.com/v1.0/groups"
    $filter = "`$filter=createdDateTime ge $filterDate"
    $select = "`$select=id,displayName,description,createdDateTime"
    $orderBy = "`$orderby=createdDateTime desc"
    
    # URL encode the components
    $encodedFilter = [System.Web.HttpUtility]::UrlEncode($filter)
    $encodedSelect = [System.Web.HttpUtility]::UrlEncode($select)
    $encodedOrderBy = [System.Web.HttpUtility]::UrlEncode($orderBy)
    
    $uri = "$baseUri`?$encodedFilter&$encodedSelect&$encodedOrderBy"
    
    do {
        try {
            Write-Verbose "Executing request: $uri"
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{
                "ConsistencyLevel" = "eventual"
                "Prefer" = "odata.maxpagesize=999"
            }
            
            if ($response.value) {
                $response.value | ForEach-Object { 
                    # Only add groups that are actually within our time window
                    $createdDate = [DateTime]$_.createdDateTime
                    if ($createdDate -ge (Get-Date).AddHours(-$HoursBack)) {
                        [void]$groups.Add($_)
                    }
                }
            }
            
            $uri = $response.'@odata.nextLink'
        }
        catch {
            Write-Error "Failed to retrieve groups: $_"
            Write-Verbose "Full error details: $($_.Exception.Message)"
            return $null
        }
    } while ($uri)
    
    Write-Verbose "Found $($groups.Count) groups created in the last $HoursBack hour(s)"
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
    
    # Add System.Web for URL encoding
    Add-Type -AssemblyName System.Web
    
    # Connect to Microsoft Graph if not already connected
    try {
        $context = Get-MgContext -ErrorAction Stop
        Write-Host "Connected to Microsoft Graph as: $($context.Account)" -ForegroundColor Green
    }
    catch {
        Write-Host "Please connect to Microsoft Graph first using Connect-MgGraph -Scopes 'Group.ReadWrite.All'" -ForegroundColor Yellow
        return
    }

    $recentGroups = Get-RecentEntraGroups -Verbose
    
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