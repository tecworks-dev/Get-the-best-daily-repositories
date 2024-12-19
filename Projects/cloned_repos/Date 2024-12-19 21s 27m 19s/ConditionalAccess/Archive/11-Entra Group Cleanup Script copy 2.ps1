# Requires -Modules Microsoft.Graph.Groups, PSWriteHTML

function Test-GraphRequest {
    [CmdletBinding()]
    param()
    
    Write-Host "`n=== Step 1: Testing Graph Connection ===" -ForegroundColor Cyan
    try {
        $context = Get-MgContext -ErrorAction Stop
        Write-Host " Connected as: $($context.Account)" -ForegroundColor Green
    }
    catch {
        Write-Host " Not connected to Graph" -ForegroundColor Red
        return
    }

    Write-Host "`n=== Step 2: Testing Basic Groups Query ===" -ForegroundColor Cyan
    try {
        $testUri = "https://graph.microsoft.com/v1.0/groups?`$top=1"
        Write-Host "Testing URI: $testUri"
        $response = Invoke-MgGraphRequest -Uri $testUri -Method GET
        Write-Host " Basic query successful" -ForegroundColor Green
        Write-Host "Sample response: $($response.value | ConvertTo-Json -Depth 1)"
    }
    catch {
        Write-Host " Basic query failed: $_" -ForegroundColor Red
        return
    }

    Write-Host "`n=== Step 3: Testing Date Filter Construction ===" -ForegroundColor Cyan
    $filterDate = (Get-Date).AddHours(-1).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-Host "Filter date: $filterDate"
    
    # Build query parts
    $baseUri = "https://graph.microsoft.com/v1.0/groups"
    $filter = "`$filter=createdDateTime ge $filterDate"  # Removed quotes around date
    $select = "`$select=id,displayName,description,createdDateTime"
    $orderBy = "`$orderby=createdDateTime desc"
    
    Write-Host "`nQuery components:"
    Write-Host "Base URI: $baseUri" -ForegroundColor Yellow
    Write-Host "Filter: $filter" -ForegroundColor Yellow
    Write-Host "Select: $select" -ForegroundColor Yellow
    Write-Host "Order By: $orderBy" -ForegroundColor Yellow

    Write-Host "`n=== Step 4: Testing Complete Query ===" -ForegroundColor Cyan
    # URL encode the components properly
    $encodedFilter = [System.Web.HttpUtility]::UrlEncode($filter)
    $encodedSelect = [System.Web.HttpUtility]::UrlEncode($select)
    $encodedOrderBy = [System.Web.HttpUtility]::UrlEncode($orderBy)
    
    $fullUri = "$baseUri`?$encodedFilter&$encodedSelect&$encodedOrderBy"
    Write-Host "Full URI: $fullUri"
    
    try {
        Write-Host "`nAttempting first query format..." -ForegroundColor Yellow
        $response = Invoke-MgGraphRequest -Uri $fullUri -Method GET -Headers @{
            "ConsistencyLevel" = "eventual"
            "Prefer" = "odata.maxpagesize=999"
        }
        
        Write-Host " Query successful!" -ForegroundColor Green
        Write-Host "Results found: $($response.value.Count)"
        if ($response.value.Count -gt 0) {
            Write-Host "Sample group: $($response.value[0] | ConvertTo-Json -Depth 1)"
        }
    }
    catch {
        Write-Host " First query format failed, trying alternative..." -ForegroundColor Yellow
        Write-Host "Error details: $($_.ErrorDetails)" -ForegroundColor Red
        
        # Try alternative format
        try {
            $alternativeFilter = "`$filter=createdDateTime ge $($filterDate.Replace('Z', ''))"
            $alternativeUri = "$baseUri`?$alternativeFilter&$select&$orderBy"
            Write-Host "Trying alternative URI: $alternativeUri"
            
            $response = Invoke-MgGraphRequest -Uri $alternativeUri -Method GET -Headers @{
                "ConsistencyLevel" = "eventual"
                "Prefer" = "odata.maxpagesize=999"
            }
            
            Write-Host " Alternative query successful!" -ForegroundColor Green
            Write-Host "Results found: $($response.value.Count)"
            if ($response.value.Count -gt 0) {
                Write-Host "Sample group: $($response.value[0] | ConvertTo-Json -Depth 1)"
            }
        }
        catch {
            Write-Host " Alternative query also failed:" -ForegroundColor Red
            Write-Host "Error message: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Status code: $($_.Exception.Response.StatusCode)" -ForegroundColor Red
            if ($_.ErrorDetails) {
                Write-Host "Error details: $($_.ErrorDetails)" -ForegroundColor Red
            }
        }
    }
}

# Add System.Web for URL encoding
Add-Type -AssemblyName System.Web

# Run the test
Test-GraphRequest -Verbose