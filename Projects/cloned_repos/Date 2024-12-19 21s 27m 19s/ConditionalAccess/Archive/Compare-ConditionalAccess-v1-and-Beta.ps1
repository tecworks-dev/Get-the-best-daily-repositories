function Get-ConditionalAccessPolicies {
    param (
        [string]$version  # "v1.0" or "beta"
    )

    $uri = "https://graph.microsoft.com/$version/identity/conditionalAccess/policies"
    $allPolicies = @()

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $policies = $response.Value
        $allPolicies += $policies

        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies
}

# Retrieve policies from both endpoints
$betaPolicies = Get-ConditionalAccessPolicies -version "beta"
$v1Policies = Get-ConditionalAccessPolicies -version "v1.0"



# $betaPolicies | Format-Table
# $v1Policies | Format-Table


# Compare and identify deprecated policies
# $deprecatedPolicies = $betaPolicies | Where-Object { $_.id -notin $v1Policies.id }





# Extract IDs from v1.0 policies
$v1PolicyIds = $v1Policies | ForEach-Object { $_.id }

# Compare and identify deprecated policies by checking if their IDs are in the list of v1.0 policy IDs
$deprecatedPolicies = $betaPolicies | Where-Object { $v1PolicyIds -notcontains $_.id }

# Proceed with formatting, outputting, and exporting $deprecatedPolicies as before


# Format and output to console
# $deprecatedPolicies | Format-Table id, displayName -AutoSize | Out-Host

$deprecatedPolicies | Out-GridView -Title "Deprecated Conditional Access Policies"

# $DBG

# Export to CSV
$deprecatedPolicies | Select-Object id, displayName | Export-Csv -Path "D:\Code\CB\Entra\CCI\Graph\export\DeprecatedCAPolicies-v2.csv" -NoTypeInformation

Write-Host "Deprecated policies have been exported to 'DeprecatedCAPolicies.csv' and displayed in the console and Out-GridView."
