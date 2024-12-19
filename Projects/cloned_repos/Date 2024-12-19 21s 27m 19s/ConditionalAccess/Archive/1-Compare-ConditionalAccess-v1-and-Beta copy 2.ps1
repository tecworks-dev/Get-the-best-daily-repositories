# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

function Get-ConditionalAccessPoliciesViaMgGraph {
    param (
        [string]$GraphVersion
    )

    $uri = "https://graph.microsoft.com/$GraphVersion/identity/conditionalAccess/policies"
    $allPolicies = @()

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $policies = $response.Value
        $allPolicies += $policies
        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies
}

# Fetch policies using Beta endpoint
$betaPolicies = Get-ConditionalAccessPoliciesViaMgGraph -GraphVersion "beta"

# Fetch policies using v1.0 endpoint
$v1Policies = Get-ConditionalAccessPoliciesViaMgGraph -GraphVersion "v1.0"

# Extract IDs into simple arrays
$betaPolicyIds = $betaPolicies | ForEach-Object { $_.id }
$v1PolicyIds = $v1Policies | ForEach-Object { $_.id }

# Identify deprecated policies by comparing IDs
$deprecatedPolicyIds = $betaPolicyIds | Where-Object { $_ -notin $v1PolicyIds }


# Filter the full policy objects for deprecated policies
$deprecatedPolicies = $betaPolicies | Where-Object { $deprecatedPolicyIds -contains $_.id }



# Output to console in a formatted table
# $deprecatedPolicies | Format-Table id, displayName, createdDateTime, state -AutoSize | Out-GridView
# $deprecatedPolicies | Out-GridView


# $DBG





# Assuming $deprecatedPolicies contains hashtables with the desired keys

# Prepare the data for CSV export and Out-GridView by accessing the hashtable properties directly
$exportData = $deprecatedPolicies | ForEach-Object {
    [PSCustomObject]@{
        id              = $_.id
        displayName     = $_.displayName
        createdDateTime = $_.createdDateTime
        state           = $_.state
    }
}

# Export to CSV
$exportData | Export-Csv -Path "C:\Code\CB\Entra\RAC\Graph\DeprecatedPolicies-v3.csv" -NoTypeInformation

# Display in Out-GridView
$exportData | Out-GridView -Title "Deprecated Policies"

# Output policy details to console in a well-formatted table
Write-Host "Deprecated Policies:"
$exportData | Format-Table -AutoSize


# # Export to CSV
# $deprecatedPolicies | Select-Object id, displayName, createdDateTime, state | Export-Csv -Path "DeprecatedPoliciesv4.csv" -NoTypeInformation

# # Output policy details to console
# Write-Host "Deprecated Policies:"
# $deprecatedPolicies | Format-Table id, displayName, createdDateTime, state -AutoSize
