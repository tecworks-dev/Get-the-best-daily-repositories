# Connect to Microsoft Graph with the necessary permissions
Connect-MgGraph -Scopes 'Policy.Read.All', 'Policy.ReadWrite.ConditionalAccess'

# Confirm before proceeding as this will delete all Conditional Access policies
$confirmation = Read-Host "Are you sure you want to delete ALL Conditional Access policies? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    Disconnect-MgGraph
    exit
}



function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $policies = $response.Value
        $allPolicies += $policies

        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies
}


# $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph
$policies = Get-ConditionalAccessPoliciesViaMgGraph

# Fetch all Conditional Access policies
# $policies = Get-MgIdentityConditionalAccessPolicy

# Iterate and attempt to delete each policy
foreach ($policy in $policies) {
    $policyId = $policy.Id
    $policyName = $policy.DisplayName
    try {
        # Adjusted cmdlet call without the -Force parameter
        Remove-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId
        Write-Output "Successfully requested deletion of Conditional Access policy: $policyName ($policyId)"
    } catch {
        Write-Error "Failed to delete Conditional Access policy: $policyName ($policyId). Error: $_"
    }
}


# Disconnect from Microsoft Graph
# Disconnect-MgGraph
