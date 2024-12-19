# Connect to Microsoft Graph with the necessary permissions
Connect-MgGraph -Scopes 'Policy.Read.All', 'Policy.ReadWrite.ConditionalAccess'

# Confirm before proceeding as this will delete all Conditional Access policies
$confirmation = Read-Host "Are you sure you want to delete ALL Conditional Access policies? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    Disconnect-MgGraph
    exit
}

# Fetch all Conditional Access policies
$policies = Get-MgIdentityConditionalAccessPolicy

# Iterate and delete each policy
foreach ($policy in $policies) {
    $policyId = $policy.Id
    $policyName = $policy.DisplayName
    try {
        # Assuming there's a cmdlet to remove policies by ID. If not, you may need to use Invoke-MgGraphRequest with the appropriate HTTP method and URI
        Remove-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId -Force
        Write-Output "Successfully deleted Conditional Access policy: $policyName ($policyId)"
    } catch {
        Write-Error "Failed to delete Conditional Access policy: $policyName ($policyId). Error: $_"
    }
}

# Disconnect from Microsoft Graph
Disconnect-MgGraph
