# Install the required modules if not already installed
Install-Module Microsoft.Graph.Authentication -Scope allusers
Install-Module Microsoft.Graph.Identity.SignIns -Scope AllUsers

# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess"

# Confirm before proceeding as this will delete all Conditional Access policies
$confirmation = Read-Host "Are you sure you want to delete ALL Conditional Access policies? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    exit
}

# Fetch all Conditional Access policies
$policies = Get-MgConditionalAccessPolicy

# Iterate and delete each policy
foreach ($policy in $policies) {
    $policyId = $policy.Id
    $policyName = $policy.DisplayName
    try {
        Remove-MgConditionalAccessPolicy -ConditionalAccessPolicyId $policyId -Force
        Write-Output "Successfully deleted Conditional Access policy: $policyName ($policyId)"
    } catch {
        Write-Error "Failed to delete Conditional Access policy: $policyName ($policyId). Error: $_"
    }
}

# Disconnect from Microsoft Graph
Disconnect-MgGraph
