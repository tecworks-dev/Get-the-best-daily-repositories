# Install the Microsoft.Graph module if not already installed
if (-not (Get-Module -ListAvailable -Name Microsoft.Graph)) {
    Install-Module -Name Microsoft.Graph.Beta.Identity.SignIns -Scope Allusers -Force -AllowClobber
}

# Import the required Microsoft.Graph submodules
# Import-Module Microsoft.Graph.Bate.Identity



# Save the policies to a CSV file
$csvPath = "C:\Code\CB\Entra\Ladco\Graph\export\ConditionalAccessPolicies.csv"
$policies | Export-Csv -Path $csvPath -NoTypeInformation

# Count of policies before deletion
$initialPolicyCount = $policies.Count
Write-Output "Number of Conditional Access policies before deletion: $initialPolicyCount"

# Confirm before proceeding with deletion
$confirmation = Read-Host "Are you sure you want to delete all Conditional Access policies? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    Disconnect-MgGraph
    exit
}

# Iterate through the policies and delete each one by ID
foreach ($policy in $policies) {
    $policyId = $policy.Id
    try {
        # Delete the Conditional Access policy by its ID
        Remove-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId
        Write-Output "Successfully requested deletion of Conditional Access policy with ID: $policyId"
    } catch {
        Write-Error "Failed to delete Conditional Access policy with ID: $policyId. Error: $_"
    }
}

# Retrieve the remaining policies after deletion
$remainingPolicies = Get-MgBetaIdentityConditionalAccessPolicy
$finalPolicyCount = $remainingPolicies.Count
Write-Output "Number of Conditional Access policies after deletion: $finalPolicyCount"

# Disconnect from Microsoft Graph
Disconnect-MgGraph
