# $orphanedUserId = 'b81277fa-d79a-4d9c-853b-154538145947' # Replace with the actual orphaned user ID
$orphanedUserId = 'd4d340c8-78ce-4d3f-959e-ba0e03fffaf8' # Replace with the actual orphaned user ID

Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer" = "odata.maxpagesize=999" }
        $policies = $response.Value
        $allPolicies += $policies

        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies
}



$allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

foreach ($policy in $allPolicies) {
    # Check for orphaned user in excludeUsers
    if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
        # Prepare to update the policy by removing the orphaned user ID from excludeUsers
        $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

        # Prepare the updated conditions without the orphaned user ID
        $conditions = @{
            users = @{
                excludeUsers = $updatedExcludeUsers
            }
        }

        # Update the policy
        # Update the policy
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
            Write-Output "Removed orphaned user from Policy: $($policy.displayName)"
        }
        catch {
            Write-Error "Failed to remove orphaned user from Policy: $($policy.displayName). Error: $($_.Exception.Message)"
            if ($_.Exception.Response) {
                Write-Error "Response: $($_.Exception.Response)"
            }
        }

    }
    
    # Optionally, also check for orphaned user in includeUsers and apply similar logic if needed
    # This depends on whether you need to remove the user from inclusion lists as well.
}