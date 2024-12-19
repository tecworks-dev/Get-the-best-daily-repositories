$orphanedGroupId = '614050e3-2fa4-4ab3-ad4e-12ef93d805e3' # The ID of the deleted (orphan) group


Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"



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

$allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

foreach ($policy in $allPolicies) {
    if ($policy.conditions.users.excludeGroups -contains $orphanedGroupId) {
        # Prepare to update the policy by removing the orphaned group ID
        $updatedExcludeGroups = $policy.conditions.users.excludeGroups | Where-Object { $_ -ne $orphanedGroupId }

        # Prepare the updated conditions without the orphaned group ID
        $conditions = @{
            users = @{
                excludeGroups = $updatedExcludeGroups
            }
        }

        # Update the policy
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
            Write-Output "Removed orphaned group from Policy: $($policy.displayName)"
        } catch {
            Write-Error "Failed to remove orphaned group from Policy: $($policy.displayName). Error: $_"
        }
    }
}