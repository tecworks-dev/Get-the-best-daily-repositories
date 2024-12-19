function Include-GroupInAllCAPoliciesUsingBeta {
    param (
        [string]$IncludeGroupId  # The ID of the group to be included back
    )

    $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

    foreach ($policy in $allPolicies) {
        # Check if the group is currently excluded
        if ($policy.conditions.users.excludeGroups -contains $IncludeGroupId) {
            # Remove the group from the excludeGroups array
            $updatedExcludeGroups = $policy.conditions.users.excludeGroups | Where-Object { $_ -ne $IncludeGroupId }

            # Prepare the conditions parameter with the updated excludeGroups
            $conditions = @{
                users = @{
                    excludeGroups = $updatedExcludeGroups
                }
            }

            # Update the Conditional Access Policy using the Beta endpoint
            try {
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
                Write-Output "Updated Policy: $($policy.displayName) to include Group '$IncludeGroupId'"
            } catch {
                Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
            }
        }
        else {
            Write-Output "Group '$IncludeGroupId' is not excluded in Policy: $($policy.displayName)"
        }
    }
}

$groupIdToInclude = '1e83e83f-6b88-4e6e-b979-37ad3c938d7e' # Replace this with the actual group ID you want to include
Include-GroupInAllCAPoliciesUsingBeta -IncludeGroupId $groupIdToInclude