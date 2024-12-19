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


function Remove-OrphanedUserFromCAPolicy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$orphanedUserId,
        [Parameter(Mandatory = $true)]
        [object]$policy
    )

    Process {
        Write-Verbose "Processing Policy: $($policy.displayName)"

        # Check if the orphaned user is in the excludeUsers list
        if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
            Write-Verbose "Found orphaned user in Policy: $($policy.displayName)"

            # Remove the orphaned user ID from excludeUsers
            $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

            # Check if updatedExcludeUsers is empty to determine if we need to pass an empty array
            if (-not $updatedExcludeUsers) {
                $updatedExcludeUsers = @()
            }

            # Prepare the body parameter for Update-MgBetaIdentityConditionalAccessPolicy
            $bodyParams = @{
                Conditions = @{
                    Users = @{
                        ExcludeUsers = $updatedExcludeUsers
                    }
                }
            }

            try {
                # Update the policy using Update-MgBetaIdentityConditionalAccessPolicy
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -BodyParameter $bodyParams
                Write-Output "Successfully updated Policy: $($policy.displayName)"
            }
            catch {
                Write-Error "Failed to update Policy: $($policy.displayName). Error: $($_.Exception.Message)"
            }
        }
        else {
            Write-Verbose "Orphaned user not found in Policy: $($policy.displayName)"
        }
    }
}


# Connect to Microsoft Graph
$graphScopes = "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"
Connect-MgGraph -Scopes $graphScopes -ErrorAction Stop
Write-Verbose "Connected to Microsoft Graph with necessary scopes."

# Retrieve all Conditional Access Policies
$allPolicies = Get-ConditionalAccessPoliciesViaMgGraph
Write-Verbose "Retrieved all Conditional Access Policies."


$orphanedUserId = $null
$orphanedUserId = '6303c7ce-c078-47ce-a1f3-9750d8d3ff68' # Replace this with the actual orphaned user ID

foreach ($policy in $allPolicies) {
    Remove-OrphanedUserFromCAPolicy -orphanedUserId $orphanedUserId -policy $policy -Verbose
}

# Optionally, disconnect from Microsoft Graph after operations are complete
# Disconnect-MgGraph
# Write-Verbose "Disconnected from Microsoft Graph."
