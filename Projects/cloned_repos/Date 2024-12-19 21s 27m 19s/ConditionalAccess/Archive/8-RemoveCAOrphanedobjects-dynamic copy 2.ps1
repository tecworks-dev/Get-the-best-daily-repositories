# Function to connect to Microsoft Graph
function Connect-ToGraph {
    $graphScopes = "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"
    Connect-MgGraph -Scopes $graphScopes -ErrorAction Stop
    Write-Verbose "Connected to Microsoft Graph with necessary scopes."
}

# Function to get a specific conditional access policy
function Get-ConditionalAccessPolicyById {
    param (
        [string]$policyId
    )
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$policyId"
    $response = Invoke-MgGraphRequest -Uri $uri -Method GET
    return $response
}

# Function to remove a specific orphaned user from the policy
function Remove-OrphanedUserFromPolicy {
    param (
        [string]$policyId,
        [string]$orphanedUserId
    )

    $policy = Get-ConditionalAccessPolicyById -policyId $policyId

    if ($policy -eq $null) {
        Write-Error "Policy with ID $policyId not found."
        return
    }

    Write-Verbose "Processing Policy: $($policy.displayName) (ID: $policyId)"

    $isUpdated = $false

    # Check and remove orphaned user from includeUsers
    if ($policy.conditions.users.includeUsers -contains $orphanedUserId) {
        $updatedIncludeUsers = $policy.conditions.users.includeUsers | Where-Object { $_ -ne $orphanedUserId }
        $policy.conditions.users.includeUsers = $updatedIncludeUsers
        Write-Host "Orphaned user found in includeUsers in Policy: $($policy.displayName). User ID: $orphanedUserId" -ForegroundColor Yellow
        $isUpdated = $true
    }

    # Check and remove orphaned user from excludeUsers
    if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
        $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }
        $policy.conditions.users.excludeUsers = $updatedExcludeUsers
        Write-Host "Orphaned user found in excludeUsers in Policy: $($policy.displayName). User ID: $orphanedUserId" -ForegroundColor Yellow
        $isUpdated = $true
    }

    # Update the policy if changes were made
    if ($isUpdated) {
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId -Conditions $policy.conditions
            Write-Host "Updated policy: $($policy.displayName) (ID: $policyId)" -ForegroundColor Green
        } catch {
            Write-Error "Failed to update policy: $($policy.displayName) (ID: $policyId). Error: $_"
        }

        # Re-fetch the policy to confirm the changes
        $updatedPolicy = Get-ConditionalAccessPolicyById -policyId $policyId
        if ($updatedPolicy.conditions.users.includeUsers -contains $orphanedUserId -or $updatedPolicy.conditions.users.excludeUsers -contains $orphanedUserId) {
            Write-Error "Failed to remove orphaned user ID: $orphanedUserId from policy: $($policy.displayName) (ID: $policyId)"
        } else {
            Write-Host "Successfully removed orphaned user ID: $orphanedUserId from policy: $($policy.displayName) (ID: $policyId)" -ForegroundColor Green
        }
    } else {
        Write-Host "No orphaned users found in Policy: $($policy.displayName) (ID: $policyId)" -ForegroundColor Cyan
    }
}

# Main script
Connect-ToGraph
$policyId = "fd69003d-182c-4cb1-899a-ec54f0a0105d"
$orphanedUserId = "8491705c-5788-4616-9909-59d6894adb2d"
Remove-OrphanedUserFromPolicy -policyId $policyId -orphanedUserId $orphanedUserId
