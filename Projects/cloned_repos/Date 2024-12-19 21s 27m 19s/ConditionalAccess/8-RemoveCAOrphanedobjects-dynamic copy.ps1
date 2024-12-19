# Function to get all users and groups from Microsoft Graph
function Get-AllUsersAndGroups {
    $allUsers = [System.Collections.Generic.List[PSCustomObject]]::new()
    $allGroups = [System.Collections.Generic.List[PSCustomObject]]::new()
    
    # Get all users
    $uriUsers = "https://graph.microsoft.com/v1.0/users"
    do {
        $responseUsers = Invoke-MgGraphRequest -Uri $uriUsers -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $users = [System.Collections.Generic.List[PSCustomObject]]::new()
        $responseUsers.value | ForEach-Object { [void]$users.Add($_) }
        $allUsers.Add($users)

        $uriUsers = if ($responseUsers.'@odata.nextLink') { $responseUsers.'@odata.nextLink' } else { $null }
    } while ($uriUsers)
    
    # Get all groups
    $uriGroups = "https://graph.microsoft.com/v1.0/groups"
    do {
        $responseGroups = Invoke-MgGraphRequest -Uri $uriGroups -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $groups = [System.Collections.Generic.List[PSCustomObject]]::new()
        $responseGroups.value | ForEach-Object { [void]$groups.Add($_) }
        $allGroups.Add($groups)

        $uriGroups = if ($responseGroups.'@odata.nextLink') { $responseGroups.'@odata.nextLink' } else { $null }
    } while ($uriGroups)
    
    return [PSCustomObject]@{ Users = $allUsers; Groups = $allGroups }
}

# Function to get all conditional access policies
function Get-ConditionalAccessPoliciesViaMgGraph {
    $allPolicies = [System.Collections.Generic.List[PSCustomObject]]::new()
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer" = "odata.maxpagesize=999" }
        $policies = [System.Collections.Generic.List[PSCustomObject]]::new()
        $response.Value | ForEach-Object { [void]$policies.Add($_) }
        $allPolicies.Add($policies)

        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies
}

# Function to remove orphaned users and groups from conditional access policies
function Remove-OrphanedUsersAndGroups {
    param(
        [Parameter(Mandatory = $true)]
        [object]$allEntities,
        [Parameter(Mandatory = $true)]
        [object]$policy
    )

    Process {
        Write-Verbose "Processing Policy: $($policy.displayName)"
        
        $isUpdated = $false

        # Function to validate if a user or group ID is present in the respective entities list
        function Is-Orphaned {
            param (
                [string]$id,
                [System.Collections.Generic.List[PSCustomObject]]$entities
            )
            return (-not ($entities.id -contains $id)) -and ($id -ne "All")
        }

        # Remove orphaned users from includeUsers and excludeUsers
        if ($policy.conditions.users.includeUsers) {
            $orphanedIncludeUsers = $policy.conditions.users.includeUsers | Where-Object { Is-Orphaned -id $_ -entities $allEntities.Users }
            if ($orphanedIncludeUsers) {
                $updatedIncludeUsers = $policy.conditions.users.includeUsers | Where-Object { -not ($orphanedIncludeUsers -contains $_) }
                foreach ($userId in $orphanedIncludeUsers) {
                    Write-Host "Orphaned user found in includeUsers in Policy: $($policy.displayName). User ID: $userId" -ForegroundColor Yellow
                }
                $policy.conditions.users.includeUsers = $updatedIncludeUsers
                $isUpdated = $true
            }
        }

        if ($policy.conditions.users.excludeUsers) {
            $orphanedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { Is-Orphaned -id $_ -entities $allEntities.Users }
            if ($orphanedExcludeUsers) {
                $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { -not ($orphanedExcludeUsers -contains $_) }
                foreach ($userId in $orphanedExcludeUsers) {
                    Write-Host "Orphaned user found in excludeUsers in Policy: $($policy.displayName). User ID: $userId" -ForegroundColor Yellow
                }
                $policy.conditions.users.excludeUsers = $updatedExcludeUsers
                $isUpdated = $true
            }
        }

        # Remove orphaned groups from includeGroups and excludeGroups
        if ($policy.conditions.users.includeGroups) {
            $orphanedIncludeGroups = $policy.conditions.users.includeGroups | Where-Object { Is-Orphaned -id $_ -entities $allEntities.Groups }
            if ($orphanedIncludeGroups) {
                $updatedIncludeGroups = $policy.conditions.users.includeGroups | Where-Object { -not ($orphanedIncludeGroups -contains $_) }
                foreach ($groupId in $orphanedIncludeGroups) {
                    Write-Host "Orphaned group found in includeGroups in Policy: $($policy.displayName). Group ID: $groupId" -ForegroundColor Yellow
                }
                $policy.conditions.users.includeGroups = $updatedIncludeGroups
                $isUpdated = $true
            }
        }

        if ($policy.conditions.users.excludeGroups) {
            $orphanedExcludeGroups = $policy.conditions.users.excludeGroups | Where-Object { Is-Orphaned -id $_ -entities $allEntities.Groups }
            if ($orphanedExcludeGroups) {
                $updatedExcludeGroups = $policy.conditions.users.excludeGroups | Where-Object { -not ($orphanedExcludeGroups -contains $_) }
                foreach ($groupId in $orphanedExcludeGroups) {
                    Write-Host "Orphaned group found in excludeGroups in Policy: $($policy.displayName). Group ID: $groupId" -ForegroundColor Yellow
                }
                $policy.conditions.users.excludeGroups = $updatedExcludeGroups
                $isUpdated = $true
            }
        }

        # Update the policy if there were changes
        if ($isUpdated) {
            try {
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $policy.conditions
                Write-Host "Updated policy: $($policy.displayName)" -ForegroundColor Green
            } catch {
                Write-Error "Failed to update policy: $($policy.displayName). Error: $_"
            }
        } else {
            Write-Host "No orphaned users or groups found in Policy: $($policy.displayName)" -ForegroundColor Cyan
        }
    }
}

# Connect to Microsoft Graph
$graphScopes = "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"
Connect-MgGraph -Scopes $graphScopes -ErrorAction Stop
Write-Verbose "Connected to Microsoft Graph with necessary scopes."

# Retrieve all users and groups
$allEntities = Get-AllUsersAndGroups
Write-Verbose "Retrieved all users and groups."

# Retrieve all Conditional Access Policies
$allPolicies = Get-ConditionalAccessPoliciesViaMgGraph
Write-Verbose "Retrieved all Conditional Access Policies."

foreach ($policy in $allPolicies) {
    Remove-OrphanedUsersAndGroups -allEntities $allEntities -policy $policy -Verbose
}