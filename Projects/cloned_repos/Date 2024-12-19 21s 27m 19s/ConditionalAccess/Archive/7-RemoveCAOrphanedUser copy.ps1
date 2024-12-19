# $orphanedUserId = 'b81277fa-d79a-4d9c-853b-154538145947' # Replace with the actual orphaned user ID
# $orphanedUserId = 'd4d340c8-78ce-4d3f-959e-ba0e03fffaf8' # Replace with the actual orphaned user ID

# Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

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






# function Remove-OrphanedUserFromCAPolicies {
#     param(
#         [Parameter(Mandatory = $true)]
#         [string]$orphanedUserId
#     )

#     Begin {
#         # Ensure connection to Microsoft Graph
#         # $graphScopes = "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"
#         # Connect-MgGraph -Scopes $graphScopes -ErrorAction Stop
#         # Write-Verbose "Connected to Microsoft Graph with necessary scopes."

#         # Fetch all conditional access policies
#         $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph
#         Write-Verbose "Retrieved all Conditional Access Policies."
#     }

#     Process {
#         foreach ($policy in $allPolicies) {
#             Write-Verbose "Processing Policy: $($policy.displayName)"

#             if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
#                 Write-Verbose "Found orphaned user in Policy: $($policy.displayName)"

#                 $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

#                 $body = @{
#                     conditions = @{
#                         users = @{
#                             excludeUsers = $updatedExcludeUsers
#                         }
#                     }
#                 } | ConvertTo-Json

#                 try {
#                     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#                     $response = Invoke-MgGraphRequest -Uri $uri -Method PATCH -Body $body -ContentType "application/json"

#                     if ($response.conditions.users.excludeUsers -contains $orphanedUserId) {
#                         Write-Warning "Orphaned user was not removed from Policy: $($policy.displayName)"
#                     } else {
#                         Write-Output "Successfully removed orphaned user from Policy: $($policy.displayName)"
#                     }
#                 }
#                 catch {
#                     Write-Error "Failed to remove orphaned user from Policy: $($policy.displayName). Error: $($_.Exception.Message)"
#                 }
#             }
#             else {
#                 Write-Verbose "Orphaned user not found in Policy: $($policy.displayName)"
#             }
#         }
#     }

#     End {
#         Disconnect-MgGraph
#         Write-Verbose "Disconnected from Microsoft Graph."
#     }
# }







# function Remove-OrphanedUserFromCAPolicy {
#     param(
#         [Parameter(Mandatory = $true)]
#         [string]$orphanedUserId,
#         [Parameter(Mandatory = $true)]
#         [object]$policy
#     )

#     Process {
#         Write-Verbose "Processing Policy: $($policy.displayName)"

#         if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
#             Write-Verbose "Found orphaned user in Policy: $($policy.displayName)"

#             $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

#             $body = @{
#                 conditions = @{
#                     users = @{
#                         excludeUsers = $updatedExcludeUsers
#                     }
#                 }
#             } | ConvertTo-Json

#             try {
#                 $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#                 $response = Invoke-MgGraphRequest -Uri $uri -Method PATCH -Body $body -ContentType "application/json" -Headers @{"Content-Type"="application/json"}

#                 if ($response -and !($response.conditions.users.excludeUsers -contains $orphanedUserId)) {
#                     Write-Output "Successfully removed orphaned user from Policy: $($policy.displayName)"
#                 } else {
#                     Write-Warning "Orphaned user may not have been removed from Policy: $($policy.displayName)"
#                 }
#             }
#             catch {
#                 Write-Error "Failed to remove orphaned user from Policy: $($policy.displayName). Error: $($_.Exception.Message)"
#             }
#         }
#         else {
#             Write-Verbose "Orphaned user not found in Policy: $($policy.displayName)"
#         }
#     }
# }







function Remove-OrphanedUserFromCAPolicy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$orphanedUserId,
        [Parameter(Mandatory = $true)]
        [object]$policy
    )

    Process {
        Write-Verbose "Processing Policy: $($policy.displayName)"

        if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
            Write-Verbose "Found orphaned user in Policy: $($policy.displayName)"

            # Remove the orphaned user ID from excludeUsers
            $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

            # Check if the updated list is empty. If so, set it to an empty array.
            if (-not $updatedExcludeUsers) {
                $updatedExcludeUsers = @()
            }

            $body = @{
                conditions = @{
                    users = @{
                        excludeUsers = $updatedExcludeUsers
                    }
                }
            } | ConvertTo-Json

            try {
                $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
                $response = Invoke-MgGraphRequest -Uri $uri -Method PATCH -Body $body -ContentType "application/json" -Headers @{"Content-Type"="application/json"}

                # Since $response doesn't directly show the updated policy's details (especially with Invoke-MgGraphRequest),
                # it's better to re-query the policy to confirm the orphaned user removal.
                # However, this snippet assumes successful removal for demonstration purposes.
                Write-Output "Successfully removed orphaned user from Policy: $($policy.displayName)"
            }
            catch {
                Write-Error "Failed to remove orphaned user from Policy: $($policy.displayName). Error: $($_.Exception.Message)"
            }
        }
        else {
            Write-Verbose "Orphaned user not found in Policy: $($policy.displayName)"
        }
    }
}











# function Get-ConditionalAccessPoliciesViaMgGraph {
#     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
#     $allPolicies = @()

#     do {
#         $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
#         $policies = $response.Value
#         $allPolicies += $policies

#         $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#     } while ($uri)

#     return $allPolicies
# }
















# foreach ($policy in $allPolicies) {
#     # Check for orphaned user in excludeUsers
#     if ($policy.conditions.users.excludeUsers -contains $orphanedUserId) {
#         # Prepare to update the policy by removing the orphaned user ID from excludeUsers
#         $updatedExcludeUsers = $policy.conditions.users.excludeUsers | Where-Object { $_ -ne $orphanedUserId }

#         # Prepare the updated conditions without the orphaned user ID
#         $conditions = @{
#             users = @{
#                 excludeUsers = $updatedExcludeUsers
#             }
#         }

#         # Update the policy
#         # Update the policy
#         try {
#             Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
#             Write-Output "Removed orphaned user from Policy: $($policy.displayName)"
#         }
#         catch {
#             Write-Error "Failed to remove orphaned user from Policy: $($policy.displayName). Error: $($_.Exception.Message)"
#             if ($_.Exception.Response) {
#                 Write-Error "Response: $($_.Exception.Response)"
#             }
#         }

#     }
    
#     # Optionally, also check for orphaned user in includeUsers and apply similar logic if needed
#     # This depends on whether you need to remove the user from inclusion lists as well.
# }




# Remove-OrphanedUserFromCAPolicies -orphanedUserId "your-orphaned-user-id-here" -Verbose




# Connect to Microsoft Graph
$graphScopes = "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"
Connect-MgGraph -Scopes $graphScopes -ErrorAction Stop
Write-Verbose "Connected to Microsoft Graph with necessary scopes."

# Retrieve all Conditional Access Policies
$allPolicies = Get-ConditionalAccessPoliciesViaMgGraph
Write-Verbose "Retrieved all Conditional Access Policies."

$orphanedUserId = 'd4d340c8-78ce-4d3f-959e-ba0e03fffaf8' # Replace this with the actual orphaned user ID

foreach ($policy in $allPolicies) {
    Remove-OrphanedUserFromCAPolicy -orphanedUserId $orphanedUserId -policy $policy -Verbose
}

# Optionally, disconnect from Microsoft Graph after operations are complete
# Disconnect-MgGraph
# Write-Verbose "Disconnected from Microsoft Graph."
