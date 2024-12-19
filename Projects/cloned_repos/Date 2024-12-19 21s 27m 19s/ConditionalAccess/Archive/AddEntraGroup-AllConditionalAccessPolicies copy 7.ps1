# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

# function Get-ConditionalAccessPoliciesViaMgGraph {
#     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
#     $allPolicies = @()

#     try {
#         do {
#             $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
#             $policies = $response.Value
#             $allPolicies += $policies

#             $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#         } while ($uri)

#         Write-Output "Total Conditional Access Policies: $($allPolicies.Count)"
#         foreach ($policy in $allPolicies) {
#             # Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
#         }
#     } catch {
#         Write-Error "Failed to retrieve Conditional Access Policies. Error: $_"
#     }
# }

function New-ConditionalAccessGroup {
    param (
        [string]$GroupName
    )

    $groupParams = @{
        DisplayName = $GroupName
        MailEnabled = $false
        MailNickname = "NotSet"
        SecurityEnabled = $true
    }
    $group = New-MgGroup @groupParams
    Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    return $group.Id
}


# function Add-GroupToConditionalAccessPoliciesUsingBeta {
#     param (
#         [string]$GroupId
#     )

#     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
#     $policies = @()

#     do {
#         $response = Invoke-MgGraphRequest -Method GET -Uri $uri
#         $policies += $response.Value
#         $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#     } while ($uri)

#     foreach ($policy in $policies) {
#         $includeGroups = $policy.conditions.users.includeGroups
#         $excludeGroups = $policy.conditions.users.excludeGroups

#         Write-Output "Policy: $($policy.displayName)"
#         Write-Output "Excluded Groups Before Update: $($excludeGroups -join ', ')"

#         if ($includeGroups -notcontains $GroupId) {
#             $updatedIncludeGroups = $includeGroups + $GroupId

#             # Construct minimal payload for updating the policy
#             $updatePayload = @{
#                 conditions = @{
#                     users = @{
#                         includeGroups = $updatedIncludeGroups
#                     }
#                 }
#             }

#             $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#             $body = $updatePayload | ConvertTo-Json -Depth 10
#             Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"

#             Write-Output "Added Group to Policy: $($policy.displayName)"
#         }
#     }
# }







# function Add-GroupToConditionalAccessPoliciesUsingBeta {
#     param (
#         [string]$GroupId
#     )

#     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
#     $policies = @()

#     do {
#         $response = Invoke-MgGraphRequest -Method GET -Uri $uri
#         $policies += $response.Value
#         $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#     } while ($uri)

#     foreach ($policy in $policies) {
#         $includeGroups = $policy.conditions.users.includeGroups
#         $excludeGroups = $policy.conditions.users.excludeGroups

#         Write-Output "Policy: $($policy.displayName)"
#         Write-Output "Excluded Groups Before Update: $($excludeGroups -join ', ')"

#         if ($includeGroups -notcontains $GroupId) {
#             $updatedIncludeGroups = $includeGroups + $GroupId

#             # Construct the payload for updating the policy's included groups
#             $updatePayload = @{
#                 conditions = @{
#                     users = @{
#                         includeGroups = $updatedIncludeGroups
#                         excludeGroups = $excludeGroups # Retain existing excluded groups
#                     }
#                 }
#             }

#             $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#             $body = $updatePayload | ConvertTo-Json -Depth 10
#             Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"

#             Write-Output "Added Group to Policy: $($policy.displayName)"
#             Write-Output "Included Groups After Update: $($updatedIncludeGroups -join ', ')"
#             Write-Output "Excluded Groups After Update: $($excludeGroups -join ', ')"
#             Write-Output "---"
#         }
#     }
# }




# function Add-GroupToConditionalAccessPoliciesUsingBeta {
#     param (
#         [string]$GroupId
#     )

#     $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
#     $policies = @()

#     do {
#         $response = Invoke-MgGraphRequest -Method GET -Uri $uri
#         $policies += $response.Value
#         $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#     } while ($uri)

#     foreach ($policy in $policies) {
#         $includeGroups = $policy.conditions.users.includeGroups
#         $excludeGroups = $policy.conditions.users.excludeGroups

#         Write-Output "Policy: $($policy.displayName)"
#         Write-Output "Included Groups Before Update: $($includeGroups -join ', ')"
#         Write-Output "Excluded Groups Before Update: $($excludeGroups -join ', ')"

#         if ($includeGroups -notcontains $GroupId) {
#             $updatedIncludeGroups = $includeGroups + $GroupId

#             # Construct the payload for updating the policy's included groups
#             $updatePayload = @{
#                 conditions = @{
#                     users = @{
#                         includeGroups = $updatedIncludeGroups
#                         excludeGroups = $excludeGroups # Retain existing excluded groups
#                     }
#                 }
#             }

#             $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#             $body = $updatePayload | ConvertTo-Json -Depth 10
#             $response = Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"

#             if ($response.StatusCode -eq 204) {
#                 Write-Output "Successfully added group to Policy: $($policy.displayName)"
#             } else {
#                 Write-Output "Failed to add group to Policy: $($policy.displayName)"
#             }

#             # Optionally, retrieve and display the updated policy details
#             $updatedPolicy = Invoke-MgGraphRequest -Method GET -Uri $updateUri
#             $updatedIncludeGroups = $updatedPolicy.conditions.users.includeGroups
#             $updatedExcludeGroups = $updatedPolicy.conditions.users.excludeGroups

#             Write-Output "Included Groups After Update: $($updatedIncludeGroups -join ', ')"
#             Write-Output "Excluded Groups After Update: $($updatedExcludeGroups -join ', ')"
#             Write-Output "---"
#         } else {
#             Write-Output "Group already included in Policy: $($policy.displayName)"
#         }
#     }
# }













# function Update-AllConditionalAccessPolicies {
#     param (
#         [string]$GroupId,  # The ID of the group you want to exclude from all policies
#         [string]$Operation = "Exclude"  # Operation can be "Exclude" or "Include"
#     )

#     # Retrieve all Conditional Access Policies
#     $policies = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/beta/identity/conditionalAccess/policies" | Select-Object -ExpandProperty Value


#     #     do {
#     #     $response = Invoke-MgGraphRequest -Method GET -Uri $uri
#     #     $policies += $response.Value
#     #     $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
#     # } while ($uri)


#     foreach ($policy in $policies) {
#         $policyConditions = $policy.conditions.users

#         # Depending on the operation, update the includeGroups or excludeGroups list
#         if ($Operation -eq "Exclude" -and $GroupId -notin $policyConditions.excludeGroups) {
#             $policyConditions.excludeGroups += $GroupId
#         } elseif ($Operation -eq "Include" -and $GroupId -notin $policyConditions.includeGroups) {
#             $policyConditions.includeGroups += $GroupId
#         } else {
#             Write-Output "No changes required for Policy: $($policy.displayName)"
#             continue
#         }

#         # Construct the payload for updating the policy's conditions
#         $updatePayload = @{
#             conditions = @{
#                 users = @{
#                     includeUsers = $policyConditions.includeUsers
#                     excludeUsers = $policyConditions.excludeUsers
#                     includeGroups = $policyConditions.includeGroups
#                     excludeGroups = $policyConditions.excludeGroups
#                 }
#             }
#         }

#         $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#         $body = $updatePayload | ConvertTo-Json -Depth 10

#         # Update the Conditional Access Policy
#         Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"
#         Write-Output "Updated Policy: $($policy.displayName)"
#     }
# }

# Example usage:
# $GroupId = "Your-Group-ID-To-Exclude"
# Update-AllConditionalAccessPolicies -GroupId $GroupId -Operation "Exclude"











# function Update-AllConditionalAccessPolicies {
#     param (
#         [string]$GroupId,
#         [string]$Operation = "Exclude"
#     )

#     # Retrieve all Conditional Access Policies
#     $response = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    
#     if (-not $response -or -not $response.Value) {
#         Write-Error "Failed to retrieve Conditional Access Policies."
#         return
#     }

#     $policies = $response.Value

#     foreach ($policy in $policies) {
#         $policyConditions = $policy.conditions.users

#         if ($Operation -eq "Exclude" -and $GroupId -notin $policyConditions.excludeGroups) {
#             $policyConditions.excludeGroups += $GroupId
#         } elseif ($Operation -eq "Include" -and $GroupId -notin $policyConditions.includeGroups) {
#             $policyConditions.includeGroups += $GroupId
#         } else {
#             Write-Output "No changes required for Policy: $($policy.displayName)"
#             continue
#         }

#         $updatePayload = @{
#             conditions = @{
#                 users = @{
#                     includeUsers = $policyConditions.includeUsers
#                     excludeUsers = $policyConditions.excludeUsers
#                     includeGroups = $policyConditions.includeGroups
#                     excludeGroups = $policyConditions.excludeGroups
#                 }
#             }
#         }

#         $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
#         $body = $updatePayload | ConvertTo-Json -Depth 10

#         # Attempt to update the Conditional Access Policy
#         try {
#             Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"
#             Write-Output "Updated Policy: $($policy.displayName)"
#         } catch {
#             Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
#         }
#     }
# }














# Import-Module Microsoft.Graph.Identity.SignIns

# function Exclude-GroupFromAllCAPolicies {
#     param (
#         [string]$ExcludeGroupId  # The ID of the group to be excluded
#     )

#     # Retrieve all Conditional Access Policies
#     $policies = Get-MgIdentityConditionalAccessPolicy

#     foreach ($policy in $policies) {
#         # Check if the group is already excluded
#         if ($policy.Conditions.Users.ExcludeGroups -contains $ExcludeGroupId) {
#             Write-Output "Group '$ExcludeGroupId' is already excluded in Policy: $($policy.DisplayName)"
#             continue
#         }

#         # Add the group to the excludeGroups array
#         $updatedExcludeGroups = $policy.Conditions.Users.ExcludeGroups + $ExcludeGroupId

#         # Prepare the parameters for updating the policy
#         $params = @{
#             Conditions = @{
#                 Applications = $policy.Conditions.Applications
#                 Users = @{
#                     IncludeUsers = $policy.Conditions.Users.IncludeUsers
#                     ExcludeUsers = $policy.Conditions.Users.ExcludeUsers
#                     IncludeGroups = $policy.Conditions.Users.IncludeGroups
#                     ExcludeGroups = $updatedExcludeGroups
#                     IncludeRoles = $policy.Conditions.Users.IncludeRoles
#                     ExcludeRoles = $policy.Conditions.Users.ExcludeRoles
#                 }
#                 ClientApplications = $policy.Conditions.ClientApplications
#                 Platforms = $policy.Conditions.Platforms
#                 Locations = $policy.Conditions.Locations
#                 UserRiskLevels = $policy.Conditions.UserRiskLevels
#                 SignInRiskLevels = $policy.Conditions.SignInRiskLevels
#                 SignInRiskDetections = $policy.Conditions.SignInRiskDetections
#                 ClientAppTypes = $policy.Conditions.ClientAppTypes
#                 Times = $policy.Conditions.Times
#                 Devices = $policy.Conditions.Devices
#                 ServicePrincipalRiskLevels = $policy.Conditions.ServicePrincipalRiskLevels
#                 AuthenticationFlows = $policy.Conditions.AuthenticationFlows
#             }
#             DisplayName = $policy.DisplayName
#             GrantControls = $policy.GrantControls
#             SessionControls = $policy.SessionControls
#             State = $policy.State
#         }

#         # Update the Conditional Access Policy
#         Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.Id -BodyParameter $params
#         Write-Output "Updated Policy: $($policy.DisplayName) to exclude Group '$ExcludeGroupId'"
#     }
# }

# Example usage:
# $ExcludeGroupId = "e5519b8e-7483-47c5-8286-dfacdfe58b37"  # Replace with the actual Group ID to exclude
# Exclude-GroupFromAllCAPolicies -ExcludeGroupId $ExcludeGroupId






# Example usage:
# $GroupId = "Your-Group-ID-To-Exclude"
# Update-AllConditionalAccessPolicies -GroupId $GroupId -Operation "Exclude"


















# Ensure the Microsoft.Graph.Identity.SignIns module is imported
# Import-Module Microsoft.Graph.Identity.SignIns



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

function Exclude-GroupFromAllCAPolicies {
    param (
        [string]$ExcludeGroupId  # The ID of the group to be excluded
    )

    # Retrieve all Conditional Access Policies using the custom function
    $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

    foreach ($policy in $allPolicies) {
        # Check if the group is already excluded
        if ($policy.conditions.users.excludeGroups -contains $ExcludeGroupId) {
            Write-Output "Group '$ExcludeGroupId' is already excluded in Policy: $($policy.displayName)"
            continue
        }

        # Add the group to the excludeGroups array
        $updatedExcludeGroups = $policy.conditions.users.excludeGroups + $ExcludeGroupId

        # Prepare the parameters for updating the policy, only modifying the excludeGroups under users conditions
        $params = @{
            Conditions = @{
                Users = @{
                    ExcludeGroups = $updatedExcludeGroups
                }
            }
        }

        # Update the Conditional Access Policy
        try {
            Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -BodyParameter $params
            Write-Output "Updated Policy: $($policy.displayName) to exclude Group '$ExcludeGroupId'"
        } catch {
            Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
        }
    }
}



# Example usage:
# $ExcludeGroupId = "Your-Group-ID"  # Replace with the actual Group ID to exclude
# Exclude-GroupFromAllCAPolicies -ExcludeGroupId $ExcludeGroupId


















$allconditionalaccesspoliciescount = (Get-ConditionalAccessPoliciesViaMgGraph).Count

# Uncomment the following lines to create a new group and add it to Conditional Access Policies
$groupId = New-ConditionalAccessGroup -GroupName "SG012 - Conditional Access - Exclusion Group"


# $groupId
# Add-GroupToConditionalAccessPoliciesUsingBeta -GroupId $groupId


# Update-AllConditionalAccessPolicies -GroupId $GroupId -Operation "Exclude"




# $ExcludeGroupId = "e5519b8e-7483-47c5-8286-dfacdfe58b37"  # Replace with the actual Group ID to exclude
Exclude-GroupFromAllCAPolicies -ExcludeGroupId $groupId








# function Update-ConditionalAccessPolicy {
#     param (
#         [string]$conditionalAccessPolicyId,
#         [string]$groupId,
#         [string]$policyState = "enabled"
#     )

#     # Import the required module
#     Import-Module Microsoft.Graph.Identity.SignIns

#     # Define the policy parameters
#     $params = @{
#         conditions = @{
#             users = @{
#                 includeUsers = @("All")  # Include all users
#                 excludeGroups = @($groupId)  # Exclude the specified group
#             }
#             # Add other conditions as needed
#         }
#         displayName = "Custom Policy Name"  # Update with your policy name
#         state = $policyState  # Set the policy state ('enabled' or 'disabled')
#         # Define other policy properties such as grantControls, sessionControls, etc., as needed
#     }

#     # Update the Conditional Access Policy
#     Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $conditionalAccessPolicyId -BodyParameter $params

#     # Optionally, retrieve and display details of the updated policy
#     # Import-Module Microsoft.Graph.Beta.Identity.SignIns
#     # Get-MgBetaIdentityConditionalAccessPolicy -Filter "id eq '$conditionalAccessPolicyId'" | Select-Object id, displayName, state, createdDateTime, modifiedDateTime
# }





