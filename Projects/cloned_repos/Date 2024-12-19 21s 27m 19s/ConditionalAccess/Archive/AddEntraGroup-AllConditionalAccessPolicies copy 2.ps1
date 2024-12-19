# Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All" -UseBeta
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"


function Get-ConditionalAccessPoliciesViaMgGraph {
    # Use the beta endpoint for Conditional Access policies
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
# Install-Module Microsoft.Graph -Scope CurrentUser -AllowClobber



# Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"


# function Get-ConditionalAccessPolicies {
#     Connect-MgGraph -Scopes 'Policy.Read.All', 'Policy.ReadWrite.ConditionalAccess'
#     $policies = Get-MgIdentityConditionalAccessPolicy
#     Write-Output "Total Conditional Access Policies: $($policies.Count)"
#     foreach ($policy in $policies) {
#         Write-Output "Policy ID: $($policy.Id), Policy Name: $($policy.DisplayName)"
#     }
#     Disconnect-MgGraph
# }



# function Get-ConditionalAccessPoliciesWithDetails {
#     # Retrieve all Conditional Access policies without manual pagination
#     $policies = Get-MgIdentityConditionalAccessPolicy -All

#     Write-Output "Total Conditional Access Policies: $($policies.Count)"
#     foreach ($policy in $policies) {
#         # Check if there are any excluded groups and join their IDs if present
#         $excludedGroups = if ($policy.Conditions.Users.ExcludeGroups) { $policy.Conditions.Users.ExcludeGroups -join ', ' } else { 'None' }
        
#         # Determine the policy status
#         $status = switch ($policy.State) {
#             "enabled" { "On" }
#             "disabled" { "Off" }
#             "enabledForReportingButNotEnforced" { "Report Only" }
#             Default { "Unknown" }
#         }

#         Write-Output "Policy ID: $($policy.Id)`nPolicy Name: $($policy.DisplayName)`nExcluded Groups: $($excludedGroups)`nStatus: $status`n---"
#     }
# }













# function Get-ConditionalAccessPoliciesViaMgGraph {
#     $uri = "https://graph.microsoft.com/v1.0/identity/conditionalAccess/policies"

#     try {
#         $response = Invoke-MgGraphRequest -Uri $uri -Method GET
#         $policies = $response.Value

#         Write-Output "Total Conditional Access Policies: $($policies.Count)"
#         foreach ($policy in $policies) {
#             # Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
#         }
#     }
#     catch {
#         Write-Error "Failed to retrieve Conditional Access Policies. Error: $_"
#     }
# }



function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/v1.0/identity/conditionalAccess/policies"
    $allPolicies = @()

    try {
        do {
            # $response = Invoke-MgGraphRequest -Uri $uri -Method GET -ConsistencyLevel "eventual" -Headers @{"Prefer"="odata.maxpagesize=999"}
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
            # $response = Invoke-MgGraphRequest -Uri $uri -Method GET
            $policies = $response.Value
            $allPolicies += $policies

            # Check for a nextLink to handle pagination
            $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
        }
        while ($uri)

        Write-Output "Total Conditional Access Policies: $($allPolicies.Count)"
        foreach ($policy in $allPolicies) {
            Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
            # Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
        }
    }
    catch {
        Write-Error "Failed to retrieve Conditional Access Policies. Error: $_"
    }
}












# function Get-ConditionalAccessPoliciesWithDetails {
#     # Retrieve all Conditional Access policies
#     $policies = Get-MgIdentityConditionalAccessPolicy -All

#     Write-Output "Total Conditional Access Policies: $($policies.Count)"
#     foreach ($policy in $policies) {
#         # Initialize variable for excluded groups
#         $excludedGroups = 'None'
#         if ($policy.Conditions.Users.ExcludeGroups) {
#             # Retrieve details of excluded groups
#             $excludedGroupsDetails = Get-MgGroup -Filter "id eq '$($_)'" -Property DisplayName -All
#             # Concatenate the display names of all excluded groups
#             $excludedGroups = ($excludedGroupsDetails | ForEach-Object { $_.DisplayName }) -join ', '
#         }

#         # Determine the policy status
#         $status = switch ($policy.State) {
#             "enabled" { "On" }
#             "disabled" { "Off" }
#             "enabledForReportingButNotEnforced" { "Report Only" }
#             Default { "Unknown" }
#         }

#         Write-Output "Policy ID: $($policy.Id)`nPolicy Name: $($policy.DisplayName)`nExcluded Groups: $($excludedGroups)`nStatus: $status`n---"
#     }
# }









function New-ConditionalAccessGroup {
    param (
        [string]$GroupName
    )

    # Connect-MgGraph -Scopes 'Group.ReadWrite.All'
    $groupParams = @{
        DisplayName = $GroupName
        MailEnabled = $false
        MailNickname = "NotSet"
        SecurityEnabled = $true
    }
    $group = New-MgGroup @groupParams
    Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    Disconnect-MgGraph
    return $group.Id
}




function Add-GroupToConditionalAccessPolicies {
    param (
        [string]$GroupId
    )

    # Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'
    $policies = Get-MgIdentityConditionalAccessPolicy

    foreach ($policy in $policies) {
        $includeGroups = $policy.Conditions.Users.IncludeGroups
        if ($includeGroups -notcontains $GroupId) {
            $includeGroups += $GroupId
            $updatedConditions = $policy.Conditions | Select-Object *
            $updatedConditions.Users.IncludeGroups = $includeGroups
            Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.Id -Conditions $updatedConditions
            Write-Output "Added Group to Policy: $($policy.DisplayName)"
        }
    }
    # Disconnect-MgGraph
}




Get-ConditionalAccessPoliciesViaMgGraph
# Get-ConditionalAccessPoliciesWithDetails



# Get-ConditionalAccessPoliciesViaMgGraph

# $groupId = New-ConditionalAccessGroup -GroupName "YourGroupName"

# $groupId = New-ConditionalAccessGroup -GroupName "sg001 - Global - Exclusion - All Conditional Access Policies"


# Add-GroupToConditionalAccessPolicies -GroupId $groupId





# # Define the endpoint for Conditional Access policies
# $Uri = "https://graph.microsoft.com/v1.0/identity/conditionalAccess/policies"

# # Use Invoke-MgGraphRequest to make the API call
# $AllPolicies = Invoke-MgGraphRequest -Uri $Uri -Method GET


# Write-Output "Total Conditional Access Policies: $($policies.Count)"

# # Process and output the policies
# foreach ($policy in $AllPolicies) {
#     Write-Output "Policy ID: $($policy.id), Display Name: $($policy.displayName)"
# }













function Get-ConditionalAccessPoliciesViaMgGraph {
    # Use the beta endpoint for Conditional Access policies
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()

    try {
        do {
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -ConsistencyLevel "eventual" -Headers @{"Prefer"="odata.maxpagesize=999"}
            $policies = $response.Value
            $allPolicies += $policies

            # Check for a nextLink to handle pagination
            $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
        }
        while ($uri)

        Write-Output "Total Conditional Access Policies: $($allPolicies.Count)"
        foreach ($policy in $allPolicies) {
            Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
        }
    }
    catch {
        Write-Error "Failed to retrieve Conditional Access Policies. Error: $_"
    }
}






Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All" -UseBeta


Get-ConditionalAccessPoliciesViaMgGraph