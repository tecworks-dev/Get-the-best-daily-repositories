

Start-Transcript -Path "C:\Code\CB\Entra\RAC\Graph\Exclude_GroupFromAllCAPoliciesUsingBeta-v4.log"
# Install the Microsoft Graph Beta module if not already installed
# Install-Module Microsoft.Graph.Beta -Scope CurrentUser -AllowClobber -Force

# Import the Microsoft Graph Beta module
# Import-Module Microsoft.Graph.Beta


# Import-Module Microsoft.Graph.Identity.SignIns


# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

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
    # Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    return $group.Id
}


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


function Exclude-GroupFromAllCAPoliciesUsingBeta {
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

        # Prepare the conditions parameter
        $conditions = @{
            users = @{
                excludeGroups = $updatedExcludeGroups
            }
        }

        # Update the Conditional Access Policy using the Beta endpoint
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
            Write-Output "Updated Policy: $($policy.displayName) to exclude Group '$ExcludeGroupId'"
        } catch {
            Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
        }
    }
}


$allconditionalaccesspoliciescount = (Get-ConditionalAccessPoliciesViaMgGraph).Count
$allconditionalaccesspoliciescount

# Uncomment the following lines to create a new group and add it to Conditional Access Policies
# $groupId = New-ConditionalAccessGroup -GroupName "SG001 - Conditional Access - GLOBAL - DMZ - Exclusion Group"

# $groupId = '4ecfb1e1-d76d-464a-866a-203bd77815c2' #CCI
# $groupId = '2b56f107-e156-4ba6-ac5d-0db1e412a1d5' #Bellwoods
$groupId = '1e83e83f-6b88-4e6e-b979-37ad3c938d7e' #RAC
# $groupId = '7b8fdb31-6ae0-460d-8c4c-afc671f52ecf' #Sandbox

# Exclude-GroupFromAllCAPolicies -ExcludeGroupId $groupId
Exclude-GroupFromAllCAPoliciesUsingBeta -ExcludeGroupId $groupId



Stop-Transcript