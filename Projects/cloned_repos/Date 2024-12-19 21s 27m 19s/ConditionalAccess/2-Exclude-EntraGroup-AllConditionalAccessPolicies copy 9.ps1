

Start-Transcript -Path "C:\Code\CB\Entra\CNA\ConditionalAccess\Exclude_GroupFromAllCAPoliciesUsingBeta-v8.log"
# Install the Microsoft Graph Beta module if not already installed
# Install-Module Microsoft.Graph.Beta -Scope Allusers -AllowClobber -Force
# Install-Module Microsoft.Graph.Groups -Scope Allusers -AllowClobber -Force

# Import the Microsoft Graph Beta module
# Import-Module Microsoft.Graph.Beta


# Import-Module Microsoft.Graph.Identity.SignIns

# Disconnect-MgGraph
# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All", "Application.Read.All"

# Disconnect-MgGraph


# Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All", "Application.Read.All"
function New-ConditionalAccessGroup {
    param (
        [string]$GroupName
    )

    $groupParams = @{
        DisplayName     = $GroupName
        MailEnabled     = $false
        MailNickname    = "NotSet"
        SecurityEnabled = $true
    }
    $group = New-MgGroup @groupParams
    # Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    return $group.Id
}


function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    
    # Initialize a list for better performance
    $allPolicies = [System.Collections.Generic.List[PSObject]]::new()

    do {
        # Fetch the policies via Graph API
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer" = "odata.maxpagesize=999" }
        $policies = $response.Value

        # Add the policies to the list
        $allPolicies.Add($policies)

        # Check for next link for pagination
        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    # Convert the list back to an array (optional)
    return $allPolicies.ToArray()
}


# function Exclude-GroupFromAllCAPoliciesUsingBeta {
#     param (
#         [string]$ExcludeGroupId  # The ID of the group to be excluded
#     )

#     # Retrieve all Conditional Access Policies using the custom function
#     $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

#     foreach ($policy in $allPolicies) {
#         # Check if the group is already excluded
#         if ($policy.conditions.users.excludeGroups -contains $ExcludeGroupId) {
#             Write-Output "Group '$ExcludeGroupId' is already excluded in Policy: $($policy.displayName)"
#             continue
#         }

#         # Add the group to the excludeGroups array
#         $updatedExcludeGroups = $policy.conditions.users.excludeGroups + $ExcludeGroupId

#         # Prepare the conditions parameter
#         $conditions = @{
#             users = @{
#                 excludeGroups = $updatedExcludeGroups
#             }
#         }

#         # Update the Conditional Access Policy using the Beta endpoint
#         try {
#             Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $conditions
#             Write-Output "Updated Policy: $($policy.displayName) to exclude Group '$ExcludeGroupId'"
#         } catch {
#             Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
#         }
#     }
# }





function Exclude-GroupFromAllCAPoliciesUsingBeta {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ExcludeGroupId  # The ID of the group to be excluded
    )

    # Ensure a connection to Microsoft Graph
    # $graphScopes = "Policy.ReadWrite.ConditionalAccess"
    # Connect-MgGraph -Scopes $graphScopes

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

        # Prepare the body parameter for Update-MgBetaIdentityConditionalAccessPolicy
        $bodyParams = @{
            Conditions = @{
                Users = @{
                    ExcludeGroups = $updatedExcludeGroups
                }
            }
        }

        # Update the Conditional Access Policy using the Beta endpoint
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -BodyParameter $bodyParams
            Write-Output "Updated Policy: $($policy.displayName) to exclude Group '$ExcludeGroupId'"
        }
        catch {
            Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
        }
    }

    # Optionally, disconnect from Microsoft Graph after operations are complete
    # Disconnect-MgGraph
}





$groupId = $null


$allconditionalaccesspoliciescount = (Get-ConditionalAccessPoliciesViaMgGraph).Count
$allconditionalaccesspoliciescount

# Uncomment the following lines to create a new group and add it to Conditional Access Policies
# $groupId = New-ConditionalAccessGroup -GroupName "SG002 - Conditional Access - GLOBAL - DMZ - Exclusion Group"


# $groupId = '4ecfb1e1-d76d-464a-866a-203bd77815c2' #CCI
# $groupId = '2b56f107-e156-4ba6-ac5d-0db1e412a1d5' #Bellwoods
# $groupId = '1e83e83f-6b88-4e6e-b979-37ad3c938d7e' #RAC
# $groupId = 'b7f7e66a-a0f8-4113-ade7-987d8bae0cb6' #BCFHT
# $groupId = '7b8fdb31-6ae0-460d-8c4c-afc671f52ecf' #Sandbox
# $groupId = 'eccafc70-52d9-4935-b20b-543507559930' #ICTC
# $groupId = 'f995c07d-5258-4478-ba41-503c9e8bde59' #Antea
# $groupId = 'ef0f9b0f-52a0-4369-a955-aa5703aa8518' #CARMS
# $groupId = '13f371ff-d912-4195-a238-750840cb47d5' #ARH
# $groupId = "05fb21ca-b737-4e48-b557-e40d5feaaa58" #Ambico
# $groupId = "d230864a-3736-4556-b978-fc1a99e624f5" #CNA
# $groupId = "e7ff2f49-96f2-42fb-bcde-053dc9488e6a" #MSFT
$groupId = "7c71e1c5-bd77-4684-9ea9-369baaecd536" #TGB
# $groupId = '' #CPHA

# Exclude-GroupFromAllCAPolicies -ExcludeGroupId $groupId
Exclude-GroupFromAllCAPoliciesUsingBeta -ExcludeGroupId $groupId

Stop-Transcript