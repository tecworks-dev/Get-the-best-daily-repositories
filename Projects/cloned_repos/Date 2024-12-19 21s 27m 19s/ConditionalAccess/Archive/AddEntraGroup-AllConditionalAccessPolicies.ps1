# Install-Module Microsoft.Graph -Scope allusers -AllowClobber


Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All"


function Get-ConditionalAccessPolicies {
    $policies = Get-MgConditionalAccessPolicy
    Write-Output "Total Conditional Access Policies: $($policies.Count)"
    $policies | Select-Object Id, DisplayName
}



function New-ConditionalAccessGroup {
    param (
        [string]$GroupName
    )

    $group = New-MgGroup -DisplayName $GroupName -MailEnabled:$false -MailNickName "NotSet" -SecurityEnabled:$true -GroupTypes "Unified"
    Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    return $group.Id
}




function Add-GroupToConditionalAccessPolicies {
    param (
        [string]$GroupId
    )

    $policies = Get-MgConditionalAccessPolicy

    foreach ($policy in $policies) {
        $includeGroups = $policy.Conditions.Users.IncludeGroups
        if (-not $includeGroups.Contains($GroupId)) {
            $includeGroups += $GroupId
            $policy.Conditions.Users.IncludeGroups = $includeGroups
            Update-MgConditionalAccessPolicy -ConditionalAccessPolicyId $policy.Id -Conditions $policy.Conditions
            Write-Output "Added Group to Policy: $($policy.DisplayName)"
        }
    }
}



Get-ConditionalAccessPolicies

$groupId = New-ConditionalAccessGroup -GroupName "sg001 - Global - Exclusion - All Conditional Access Policies"


Add-GroupToConditionalAccessPolicies -GroupId $groupId


