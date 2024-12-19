# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()

    try {
        do {
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
            $policies = $response.Value
            $allPolicies += $policies

            $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
        } while ($uri)

        Write-Output "Total Conditional Access Policies: $($allPolicies.Count)"
        foreach ($policy in $allPolicies) {
            # Write-Output "Policy ID: $($policy.id)`nPolicy Name: $($policy.displayName)`nStatus: $($policy.state)`n---"
        }
    } catch {
        Write-Error "Failed to retrieve Conditional Access Policies. Error: $_"
    }
}

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


function Add-GroupToConditionalAccessPoliciesUsingBeta {
    param (
        [string]$GroupId
    )

    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $policies = @()

    do {
        $response = Invoke-MgGraphRequest -Method GET -Uri $uri
        $policies += $response.Value
        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    foreach ($policy in $policies) {
        $includeGroups = $policy.conditions.users.includeGroups
        $excludeGroups = $policy.conditions.users.excludeGroups

        # Display excluded groups before the update
        Write-Output "Policy: $($policy.displayName)"
        Write-Output "Excluded Groups Before Update: $($excludeGroups -join ', ')"

        if ($includeGroups -notcontains $GroupId) {
            $includeGroups += $GroupId
            $policy.conditions.users.includeGroups = $includeGroups

            # Update the Conditional Access Policy using the Beta endpoint
            $updateUri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies/$($policy.id)"
            $body = $policy | ConvertTo-Json -Depth 10
            Invoke-MgGraphRequest -Method PATCH -Uri $updateUri -Body $body -ContentType "application/json"

            # Display the updated lists of included and excluded groups
            Write-Output "Added Group to Policy: $($policy.displayName)"
            Write-Output "Included Groups After Update: $($policy.conditions.users.includeGroups -join ', ')"
            Write-Output "Excluded Groups After Update: $($policy.conditions.users.excludeGroups -join ', ')"
            Write-Output "---"
        }
    }
}



Get-ConditionalAccessPoliciesViaMgGraph

# Uncomment the following lines to create a new group and add it to Conditional Access Policies
$groupId = New-ConditionalAccessGroup -GroupName "SG007 - Conditional Access - Exclusion Group"
Add-GroupToConditionalAccessPoliciesUsingBeta -GroupId $groupId
