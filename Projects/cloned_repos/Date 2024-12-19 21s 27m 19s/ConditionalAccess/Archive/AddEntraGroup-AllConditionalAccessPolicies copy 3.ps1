# Connect to Microsoft Graph using the Beta endpoint
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All" -UseBeta

function Get-ConditionalAccessPoliciesViaMgGraph {
    # Use the beta endpoint for Conditional Access policies
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()

    try {
        do {
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
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
    $group = New-MgGroup @groupParams -UseBeta
    Write-Output "Group Created: $($group.DisplayName) with ID: $($group.Id)"
    return $group.Id
}

function Add-GroupToConditionalAccessPolicies {
    param (
        [string]$GroupId
    )

    $policies = Get-MgIdentityConditionalAccessPolicy -All -UseBeta

    foreach ($policy in $policies) {
        $includeGroups = $policy.Conditions.Users.IncludeGroups
        if ($includeGroups -notcontains $GroupId) {
            $includeGroups += $GroupId
            $updatedConditions = $policy.Conditions | Select-Object *
            $updatedConditions.Users.IncludeGroups = $includeGroups
            Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.Id -Conditions $updatedConditions -UseBeta
            Write-Output "Added Group to Policy: $($policy.DisplayName)"
        }
    }
}

Get-ConditionalAccessPoliciesViaMgGraph

# Uncomment the following lines to create a new group and add it to Conditional Access Policies
# $groupId = New-ConditionalAccessGroup -GroupName "YourGroupName"
# Add-GroupToConditionalAccessPolicies -GroupId $groupId
