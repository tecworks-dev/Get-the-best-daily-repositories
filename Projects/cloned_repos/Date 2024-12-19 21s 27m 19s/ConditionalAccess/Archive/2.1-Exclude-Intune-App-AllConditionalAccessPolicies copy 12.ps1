

# Start-Transcript -Path "C:\Code\CB\Entra\Sandbox\Graph\Exclude_AppFromAllCAPoliciesUsingBeta-v1.log"
# Install the Microsoft Graph Beta module if not already installed
# Install-Module Microsoft.Graph.Beta -Scope Allusers -AllowClobber -Force

# Import the Microsoft Graph Beta module
# Import-Module Microsoft.Graph.Beta


# Import-Module Microsoft.Graph.Identity.SignIns


# Connect to Microsoft Graph
Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All"

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

function Exclude-AppFromAllCAPoliciesUsingBeta {
    param (
        [string]$ExcludeAppId  # The ID of the app to be excluded
    )

    # Retrieve all Conditional Access Policies
    $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

    foreach ($policy in $allPolicies) {
        # Check if the app is already excluded
        $isAppExcluded = $policy.conditions.applications.excludeApplications -contains $ExcludeAppId
        if ($isAppExcluded) {
            Write-Output "App '$ExcludeAppId' is already excluded in Policy: $($policy.displayName)"
            continue
        }

        # Prepare the updated list of excluded apps
        $updatedExcludeApps = $policy.conditions.applications.excludeApplications + $ExcludeAppId

        # Construct the updated conditions object
        $updatedConditions = @{
            applications = @{
                excludeApplications = $updatedExcludeApps
            }
        }

        # Update the Conditional Access Policy to exclude the app
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $updatedConditions
            Write-Output "Updated Policy: $($policy.displayName) to exclude App '$ExcludeAppId'"
        } catch {
            Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
        }
    }
}

# Usage example
$excludeAppId = 'd4ebce55-015a-49b5-a083-c84d1797ae8c'  # The ID of the app to exclude Microsoft Intune Enrollment
$excludeAppId = '0000000a-0000-0000-c000-000000000000'  # The ID of the app to exclude Microsoft Intune
$excludeAppId = '00000003-0000-0ff1-ce00-000000000000'  # The ID of the app to exclude Office 365 SharePoint Online
Exclude-AppFromAllCAPoliciesUsingBeta -ExcludeAppId $excludeAppId



# Stop-Transcript