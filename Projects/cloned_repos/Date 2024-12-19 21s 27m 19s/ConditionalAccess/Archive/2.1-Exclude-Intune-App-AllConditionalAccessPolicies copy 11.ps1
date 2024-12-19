

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
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$ExcludeAppId  # The ID of the app to be excluded
    )

    # Retrieve all Conditional Access Policies using the custom function
    $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

    foreach ($policy in $allPolicies) {
        # Check if the app is already excluded
        if ($policy.conditions.applications.excludeApplications -contains $ExcludeAppId) {
            Write-Host "App '$ExcludeAppId' is already excluded in Policy: $($policy.displayName)"
            continue
        }

        # Add the app to the excludeApplications array
        $updatedExcludeApplications = $policy.conditions.applications.excludeApplications + $ExcludeAppId

        # Construct the BodyParameter to match the expected structure
        $bodyParameter = @{
            conditions = @{
                applications = @{
                    excludeApplications = $updatedExcludeApplications
                    # Include other necessary properties from the original policy structure if needed
                }
                # Replicate other conditions from the original policy if needed
            }
            # Include other top-level properties like grantControls from the original policy if needed
        }

        # Convert to JSON
        $bodyParameterJson = $bodyParameter | ConvertTo-Json -Depth 10

        # Update the Conditional Access Policy using the Beta endpoint
        try {
            Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -BodyParameter $bodyParameterJson
            Write-Host "Updated Policy: $($policy.displayName) to exclude App '$ExcludeAppId'"
        } catch {
            Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
        }
    }
}

# Example usage of the function
# $excludeAppId = 'd4ebce55-015a-49b5-a083-c84d1797ae8c'  # The ID of the app to exclude Microsoft Intune Enrollment
$excludeAppId = '0000000a-0000-0000-c000-000000000000'  # The ID of the app to exclude Microsoft Intune
Exclude-AppFromAllCAPoliciesUsingBeta -ExcludeAppId $excludeAppId


# Stop-Transcript