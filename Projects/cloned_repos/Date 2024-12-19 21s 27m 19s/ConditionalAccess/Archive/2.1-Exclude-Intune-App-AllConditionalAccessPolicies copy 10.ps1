

Start-Transcript -Path "C:\Code\CB\Entra\Sandbox\Graph\Exclude_AppFromAllCAPoliciesUsingBeta-v1.log"
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
        # Initialize a flag to check if app is already excluded
        $isAppAlreadyExcluded = $false

        # Check if the policy already excludes the app
        if ($policy.Conditions.Applications.ExcludeApplications -contains $ExcludeAppId) {
            Write-Host "App '$ExcludeAppId' is already excluded in Policy: $($policy.DisplayName)"
            $isAppAlreadyExcluded = $true
        }

        if (-not $isAppAlreadyExcluded) {
            # Prepare the policy update with the app excluded
            $bodyParameter = @{
                Conditions = @{
                    Applications = @{
                        ExcludeApplications = $policy.Conditions.Applications.ExcludeApplications + $ExcludeAppId
                    }
                }
            }

            # Convert to JSON as BodyParameter expects a JSON string
            $bodyParameterJson = $bodyParameter | ConvertTo-Json -Depth 10

            try {
                # Update the Conditional Access Policy to exclude the app
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.Id -BodyParameter $bodyParameterJson
                Write-Host "Updated Policy: $($policy.DisplayName) to exclude App '$ExcludeAppId'"
            } catch {
                Write-Error "Failed to update Policy: $($policy.DisplayName). Error: $_"
            }
        }
    }
}

# Example usage of the function
# $excludeAppId = 'd4ebce55-015a-49b5-a083-c84d1797ae8c'  # The ID of the app to exclude Microsoft Intune Enrollment
$excludeAppId = '0000000a-0000-0000-c000-000000000000'  # The ID of the app to exclude Microsoft Intune
Exclude-AppFromAllCAPoliciesUsingBeta -ExcludeAppId $excludeAppId




Stop-Transcript