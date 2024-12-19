

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
        # Initialize a flag to track update status
        $updateRequired = $false

        # Ensure the excludeApplications property exists to avoid null reference exceptions
        if (-not $policy.conditions.applications.excludeApplications) {
            $policy.conditions.applications.excludeApplications = @()
            $updateRequired = $true
        }

        # Check if the app is already excluded
        if ($policy.conditions.applications.excludeApplications -contains $ExcludeAppId) {
            Write-Host "App '$ExcludeAppId' is already excluded in Policy: $($policy.displayName)"
        } else {
            # Mark the policy for update
            $updateRequired = $true
        }

        if ($updateRequired) {
            # Add the app to the excludeApplications array if not already included
            $policy.conditions.applications.excludeApplications += $ExcludeAppId

            # Prepare the applications parameter
            $applications = @{
                excludeApplications = $policy.conditions.applications.excludeApplications
            }

            # Update the Conditional Access Policy using the Beta endpoint
            try {
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions @{ applications = $applications } -ErrorAction Stop
                # Update-MgIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions @{ applications = $applications } -ErrorAction Stop
                Write-Host "Updated Policy: $($policy.displayName) to exclude App '$ExcludeAppId'"
            } catch {
                Write-Error "Failed to update Policy: $($policy.displayName). Error: $_"
            }
        }
    }
}

# Example Usage
$excludeAppId = 'd4ebce55-015a-49b5-a083-c84d1797ae8c' # The ID of the app to exclude
Exclude-AppFromAllCAPoliciesUsingBeta -ExcludeAppId $excludeAppId



Stop-Transcript