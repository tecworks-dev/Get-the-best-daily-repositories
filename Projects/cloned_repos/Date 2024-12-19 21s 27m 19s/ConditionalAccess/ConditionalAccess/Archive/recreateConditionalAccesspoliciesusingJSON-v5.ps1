# Connect to Microsoft Graph with necessary permissions
Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'

# Define a Conditional Access policy with required conditions
$policyDisplayName = "Sample Policy - Modern API - v2 - OFF"
$policyState = "disabled" # Set policy to be off (disabled)

# Define conditions for the policy, including 'users' and 'applications'
$conditions = @{
    Applications = @{
        IncludeApplications = @("All") # Include all applications
    }
    Users = @{
        IncludeUsers = @("All") # Include all users
    }
}

# Define grant controls for the policy
$grantControls = @{
    BuiltInControls = @("mfa") # Require MFA
    Operator = "OR"
}

# Construct the policy object
$policyObject = @{
    DisplayName = $policyDisplayName
    State = $policyState
    Conditions = $conditions
    GrantControls = $grantControls
}

# Convert policy object to JSON
$policyJson = $policyObject | ConvertTo-Json -Depth 10

try {
    # Attempt to create the Conditional Access policy in the "off" state
    $result = New-MgIdentityConditionalAccessPolicy -BodyParameter $policyJson
    Write-Host "Successfully created policy in the 'off' state: $($result.DisplayName)"
} catch {
    Write-Error "Failed to create policy: $_"
}

# Disconnect from Microsoft Graph
Disconnect-MgGraph
