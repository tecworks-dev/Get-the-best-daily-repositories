# Connect to Microsoft Graph with necessary permissions
Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'

# Define a simple Conditional Access policy
$policyDisplayName = "Sample Policy - Modern API"
$policyState = "enabled"

# Define conditions for the policy (adjust as needed)
$conditions = @{
    Applications = @{
        IncludeApplications = "All"
    }
    Users = @{
        IncludeUsers = "All"
    }
}

# Define grant controls for the policy (adjust as needed)
$grantControls = @{
    BuiltInControls = @("mfa")
    Operator = "OR"
}

# Create the policy
try {
    $policyParams = @{
        DisplayName = $policyDisplayName
        State = $policyState
        Conditions = $conditions
        GrantControls = $grantControls
    }

    $policyJson = $policyParams | ConvertTo-Json -Depth 10

    # Create the Conditional Access policy
    $result = New-MgIdentityConditionalAccessPolicy -BodyParameter $policyJson
    Write-Host "Successfully created policy: $($result.DisplayName)"
} catch {
    Write-Error "Failed to create policy: $_"
}

# Disconnect from Microsoft Graph
# Disconnect-MgGraph
