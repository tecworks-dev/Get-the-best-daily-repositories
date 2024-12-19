# Connect to Microsoft Graph with necessary permissions
Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'

# Specify the directory containing the JSON files
# $jsonDir = "C:\code\caac\Feb172024\CCI\Entra-Intune-v1\Canada Computing Inc\ConditionalAccess-Recreated-Modern-Graph-API"
$jsonDir = "C:\code\caac\Feb192024\CCI\Canada Computing Inc\ConditionalAccess-Recreated-Modern-Graph-API"

# Get all JSON files from the directory
$jsonFiles = Get-ChildItem -Path $jsonDir -Filter "*.json"

foreach ($jsonFile in $jsonFiles) {
    # Load the JSON content
    $jsonContent = Get-Content -Path $jsonFile.FullName -Raw | ConvertFrom-Json

    # Prepare the policy object for creation
    $policyParams = @{
        DisplayName = $jsonContent.displayName + " - Modern API"
        State = "disabled" # Ensure the policy is created in the "off" state
        Conditions = @{
            Users = @{
                IncludeUsers = @("All") # Specify user object IDs to include
                # ExcludeUsers = @("userObjectId3") # Specify user object IDs to exclude
            }
            Applications = @{
                # IncludeApplications = @("appId1", "appId2") # Specify application IDs to include
                IncludeApplications = @("All") # Specify application IDs to include
            }
        }
        GrantControls = @{
            BuiltInControls = @("mfa") # Require MFA
            Operator = "OR"
        }
    }

    # Convert policy object to JSON
    $policyJson = $policyParams | ConvertTo-Json -Depth 10

    try {
        # Attempt to create the Conditional Access policy
        $result = New-MgIdentityConditionalAccessPolicy -BodyParameter $policyJson
        Write-Host "Successfully created policy: $($result.DisplayName)"
    } catch {
        Write-Error "Failed to create policy from file $($jsonFile.Name): $_"
    }
}

# Disconnect from Microsoft Graph
# Disconnect-MgGraph
