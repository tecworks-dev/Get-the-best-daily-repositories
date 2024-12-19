# Connect to Microsoft Graph with necessary permissions
Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'

# Specify the directory containing the JSON files
$jsonDir = "C:\code\caac\Feb172024\CCI\Entra-Intune-v1\Canada Computing Inc\ConditionalAccess-Recreated-Modern-Graph-API"

# Get all JSON files from the directory
$jsonFiles = Get-ChildItem -Path $jsonDir -Filter "*.json"

foreach ($jsonFile in $jsonFiles) {
    # Load the JSON content
    $jsonContent = Get-Content -Path $jsonFile.FullName -Raw | ConvertFrom-Json

    # Prepare the policy object for creation
    $policyParams = @{
        DisplayName = $jsonContent.displayName + " - Modern API"
        State = "disabled" # Ensure the policy is created in the "off" state
        # Define other necessary properties for the policy here
        # For example, Conditions and GrantControls as per your requirements
        # This example uses placeholder values
        Conditions = @{
            Applications = @{
                IncludeApplications = @("All")
            }
            Users = @{
                IncludeUsers = @("All")
            }
        }
        GrantControls = @{
            BuiltInControls = @("block")
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
