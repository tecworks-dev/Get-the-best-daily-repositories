# Connect to Microsoft Graph with necessary permissions
Connect-MgGraph -Scopes 'Policy.ReadWrite.ConditionalAccess'

# Specify the directory containing the JSON files
$jsonDir = "C:\code\caac\Feb172024\CCI\Entra-Intune-v1\Canada Computing Inc\ConditionalAccess-Recreated-Modern-Graph-API"

# Get all JSON files from the directory
$jsonFiles = Get-ChildItem -Path $jsonDir -Filter "*.json"

foreach ($jsonFile in $jsonFiles) {
    # Load the JSON content
    $jsonContent = Get-Content -Path $jsonFile.FullName -Raw | ConvertFrom-Json

    # Construct a new policy object from the JSON content, ensuring to avoid deprecated attributes
    $newPolicy = @{
        DisplayName = $jsonContent.displayName + " - Modern API"
        State = $jsonContent.state
        Conditions = @{
            Applications = $jsonContent.conditions.applications
            Users = $jsonContent.conditions.users
            Platforms = $jsonContent.conditions.platforms
            Locations = $jsonContent.conditions.locations
            ClientAppTypes = $jsonContent.conditions.clientAppTypes
        }
        GrantControls = $jsonContent.grantControls
        SessionControls = $jsonContent.sessionControls
    }

    # Convert the new policy object to JSON
    $newPolicyJson = $newPolicy | ConvertTo-Json -Depth 10

    try {
        # Create the new Conditional Access policy using the Modern API
        $result = New-MgIdentityConditionalAccessPolicy -BodyParameter $newPolicyJson
        Write-Host "Successfully created policy: $($result.DisplayName)"
    } catch {
        Write-Error "Failed to create policy from $($jsonFile.Name): $_"
    }
}

# Disconnect from Microsoft Graph
Disconnect-MgGraph
