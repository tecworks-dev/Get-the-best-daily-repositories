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
        State = $jsonContent.state
        Conditions = $jsonContent.conditions
        GrantControls = $jsonContent.grantControls
        # Include other fields as needed, ensuring they match the expected schema
    }

    # Remove any null or deprecated properties to match the expected schema
    $policyParams.GetEnumerator() | ForEach-Object {
        if ($null -eq $policyParams[$_.Key]) {
            $policyParams.Remove($_.Key)
        }
    }

    # Convert the policy parameters to a JSON object
    $policyJson = $policyParams | ConvertTo-Json -Depth 10

    try {
        # Create the new Conditional Access policy
        $result = New-MgIdentityConditionalAccessPolicy -BodyParameter $policyJson
        Write-Host "Successfully created policy: $($result.DisplayName)"
    } catch {
        Write-Error "Failed to create policy from $($jsonFile.Name): $_"
    }
}

# Disconnect from Microsoft Graph
# Disconnect-MgGraph
