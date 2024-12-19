# Connect to Microsoft Graph with the necessary permissions
Connect-MgGraph -Scopes 'Policy.Read.All', 'Policy.ReadWrite.ConditionalAccess'

# Confirm before proceeding as this will delete Conditional Access policies specified in the CSV
$confirmation = Read-Host "Are you sure you want to delete Conditional Access policies listed in the CSV? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    Disconnect-MgGraph
    exit
}

# Import the CSV file containing the policy IDs
# $csvPath = Read-Host "D:\Code\CB\Entra\CCI\Graph\export\DeprecatedPolicies-v6-sandbox.csv"
$csvPath = "D:\Code\CB\Entra\Ladco\Graph\export\DeprecatedPolicies-v9-cci.csv"
$policyIds = Import-Csv -Path $csvPath

# Iterate through the CSV data and delete each policy by ID
foreach ($policy in $policyIds) {
    $policyId = $policy.Id # Assuming the column name in the CSV is 'PolicyId'
    try {
        # Delete the Conditional Access policy by its ID
        Remove-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId
        Write-Output "Successfully requested deletion of Conditional Access policy with ID: $policyId"
    } catch {
        Write-Error "Failed to delete Conditional Access policy with ID: $policyId. Error: $_"
    }
}

# Disconnect from Microsoft Graph
# Disconnect-MgGraph
