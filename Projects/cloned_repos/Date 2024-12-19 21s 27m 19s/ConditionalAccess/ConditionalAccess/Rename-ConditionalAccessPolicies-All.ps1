# Define the directory containing the JSON files
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\Main"
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\PMM\GRANT"
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\SettingsCatalog"
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\DeviceConfiguration"
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\UpdatePolicies"
# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\AdministrativeTemplates"

# Get all JSON files in the directory
$jsonFiles = Get-ChildItem -Path $jsonFilesDirectory -Filter "*.json"

# Iterate over each JSON file
foreach ($file in $jsonFiles) {
    # Read the content of the JSON file
    $jsonContent = Get-Content -Path $file.FullName | Out-String

    # Convert the JSON content to a PowerShell object
    $jsonObject = $jsonContent | ConvertFrom-Json

    # Extract the file name without the extension
    $fileNameWithoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)

    # Update the displayName property
    # $jsonObject.name = $fileNameWithoutExtension
    $jsonObject.displayName = $fileNameWithoutExtension

    # Convert the modified object back to JSON
    $updatedJsonContent = $jsonObject | ConvertTo-Json -Depth 100

    # Save the updated JSON back to the file, overwriting the original content
    $updatedJsonContent | Set-Content -Path $file.FullName
}