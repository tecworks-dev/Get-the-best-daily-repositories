# Define the path to your JSON file
$jsonFilePath = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\Main\zzz-CA004 - GRANT - ADMINS - MFA - Global Admins with default Auth Strengths.json"

# Read the content of the JSON file
$jsonContent = Get-Content -Path $jsonFilePath | Out-String

# Convert the JSON content to a PowerShell object
$jsonObject = $jsonContent | ConvertFrom-Json

# Extract the file name without the extension
$fileNameWithoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($jsonFilePath)

# Update the displayName property
$jsonObject.displayName = $fileNameWithoutExtension

# Convert the modified object back to JSON
$updatedJsonContent = $jsonObject | ConvertTo-Json -Depth 100

# Save the updated JSON back to the file, overwriting the original content
$updatedJsonContent | Set-Content -Path $jsonFilePath
