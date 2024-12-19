# Define the path to the CSV file
$csvPath = "D:\Code\CB\Entra\CCI\Graph\export\DeprecatedPolicies-v9-cci.csv"

# Define the source and destination directories
$sourceDir = "C:\code\caac\Feb192024\CCI\Canada Computing Inc\ConditionalAccess"
$destinationDir = "C:\code\caac\Feb192024\CCI\Canada Computing Inc\ConditionalAccess-Recreated-Modern-Graph-API"

# Import the CSV file
$csvData = Import-Csv -Path $csvPath

# Iterate over each row in the CSV
foreach ($row in $csvData) {
    # Extract the display name
    $displayName = $row.'displayname'

    # Define the source file path based on the display name
    $sourceFile = Join-Path -Path $sourceDir -ChildPath "$displayName.json"

    # Define the destination file path
    $destinationFile = Join-Path -Path $destinationDir -ChildPath "$displayName.json"

    # Check if the source file exists
    if (Test-Path -Path $sourceFile) {
        # Move the JSON file from the source to the destination directory
        Move-Item -Path $sourceFile -Destination $destinationFile
        Write-Host "Moved: $sourceFile to $destinationFile"
    } else {
        Write-Host "File not found: $sourceFile"
    }
}
