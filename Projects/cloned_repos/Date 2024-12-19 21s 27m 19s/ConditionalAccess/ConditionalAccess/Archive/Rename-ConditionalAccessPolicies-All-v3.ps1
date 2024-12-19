# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\Main"
$jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\PMM\GRANT"
$jsonFiles = Get-ChildItem -Path $jsonFilesDirectory -Filter "*.json"

foreach ($file in $jsonFiles) {
    try {
        # Ensure the JSON content is read with the correct encoding, UTF8 is commonly used for JSON files
        $jsonContent = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        $jsonObject = $jsonContent | ConvertFrom-Json

        $fileNameWithoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)

        # Update the displayName property
        $jsonObject.displayName = $fileNameWithoutExtension

        $updatedJsonContent = $jsonObject | ConvertTo-Json -Depth 100 -Compress
        $updatedJsonContent | Set-Content -Path $file.FullName -Force -Encoding UTF8
    } catch {
        Write-Host "An error occurred processing $($file.Name): $_"
    }
}
