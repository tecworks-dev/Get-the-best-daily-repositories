# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\Main"
$jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\PMM\GRANT"
$jsonFiles = Get-ChildItem -Path $jsonFilesDirectory -Filter "*.json"

foreach ($file in $jsonFiles) {
    try {
        # Read the file content as an array of lines and then join them into a single string
        $jsonContentLines = Get-Content -Path $file.FullName -Encoding UTF8
        $jsonContent = $jsonContentLines -join ""
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
