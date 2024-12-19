# $jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\Main"
$jsonFilesDirectory = "C:\Code\CaaC\Nova\Feb022024\AOllivierre-Sandbox\Entra-Intune-v2\MSFT\ConditionalAccess\PMM\GRANT"
$jsonFiles = Get-ChildItem -Path $jsonFilesDirectory -Filter "*.json"

foreach ($file in $jsonFiles) {
    if (Test-Path $file.FullName) {
        $jsonContent = Get-Content -Path $file.FullName | Out-String
        $jsonObject = $jsonContent | ConvertFrom-Json

        if ($null -ne $jsonObject.PSObject.Properties.Match('displayName')) {
            $fileNameWithoutExtension = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
            $jsonObject.displayName = $fileNameWithoutExtension
            $updatedJsonContent = $jsonObject | ConvertTo-Json -Depth 100
            $updatedJsonContent | Set-Content -Path $file.FullName
        } else {
            Write-Host "The file $($file.Name) does not contain a 'displayName' property."
        }
    } else {
        Write-Host "Unable to find or access the file: $($file.FullName)"
    }
}
