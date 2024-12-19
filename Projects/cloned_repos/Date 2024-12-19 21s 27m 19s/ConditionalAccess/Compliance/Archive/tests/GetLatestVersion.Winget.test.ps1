# $mode = $env:EnvironmentMode

#region FIRING UP MODULE STARTER
#################################################################################################
#                                                                                               #
#                                 FIRING UP MODULE STARTER                                      #
#                                                                                               #
#################################################################################################


# Invoke-Expression (Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1")

# Define a hashtable for splatting
# $moduleStarterParams = @{
#     Mode                   = 'dev'
#     SkipPSGalleryModules   = $true
#     SkipCheckandElevate    = $true
#     SkipPowerShell7Install = $true
#     SkipEnhancedModules    = $true
#     SkipGitRepos           = $true
#     # ExecutionMode          = 'Parallel'
# }

# Call the function using the splat
# Invoke-ModuleStarter @moduleStarterParams

#endregion FIRING UP MODULE STARTER



import-module 'C:\code\Modulesv2\EnhancedModuleStarterAO\EnhancedModuleStarterAO.psm1' -Force


# $params = @{
#     id                = "7zip.7zip"
#     # TargetVersion     = "24.8.0.0"
#     AcceptNewerVersion = $true
# }

# $result = Get-LatestWinGetVersion @params
# $result | Format-List




$params = @{
    id                = "Microsoft.PowerShell"
    AcceptNewerVersion = $true
}

$result = Get-LatestWinGetVersion @params
$result | Format-List



# # Example usage:
# $params = @{
#     id = "7zip.7zip"
#     TargetVersion = ""
#     AcceptNewerVersion = $true
# }
# $latestVersion = Get-LatestWinGetVersion @params
# if ($latestVersion) {
#     Write-Host "Latest version installed: $latestVersion"
# } else {
#     Write-Host "Latest version is not installed or not found."
# }