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
    # ExecutionMode          = 'Parallel'
# }

# Call the function using the splat
# Invoke-ModuleStarter @moduleStarterParams

# Define replacements in a hashtable
# $replacements = @{
#     '\$Mode = "dev"'                     = '$Mode = "dev"'
#     '\$SkipPSGalleryModules = \$false'   = '$SkipPSGalleryModules = $true'
#     '\$SkipCheckandElevate = \$false'    = '$SkipCheckandElevate = $true'
#     '\$SkipAdminCheck = \$false'         = '$SkipAdminCheck = $true'
#     '\$SkipPowerShell7Install = \$false' = '$SkipPowerShell7Install = $true'
#     '\$SkipModuleDownload = \$false'     = '$SkipModuleDownload = $true'
#     '\$SkipGitrepos = \$false'           = '$SkipGitrepos = $true'
# }

# # Apply the replacements
# foreach ($pattern in $replacements.Keys) {
#     $scriptContent = $scriptContent -replace $pattern, $replacements[$pattern]
# }

# # Execute the script
# Invoke-Expression $scriptContent






# function Set-ScriptModeToDev {
#     # [CmdletBinding()]
#     # param (
#     #     [Parameter(Mandatory = $true)]
#     #     [string]$ScriptContent
#     # )

#     $scriptContent = Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Module-Starter.ps1"

#     # Ensure TLS 1.2 is used for all web requests
#     [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

#     # Define replacements in a hashtable
#     $replacements = @{
#         '\$Mode = "dev"'                     = '$Mode = "dev"'
#         '\$SkipPSGalleryModules = \$false'   = '$SkipPSGalleryModules = $true'
#         '\$SkipCheckandElevate = \$false'    = '$SkipCheckandElevate = $true'
#         '\$SkipAdminCheck = \$false'         = '$SkipAdminCheck = $true'
#         '\$SkipPowerShell7Install = \$false' = '$SkipPowerShell7Install = $true'
#         '\$SkipModuleDownload = \$false'     = '$SkipModuleDownload = $true'
#         '\$SkipGitrepos = \$false'           = '$SkipGitrepos = $true'
#     }

#     # Apply the replacements
#     foreach ($pattern in $replacements.Keys) {
#         $ScriptContent = $ScriptContent -replace $pattern, $replacements[$pattern]
#     }

#     # Execute the script
#     Invoke-Expression $ScriptContent
# }

# Example usage:
# Set-ScriptModeToDev


import-module 'C:\code\Modulesv2\EnhancedModuleStarterAO\EnhancedModuleStarterAO.psm1' -Force



#endregion FIRING UP MODULE STARTER


# . "C:\code\IntuneDeviceMigration\DeviceMigration\PSAppDeployToolkit\Toolkit\AppDeployToolkit\AppDeployToolkitMain.ps1"

# $validationParams = @{
#     SoftwareName        = "Git"
#     MinVersion          = [version]"2.46.0"
#     RegistryPath        = "HKLM:\SOFTWARE\GitForWindows"
#     ExePath             = "C:\Program Files\Git\bin\git.exe"
#     MaxRetries          = 3  # Single retry after installation
#     DelayBetweenRetries = 5
# }

# Validate-SoftwareInstallation @validationParams


# $params = @{
#     SoftwareName        = '7-Zip'
#     MinVersion          = [version] "19.0.0.0"
#     LatestVersion       = [version] "22.1.0.0"
#     RegistryPath        = 'HKLM:\SOFTWARE\7-Zip'
#     ExePath             = 'C:\Program Files\7-Zip\7z.exe'
#     MaxRetries          = 1
#     DelayBetweenRetries = 5
# }

# Validate-SoftwareInstallation @params




function Check-7ZipInstallation {
    param (
        [string]$CustomRegistryPath = "HKLM:\SOFTWARE\7-Zip",
        [string]$LogFilePath = "C:\Logs\7ZipCheck.log"
    )

    Begin {
        # Initialize log file
        $logMessage = "Checking 7-Zip installation..." | Out-String
        Write-EnhancedLog -Message $logMessage -Level "INFO" -FilePath $LogFilePath

        # Define registry paths to check
        $registryPaths = @(
            "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            $CustomRegistryPath
        )

        # Initialize result variable
        $sevenZipVersion = $null
    }

    Process {
        foreach ($registryPath in $registryPaths) {
            try {
                # Get all installed software in the current registry path
                $subkeys = Get-ChildItem -Path $registryPath -ErrorAction Stop

                foreach ($subkey in $subkeys) {
                    # Check if 7-Zip is installed
                    $displayName = (Get-ItemProperty -Path $subkey.PSPath -Name DisplayName -ErrorAction SilentlyContinue).DisplayName
                    if ($displayName -like "*7-Zip*") {
                        $sevenZipVersion = (Get-ItemProperty -Path $subkey.PSPath -Name DisplayVersion -ErrorAction SilentlyContinue).DisplayVersion
                        Write-EnhancedLog -Message "7-Zip found in '$registryPath' with version $sevenZipVersion." -Level "INFO" -FilePath $LogFilePath
                        break
                    }
                }

                if ($sevenZipVersion) {
                    break
                }
            } catch {
                Write-EnhancedLog -Message "Failed to access registry path: $registryPath. Error: $_" -Level "ERROR" -FilePath $LogFilePath
            }
        }
    }

    End {
        if ($sevenZipVersion) {
            Write-EnhancedLog -Message "7-Zip version $sevenZipVersion is installed." -Level "NOTICE" -FilePath $LogFilePath
            return $true
        } else {
            Write-EnhancedLog -Message "7-Zip is not installed." -Level "WARNING" -FilePath $LogFilePath
            return $false
        }
    }
}

# Example usage:
$checkResult = Check-7ZipInstallation -CustomRegistryPath "HKLM:\SOFTWARE\7-Zip" -LogFilePath "C:\Logs\7ZipCheck.log"

if ($checkResult -eq $false) {
    Write-Host "7-Zip is not installed on this machine." -ForegroundColor Red
    exit 1
} else {
    Write-Host "7-Zip is installed." -ForegroundColor Green
    exit 0
}




# $params = @{
#     SoftwareName        = 'notepad++'
#     MinVersion          = [version] "19.0.0.0"
#     LatestVersion       = [version] "22.1.0.0"
#     # RegistryPath        = 'HKLM:\SOFTWARE\7-Zip'
#     # ExePath             = 'C:\Program Files\7-Zip\7z.exe'
#     MaxRetries          = 3
#     DelayBetweenRetries = 5
# }

# Validate-SoftwareInstallation @params
