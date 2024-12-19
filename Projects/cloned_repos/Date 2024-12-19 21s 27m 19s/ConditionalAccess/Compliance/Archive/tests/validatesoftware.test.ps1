# # $mode = $env:EnvironmentMode

# #region FIRING UP MODULE STARTER
# #################################################################################################
# #                                                                                               #
# #                                 FIRING UP MODULE STARTER                                      #
# #                                                                                               #
# #################################################################################################


# # Invoke-Expression (Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1")

# # Define a hashtable for splatting
# # $moduleStarterParams = @{
# #     Mode                   = 'dev'
# #     SkipPSGalleryModules   = $true
# #     SkipCheckandElevate    = $true
# #     SkipPowerShell7Install = $true
# #     SkipEnhancedModules    = $true
# #     SkipGitRepos           = $true
#     # ExecutionMode          = 'Parallel'
# # }

# # Call the function using the splat
# # Invoke-ModuleStarter @moduleStarterParams

# # Define replacements in a hashtable
# # $replacements = @{
# #     '\$Mode = "dev"'                     = '$Mode = "dev"'
# #     '\$SkipPSGalleryModules = \$false'   = '$SkipPSGalleryModules = $true'
# #     '\$SkipCheckandElevate = \$false'    = '$SkipCheckandElevate = $true'
# #     '\$SkipAdminCheck = \$false'         = '$SkipAdminCheck = $true'
# #     '\$SkipPowerShell7Install = \$false' = '$SkipPowerShell7Install = $true'
# #     '\$SkipModuleDownload = \$false'     = '$SkipModuleDownload = $true'
# #     '\$SkipGitrepos = \$false'           = '$SkipGitrepos = $true'
# # }

# # # Apply the replacements
# # foreach ($pattern in $replacements.Keys) {
# #     $scriptContent = $scriptContent -replace $pattern, $replacements[$pattern]
# # }

# # # Execute the script
# # Invoke-Expression $scriptContent






# # function Set-ScriptModeToDev {
# #     # [CmdletBinding()]
# #     # param (
# #     #     [Parameter(Mandatory = $true)]
# #     #     [string]$ScriptContent
# #     # )

# #     $scriptContent = Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Module-Starter.ps1"

# #     # Ensure TLS 1.2 is used for all web requests
# #     [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# #     # Define replacements in a hashtable
# #     $replacements = @{
# #         '\$Mode = "dev"'                     = '$Mode = "dev"'
# #         '\$SkipPSGalleryModules = \$false'   = '$SkipPSGalleryModules = $true'
# #         '\$SkipCheckandElevate = \$false'    = '$SkipCheckandElevate = $true'
# #         '\$SkipAdminCheck = \$false'         = '$SkipAdminCheck = $true'
# #         '\$SkipPowerShell7Install = \$false' = '$SkipPowerShell7Install = $true'
# #         '\$SkipModuleDownload = \$false'     = '$SkipModuleDownload = $true'
# #         '\$SkipGitrepos = \$false'           = '$SkipGitrepos = $true'
# #     }

# #     # Apply the replacements
# #     foreach ($pattern in $replacements.Keys) {
# #         $ScriptContent = $ScriptContent -replace $pattern, $replacements[$pattern]
# #     }

# #     # Execute the script
# #     Invoke-Expression $ScriptContent
# # }

# # Example usage:
# # Set-ScriptModeToDev


# import-module 'C:\code\Modulesv2\EnhancedModuleStarterAO\EnhancedModuleStarterAO.psm1' -Force



# #endregion FIRING UP MODULE STARTER


# # . "C:\code\IntuneDeviceMigration\DeviceMigration\PSAppDeployToolkit\Toolkit\AppDeployToolkit\AppDeployToolkitMain.ps1"

# # $validationParams = @{
# #     SoftwareName        = "Git"
# #     MinVersion          = [version]"2.46.0"
# #     RegistryPath        = "HKLM:\SOFTWARE\GitForWindows"
# #     ExePath             = "C:\Program Files\Git\bin\git.exe"
# #     MaxRetries          = 3  # Single retry after installation
# #     DelayBetweenRetries = 5
# # }

# # Validate-SoftwareInstallation @validationParams


# # $params = @{
# #     SoftwareName        = '7-Zip'
# #     MinVersion          = [version] "19.0.0.0"
# #     LatestVersion       = [version] "22.1.0.0"
# #     RegistryPath        = 'HKLM:\SOFTWARE\7-Zip'
# #     ExePath             = 'C:\Program Files\7-Zip\7z.exe'
# #     MaxRetries          = 1
# #     DelayBetweenRetries = 5
# # }

# # Validate-SoftwareInstallation @params



# # Example usage:
# # $params = @{
# #     SoftwareName = "7-Zip"
# #     RegistryPath = "HKLM:\SOFTWARE\7-Zip"
# # }
# # $result = Test-SoftwareInstallation @params

# # if ($result.IsInstalled) {
# #     Write-Host "$($result.Message)" -ForegroundColor Green
# # } else {
# #     Write-Host "$($result.Message)" -ForegroundColor Red
# # }





# # Define the software name and optional paths
# # $softwareName = "7-Zip"
# # $optionalRegistryPath = "HKLM:\Software\7-Zip"  # Optional, if you have a specific registry path to check
# # $exePath = "C:\Program Files\7-Zip\7z.exe"      # Optional, if you want to validate via executable path

# # # Call the function to test if 7-Zip is installed
# # $result = Test-SoftwareInstallation -SoftwareName $softwareName -RegistryPath $optionalRegistryPath -ExePath $exePath

# # # Output the result
# # if ($result.IsInstalled) {
# #     Write-Host "$softwareName is installed." -ForegroundColor Green
# #     Write-Host "Version: $($result.Version)"
# #     Write-Host "Installation Path: $($result.InstallationPath)"
# #     Write-Host "Validated by: $($result.ValidationSource)"
# # } else {
# #     Write-Host "$softwareName is not installed." -ForegroundColor Red
# #     Write-Host "Error Message: $($result.ErrorMessage)"
# #     Write-Host "Exit Code: $($result.ExitCode)"
# # }






# # Custom registry path where 7-Zip might be installed
# # $customRegistryPath = "HKLM:\SOFTWARE\7-Zip"

# # # Executable path to validate 7-Zip if not found in the registry
# # $exePath = "C:\Program Files\7-Zip\7z.exe"

# # # Calling the function to test if 7-Zip is installed
# # $result = Test-SoftwareInstallation -SoftwareName "7-Zip" -RegistryPath $customRegistryPath -ExePath $exePath

# # # Handling the result
# # if ($result.IsInstalled) {
# #     Write-Host "7-Zip is installed. Version: $($result.Version)"
# # } else {
# #     Write-Host "7-Zip is not installed."
# # }




# # $params = @{
# #     SoftwareName        = 'notepad++'
# #     MinVersion          = [version] "19.0.0.0"
# #     LatestVersion       = [version] "22.1.0.0"
# #     # RegistryPath        = 'HKLM:\SOFTWARE\7-Zip'
# #     # ExePath             = 'C:\Program Files\7-Zip\7z.exe'
# #     MaxRetries          = 3
# #     DelayBetweenRetries = 5
# # }

# # Validate-SoftwareInstallation @params





# # $params = @{
# #     SoftwareName        = '7-Zip'
# #     MinVersion          = [version] "19.0.0.0"
# #     LatestVersion       = [version] "22.1.0.0"
# #     RegistryPath        = 'HKLM:\SOFTWARE\7-Zip'
# #     ExePath             = 'C:\Program Files\7-Zip\7z.exe'
# #     MaxRetries          = 1
# #     DelayBetweenRetries = 5
# # }

# # Validate-SoftwareInstallation @params






# # $params = @{
# #     SoftwareName  = "7-Zip"
# #     MinVersion    = [version]"19.00"
# #     LatestVersion = [version]"26.08.00.0"
# #     RegistryPath  = "HKLM:\SOFTWARE\7-Zip"
# #     ExePath       = "C:\Program Files\7-Zip\7z.exe"
# # }
# # $result = Validate-SoftwareInstallation @params

# # if ($result.IsInstalled) {
# #     Write-Host "7-Zip is installed. Version: $($result.Version)"
# #     Write-Host "Meets Minimum Requirement: $($result.MeetsMinRequirement)"
# #     Write-Host "Is Up To Date: $($result.IsUpToDate)"
# # } else {
# #     Write-Host "7-Zip is not installed."
# # }





# # # Define the software to check
# # $softwareList = @(
# #     @{
# #         SoftwareName  = "Microsoft Edge"
# #         MinVersion    = [version]"114.0.0.0"
# #         LatestVersion = [version]"116.0.1938.62"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Edge"
# #         ExePath       = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# #     },
# #     @{
# #         SoftwareName  = "Mozilla Firefox"
# #         MinVersion    = [version]"102.0.0.0"
# #         LatestVersion = [version]"118.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Mozilla\Mozilla Firefox"
# #         ExePath       = "C:\Program Files\Mozilla Firefox\firefox.exe"
# #     },
# #     @{
# #         SoftwareName  = "Google Chrome"
# #         MinVersion    = [version]"105.0.0.0"
# #         LatestVersion = [version]"117.0.5938.89"
# #         RegistryPath  = "HKLM:\SOFTWARE\Google\Chrome"
# #         ExePath       = "C:\Program Files\Google\Chrome\Application\chrome.exe"
# #     },
# #     @{
# #         SoftwareName  = "Git"
# #         MinVersion    = [version]"2.30.0"
# #         LatestVersion = [version]"2.46.2"
# #         RegistryPath  = "HKLM:\SOFTWARE\GitForWindows"
# #         ExePath       = "C:\Program Files\Git\bin\git.exe"
# #     },
# #     @{
# #         SoftwareName  = "Everything"
# #         MinVersion    = [version]"1.4.0.0"
# #         LatestVersion = [version]"1.5.0.1342"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Everything"
# #         ExePath       = "C:\Program Files\Everything\Everything.exe"
# #     },
# #     @{
# #         SoftwareName  = "PowerShell 7"
# #         MinVersion    = [version]"7.2.0"
# #         LatestVersion = [version]"7.4.5"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\PowerShell\7"
# #         ExePath       = "C:\Program Files\PowerShell\7\pwsh.exe"
# #     },
# #     @{
# #         SoftwareName  = "GitHub Desktop"
# #         MinVersion    = [version]"2.9.0"
# #         LatestVersion = [version]"3.3.4"
# #         RegistryPath  = "HKLM:\SOFTWARE\GitHub, Inc.\GitHub Desktop"
# #         ExePath       = "C:\Users\%USERNAME%\AppData\Local\GitHubDesktop\GitHubDesktop.exe"
# #     }
# # )

# # # Loop through each software and validate
# # foreach ($software in $softwareList) {
# #     $result = Validate-SoftwareInstallation @software

# #     if ($result.IsInstalled) {
# #         Write-Host "$($software.SoftwareName) is installed. Version: $($result.Version)"
# #         Write-Host "Meets Minimum Requirement: $($result.MeetsMinRequirement)"
# #         Write-Host "Is Up To Date: $($result.IsUpToDate)"
# #     } else {
# #         Write-Host "$($software.SoftwareName) is not installed."
# #     }
# # }






# # # Define the software to check
# # $softwareList = @(
# #     @{
# #         SoftwareName  = "7-Zip"
# #         MinVersion    = [version]"19.00"
# #         RegistryPath  = "HKLM:\SOFTWARE\7-Zip"
# #         ExePath       = "C:\Program Files\7-Zip\7z.exe"
# #         WinGetID      = "7zip.7zip"
# #     },
# #     @{
# #         SoftwareName  = "Microsoft Edge"
# #         MinVersion    = [version]"114.0.0.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Edge"
# #         ExePath       = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# #         WinGetID      = "Microsoft.Edge"
# #     },
# #     @{
# #         SoftwareName  = "Mozilla Firefox"
# #         MinVersion    = [version]"102.0.0.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Mozilla\Mozilla Firefox"
# #         ExePath       = "C:\Program Files\Mozilla Firefox\firefox.exe"
# #         WinGetID      = "Mozilla.Firefox"
# #     },
# #     @{
# #         SoftwareName  = "Google Chrome"
# #         MinVersion    = [version]"105.0.0.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Google\Chrome"
# #         ExePath       = "C:\Program Files\Google\Chrome\Application\chrome.exe"
# #         WinGetID      = "Google.Chrome"
# #     },
# #     @{
# #         SoftwareName  = "Git"
# #         MinVersion    = [version]"2.30.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\GitForWindows"
# #         ExePath       = "C:\Program Files\Git\bin\git.exe"
# #         WinGetID      = "Git.Git"
# #     },
# #     @{
# #         SoftwareName  = "Everything"
# #         MinVersion    = [version]"1.4.0.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Everything"
# #         ExePath       = "C:\Program Files\Everything\Everything.exe"
# #         WinGetID      = "voidtools.Everything"
# #     },
# #     @{
# #         SoftwareName  = "PowerShell 7"
# #         MinVersion    = [version]"7.2.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\PowerShell\7"
# #         ExePath       = "C:\Program Files\PowerShell\7\pwsh.exe"
# #         WinGetID      = "Microsoft.Powershell"
# #     },
# #     @{
# #         SoftwareName  = "GitHub Desktop"
# #         MinVersion    = [version]"2.9.0"
# #         RegistryPath  = "HKLM:\SOFTWARE\GitHub, Inc.\GitHub Desktop"
# #         ExePath       = "C:\Users\%USERNAME%\AppData\Local\GitHubDesktop\GitHubDesktop.exe"
# #         WinGetID      = "GitHub.GitHubDesktop"
# #     }
# # )

# # # Loop through each software and validate
# # foreach ($software in $softwareList) {
# #     # Get the latest version from WinGet
# #     $wingetParams = @{
# #         id                = $software.WinGetID
# #         AcceptNewerVersion = $true
# #     }
# #     $latestVersionResult = Get-LatestWinGetVersion @wingetParams
# #     $software.LatestVersion = [version]$latestVersionResult.LatestVersion

# #     # Prepare parameters for Validate-SoftwareInstallation
# #     $validateParams = @{
# #         SoftwareName  = $software.SoftwareName
# #         MinVersion    = $software.MinVersion
# #         LatestVersion = $software.LatestVersion
# #         RegistryPath  = $software.RegistryPath
# #         ExePath       = $software.ExePath
# #     }

# #     # Perform the validation
# #     $result = Validate-SoftwareInstallation @validateParams

# #     # Output the results
# #     if ($result.IsInstalled) {
# #         Write-Host "$($software.SoftwareName) is installed. Version: $($result.Version)"
# #         Write-Host "Meets Minimum Requirement: $($result.MeetsMinRequirement)"
# #         Write-Host "Is Up To Date: $($result.IsUpToDate)"
# #     } else {
# #         Write-Host "$($software.SoftwareName) is not installed."
# #     }
# # }









# # Define the software to check
# $softwareList = @(
#     @{
#         SoftwareName  = "7-Zip"
#         MinVersion    = [version]"19.00"
#         RegistryPath  = "HKLM:\SOFTWARE\7-Zip"
#         ExePath       = "C:\Program Files\7-Zip\7z.exe"
#         WinGetID      = "7zip.7zip"
#     },
#     @{
#         SoftwareName  = "Microsoft Edge"
#         MinVersion    = [version]"114.0.0.0"
#         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Edge"
#         ExePath       = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
#         WinGetID      = "Microsoft.Edge"
#     },
#     @{
#         SoftwareName  = "Mozilla Firefox"
#         MinVersion    = [version]"102.0.0.0"
#         RegistryPath  = "HKLM:\SOFTWARE\Mozilla\Mozilla Firefox"
#         ExePath       = "C:\Program Files\Mozilla Firefox\firefox.exe"
#         WinGetID      = "Mozilla.Firefox"
#     },
#     @{
#         SoftwareName  = "Google Chrome"
#         MinVersion    = [version]"105.0.0.0"
#         RegistryPath  = "HKLM:\SOFTWARE\Google\Chrome"
#         ExePath       = "C:\Program Files\Google\Chrome\Application\chrome.exe"
#         WinGetID      = "Google.Chrome"
#     },
#     @{
#         SoftwareName  = "Git"
#         MinVersion    = [version]"2.30.0"
#         RegistryPath  = "HKLM:\SOFTWARE\GitForWindows"
#         ExePath       = "C:\Program Files\Git\bin\git.exe"
#         WinGetID      = "Git.Git"
#     },
#     @{
#         SoftwareName  = "Everything"
#         MinVersion    = [version]"1.4.0.0"
#         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Everything"
#         ExePath       = "C:\Program Files\Everything\Everything.exe"
#         WinGetID      = "voidtools.Everything"
#     },
#     @{
#         SoftwareName  = "PowerShell 7"
#         MinVersion    = [version]"7.2.0"
#         RegistryPath  = "HKLM:\SOFTWARE\Microsoft\PowerShell\7"
#         ExePath       = "C:\Program Files\PowerShell\7\pwsh.exe"
#         WinGetID      = "Microsoft.Powershell"
#     },
#     @{
#         SoftwareName  = "GitHub Desktop"
#         MinVersion    = [version]"2.9.0"
#         RegistryPath  = "HKLM:\SOFTWARE\GitHub, Inc.\GitHub Desktop"
#         ExePath       = "C:\Users\%USERNAME%\AppData\Local\GitHubDesktop\GitHubDesktop.exe"
#         WinGetID      = "GitHub.GitHubDesktop"
#     }
# )

# # Initialize counters
# $totalChecks = 0
# $totalSuccesses = 0
# $totalFailures = 0

# $summaryReport = @()

# # Loop through each software and validate
# foreach ($software in $softwareList) {
#     $totalChecks++

#     # Get the latest version from WinGet
#     $wingetParams = @{
#         id                = $software.WinGetID
#         AcceptNewerVersion = $true
#     }
#     $latestVersionResult = Get-LatestWinGetVersion @wingetParams
#     $software.LatestVersion = [version]$latestVersionResult.LatestVersion

#     # Prepare parameters for Validate-SoftwareInstallation
#     $validateParams = @{
#         SoftwareName  = $software.SoftwareName
#         MinVersion    = $software.MinVersion
#         LatestVersion = $software.LatestVersion
#         RegistryPath  = $software.RegistryPath
#         ExePath       = $software.ExePath
#     }

#     # Perform the validation
#     $result = Validate-SoftwareInstallation @validateParams

#     # Process and store the result
#     if ($result.IsInstalled) {
#         $totalSuccesses++
#         $status = "Success"
#         $color = "Green"
#         Write-Host "$($software.SoftwareName) is installed. Version: $($result.Version)" -ForegroundColor $color
#         Write-Host "Meets Minimum Requirement: $($result.MeetsMinRequirement)"
#         Write-Host "Is Up To Date: $($result.IsUpToDate)"
#     } else {
#         $totalFailures++
#         $status = "Failure"
#         $color = "Red"
#         Write-Host "$($software.SoftwareName) is not installed." -ForegroundColor $color
#     }

#     # Add to summary report
#     $summaryReport += [PSCustomObject]@{
#         Software    = $software.SoftwareName
#         Status      = $status
#         Version     = $result.Version
#         IsInstalled = $result.IsInstalled
#         MeetsMin    = $result.MeetsMinRequirement
#         UpToDate    = $result.IsUpToDate
#     }
# }

# # Final Summary Report
# Write-Host "`nSummary Report" -ForegroundColor "Cyan"
# Write-Host "---------------------------"
# Write-Host "Total Checks: $totalChecks"
# Write-Host "Successes: $totalSuccesses" -ForegroundColor "Green"
# Write-Host "Failures: $totalFailures" -ForegroundColor "Red"
# Write-Host "---------------------------`n"

# foreach ($entry in $summaryReport) {
#     $color = if ($entry.Status -eq "Success") { "Green" } else { "Red" }
#     Write-Host "$($entry.Software): $($entry.Status)" -ForegroundColor $color
#     Write-Host "Installed Version: $($entry.Version)"
#     Write-Host "Meets Min Requirement: $($entry.MeetsMin)"
#     Write-Host "Is Up To Date: $($entry.UpToDate)"
#     Write-Host "--------------------------------------"
# }

# # Final overall status message
# if ($totalFailures -eq 0) {
#     Write-Host "`nAll software installations validated successfully!" -ForegroundColor "Green"
# } else {
#     Write-Host "`nSome software installations failed validation. Please review the summary above." -ForegroundColor "Red"
# }









# Define the software to check
$softwareList = @(
    @{
        SoftwareName = "7-Zip"
        MinVersion   = [version]"19.00"
        RegistryPath = "HKLM:\SOFTWARE\7-Zip"
        ExePath      = "C:\Program Files\7-Zip\7z.exe"
        WinGetID     = "7zip.7zip"
    },
    @{
        SoftwareName = "Microsoft Edge"
        MinVersion   = [version]"114.0.0.0"
        RegistryPath = "HKLM:\SOFTWARE\Microsoft\Edge"
        ExePath      = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        WinGetID     = "Microsoft.Edge"
    },
    @{
        SoftwareName = "Mozilla Firefox"
        MinVersion   = [version]"102.0.0.0"
        RegistryPath = "HKLM:\SOFTWARE\Mozilla\Mozilla Firefox"
        ExePath      = "C:\Program Files\Mozilla Firefox\firefox.exe"
        WinGetID     = "Mozilla.Firefox"
    },
    @{
        SoftwareName = "Google Chrome"
        MinVersion   = [version]"105.0.0.0"
        RegistryPath = "HKLM:\SOFTWARE\Google\Chrome"
        ExePath      = "C:\Program Files\Google\Chrome\Application\chrome.exe"
        WinGetID     = "Google.Chrome"
    },
    @{
        SoftwareName = "Git"
        MinVersion   = [version]"2.30.0"
        RegistryPath = "HKLM:\SOFTWARE\GitForWindows"
        ExePath      = "C:\Program Files\Git\bin\git.exe"
        WinGetID     = "Git.Git"
    },
    @{
        SoftwareName = "Everything"
        MinVersion   = [version]"1.4.0.0"
        RegistryPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Everything"
        ExePath      = "C:\Program Files\Everything\Everything.exe"
        WinGetID     = "voidtools.Everything"
    },
    @{
        SoftwareName = "PowerShell 7"
        MinVersion   = [version]"7.2.0"
        RegistryPath = "HKLM:\SOFTWARE\Microsoft\PowerShell\7"
        ExePath      = "C:\Program Files\PowerShell\7\pwsh.exe"
        WinGetID     = "Microsoft.PowerShell"
    },
    @{
        SoftwareName = "GitHub Desktop"
        MinVersion   = [version]"2.9.0"
        RegistryPath = "HKLM:\SOFTWARE\GitHub, Inc.\GitHub Desktop"
        ExePath      = "C:\Users\%USERNAME%\AppData\Local\GitHubDesktop\GitHubDesktop.exe"
        WinGetID     = "GitHub.GitHubDesktop"
    }
)

# Initialize counters
$totalChecks = 0
$totalSuccesses = 0
$totalFailures = 0

$summaryReport = @()

# Loop through each software and validate
foreach ($software in $softwareList) {
    $totalChecks++

    # Get the latest version from WinGet
    $wingetParams = @{
        id                 = $software.WinGetID
        AcceptNewerVersion = $true
    }
    $latestVersionResult = Get-LatestWinGetVersion @wingetParams
    $software.LatestVersion = [version]$latestVersionResult.LatestVersion

    # Prepare parameters for Validate-SoftwareInstallation
    $validateParams = @{
        SoftwareName  = $software.SoftwareName
        MinVersion    = $software.MinVersion
        LatestVersion = $software.LatestVersion
        RegistryPath  = $software.RegistryPath
        ExePath       = $software.ExePath
    }

    # Perform the validation
    $result = Validate-SoftwareInstallation @validateParams

    # Process and store the result
    if ($result.IsInstalled) {
        $totalSuccesses++
        $status = "Success"
        $color = "Green"
        Write-Host "$($software.SoftwareName) is installed. Version: $($result.Version)" -ForegroundColor $color
        Write-Host "Meets Minimum Requirement: $($result.MeetsMinRequirement)"
        Write-Host "Is Up To Date: $($result.IsUpToDate)"
    }
    else {
        $totalFailures++
        $status = "Failure"
        $color = "Red"
        Write-Host "$($software.SoftwareName) is not installed." -ForegroundColor $color
    }

    # Add to summary report
    $summaryReport += [PSCustomObject]@{
        Software         = $software.SoftwareName
        Status           = $status
        InstalledVersion = $result.Version
        LatestVersion    = $software.LatestVersion
        IsInstalled      = $result.IsInstalled
        MeetsMin         = $result.MeetsMinRequirement
        UpToDate         = $result.IsUpToDate
    }
}

# Final Summary Report
Write-Host "`nSummary Report" -ForegroundColor "Cyan"
Write-Host "---------------------------"
Write-Host "Total Checks: $totalChecks"
Write-Host "Successes: $totalSuccesses" -ForegroundColor "Green"
Write-Host "Failures: $totalFailures" -ForegroundColor "Red"
Write-Host "---------------------------`n"

foreach ($entry in $summaryReport) {
    $color = if ($entry.Status -eq "Success") { "Green" } else { "Red" }
    Write-Host "$($entry.Software): $($entry.Status)" -ForegroundColor $color
    Write-Host "Installed Version: $($entry.InstalledVersion)"
    Write-Host "Winget Latest Version: $($entry.LatestVersion)"
    Write-Host "Meets Min Requirement: $($entry.MeetsMin)"
    Write-Host "Is Up To Date: $($entry.UpToDate)"
    Write-Host "--------------------------------------"
}

# Final overall status message
if ($totalFailures -eq 0) {
    Write-Host "`nAll software installations validated successfully!" -ForegroundColor "Green"
}
else {
    Write-Host "`nSome software installations failed validation. Please review the summary above." -ForegroundColor "Red"
}

