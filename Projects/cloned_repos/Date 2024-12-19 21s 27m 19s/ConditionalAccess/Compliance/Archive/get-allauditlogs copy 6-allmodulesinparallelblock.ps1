#############################################################################################################
#
#   Tool:           Intune Win32 Deployer
#   Author:         Abdullah Ollivierre
#   Website:        https://github.com/aollivierre
#   Twitter:        https://x.com/ollivierre
#   LinkedIn:       https://www.linkedin.com/in/aollivierre
#
#   Description:    https://github.com/aollivierre
#
#############################################################################################################

<#
    .SYNOPSIS
    Packages any custom app for MEM (Intune) deployment.
    Uploads the packaged into the target Intune tenant.

    .NOTES
    For details on IntuneWin32App go here: https://github.com/aollivierre

#>

#################################################################################################################################
################################################# START VARIABLES ###############################################################
#################################################################################################################################



#First, load secrets and create a credential object:
# Assuming secrets.json is in the same directory as your script
$secretsPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"

# Load the secrets from the JSON file
$secrets = Get-Content -Path $secretsPath -Raw | ConvertFrom-Json

# Read configuration from the JSON file
# Assign values from JSON to variables

# Read configuration from the JSON file
$configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
$env:MYMODULE_CONFIG_PATH = $configPath

$config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

#  Variables from JSON file
$tenantId = $secrets.tenantId
$clientId = $secrets.clientId

$certPath = Join-Path -Path $PSScriptRoot -ChildPath 'graphcert.pfx'
$CertPassword = $secrets.CertPassword


function Initialize-Environment {
    param (
        [string]$WindowsModulePath = "EnhancedBoilerPlateAO\2.0.0\EnhancedBoilerPlateAO.psm1",
        [string]$LinuxModulePath = "/usr/src/code/Modules/EnhancedBoilerPlateAO/2.0.0/EnhancedBoilerPlateAO.psm1"
    )

    function Get-Platform {
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            return $PSVersionTable.Platform
        }
        else {
            return [System.Environment]::OSVersion.Platform
        }
    }

    function Setup-GlobalPaths {
        if ($env:DOCKER_ENV -eq $true) {
            $global:scriptBasePath = $env:SCRIPT_BASE_PATH
            $global:modulesBasePath = $env:MODULES_BASE_PATH
        }
        else {
            # $global:scriptBasePath = $PSScriptRoot
            # $global:modulesBasePath = "$PSScriptRoot\modules"

            $global:scriptBasePath = $PSScriptRoot
            $global:modulesBasePath = "c:\code\modules"
        }
    }

    function Setup-WindowsEnvironment {
        # Get the base paths from the global variables
        Setup-GlobalPaths

        # Construct the paths dynamically using the base paths
        $global:modulePath = Join-Path -Path $modulesBasePath -ChildPath $WindowsModulePath
        $global:AOscriptDirectory = Join-Path -Path $scriptBasePath -ChildPath "Win32Apps-DropBox"
        $global:directoryPath = Join-Path -Path $scriptBasePath -ChildPath "Win32Apps-DropBox"
        $global:Repo_Path = $scriptBasePath
        $global:Repo_winget = "$Repo_Path\Win32Apps-DropBox"


        # Import the module using the dynamically constructed path
        Import-Module -Name $global:modulePath -Verbose -Force:$true -Global:$true

        # Log the paths to verify
        Write-Output "Module Path: $global:modulePath"
        Write-Output "Repo Path: $global:Repo_Path"
        Write-Output "Repo Winget Path: $global:Repo_winget"
    }

    function Setup-LinuxEnvironment {
        # Get the base paths from the global variables
        Setup-GlobalPaths

        # Import the module using the Linux path
        Import-Module $LinuxModulePath -Verbose

        # Convert paths from Windows to Linux format
        $global:AOscriptDirectory = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer"
        $global:directoryPath = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer\Win32Apps-DropBox"
        $global:Repo_Path = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer"
        $global:Repo_winget = "$global:Repo_Path\Win32Apps-DropBox"
    }

    $platform = Get-Platform
    if ($platform -eq 'Win32NT' -or $platform -eq [System.PlatformID]::Win32NT) {
        Setup-WindowsEnvironment
    }
    elseif ($platform -eq 'Unix' -or $platform -eq [System.PlatformID]::Unix) {
        Setup-LinuxEnvironment
    }
    else {
        throw "Unsupported operating system"
    }
}

# Call the function to initialize the environment
Initialize-Environment


# Example usage of global variables outside the function
Write-Output "Global variables set by Initialize-Environment:"
Write-Output "scriptBasePath: $scriptBasePath"
Write-Output "modulesBasePath: $modulesBasePath"
Write-Output "modulePath: $modulePath"
Write-Output "AOscriptDirectory: $AOscriptDirectory"
Write-Output "directoryPath: $directoryPath"
Write-Output "Repo_Path: $Repo_Path"
Write-Output "Repo_winget: $Repo_winget"

#################################################################################################################################
################################################# END VARIABLES #################################################################
#################################################################################################################################

###############################################################################################################################
############################################### START MODULE LOADING ##########################################################
###############################################################################################################################
















<#
.SYNOPSIS
Dot-sources all PowerShell scripts in the 'private' folder relative to the script root.

.DESCRIPTION
This function finds all PowerShell (.ps1) scripts in a 'private' folder located in the script root directory and dot-sources them. It logs the process, including any errors encountered, with optional color coding.

.EXAMPLE
Dot-SourcePrivateScripts

Dot-sources all scripts in the 'private' folder and logs the process.

.NOTES
Ensure the Write-EnhancedLog function is defined before using this function for logging purposes.
#>


Write-Host "Starting to call Get-ModulesFolderPath..."

# Store the outcome in $ModulesFolderPath
try {
  
    $ModulesFolderPath = Get-ModulesFolderPath -WindowsPath "C:\code\modules" -UnixPath "/usr/src/code/modules"
    # $ModulesFolderPath = Get-ModulesFolderPath -WindowsPath "$PsScriptRoot\modules" -UnixPath "$PsScriptRoot/modules"
    Write-host "Modules folder path: $ModulesFolderPath"

}
catch {
    Write-Error $_.Exception.Message
}


Write-Host "Starting to call Import-LatestModulesLocalRepository..."
Import-LatestModulesLocalRepository -ModulesFolderPath $ModulesFolderPath

###############################################################################################################################
############################################### END MODULE LOADING ############################################################
###############################################################################################################################
try {
    Ensure-LoggingFunctionExists
    # Continue with the rest of the script here
    # exit
}
catch {
    Write-Host "Critical error: $_"
    exit
}

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# Setup logging
Write-EnhancedLog -Message "Script Started" -Level "INFO"

################################################################################################################################
################################################################################################################################
################################################################################################################################

# Execute InstallAndImportModulesPSGallery function
InstallAndImportModulesPSGallery -moduleJsonPath "$PSScriptRoot/modules.json"

################################################################################################################################
################################################ END MODULE CHECKING ###########################################################
################################################################################################################################

    
################################################################################################################################
################################################ END LOGGING ###################################################################
################################################################################################################################

#  Define the variables to be used for the function
#  $PSADTdownloadParams = @{
#      GithubRepository     = "psappdeploytoolkit/psappdeploytoolkit"
#      FilenamePatternMatch = "PSAppDeployToolkit*.zip"
#      ZipExtractionPath    = Join-Path "$PSScriptRoot\private" "PSAppDeployToolkit"
#  }

#  Call the function with the variables
#  Download-PSAppDeployToolkit @PSADTdownloadParams

################################################################################################################################
################################################ END DOWNLOADING PSADT #########################################################
################################################################################################################################


##########################################################################################################################
############################################STARTING THE MAIN FUNCTION LOGIC HERE#########################################
##########################################################################################################################


################################################################################################################################
################################################ START GRAPH CONNECTING ########################################################
################################################################################################################################
$accessToken = Connect-GraphWithCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword

Log-Params -Params @{accessToken = $accessToken }

Get-TenantDetails
#################################################################################################################################
################################################# END Connecting to Graph #######################################################
#################################################################################################################################





# Variables
$endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
$startDate = (Get-Date).AddDays(-30).ToString("yyyy-MM-ddTHH:mm:ssZ")
$baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
$headers = @{
    "Authorization" = "Bearer $accessToken"
    "Content-Type"  = "application/json"
}
# Initial URL with filters
$url = "$baseUrl`?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"
$intuneUrl = "https://graph.microsoft.com/v1.0/deviceManagement/managedDevices"
$tenantDetailsUrl = "https://graph.microsoft.com/v1.0/organization"

# Log initial parameters
$params = @{
    AccessToken = $accessToken
    EndDate     = $endDate
    StartDate   = $startDate
    BaseUrl     = $baseUrl
    Url         = $url
    IntuneUrl   = $intuneUrl
    TenantUrl   = $tenantDetailsUrl
}
Log-Params -Params $params

# Function to make the API request and handle pagination


# Validate access to required URIs (uncomment when debugging)
# $uris = @($url, $intuneUrl, $tenantDetailsUrl)
# foreach ($uri in $uris) {
#     if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
#         Write-EnhancedLog "Validation failed. Halting script." -Color Red
#         exit 1
#     }
# }


# (we got them already now let's filter them)

# # Get all sign-in logs for the last 30 days
# $signInLogs = Get-SignInLogs -url $url -Headers $headers

# # Export to JSON for further processing
# $signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

# Write-EnhancedLog "Export complete. Check /usr/src/SignInLogs.json for results." -Color Green

# # Load the sign-in logs
# $json = Get-Content -Path '/usr/src/SignInLogs.json' | ConvertFrom-Json




# Define the root path where the scripts and exports are located
# $scriptRoot = "C:\MyScripts"

# Optionally, specify the names for the exports folder and subfolder
$exportsFolderName = "CustomExports"
$exportSubFolderName = "CustomSignInLogs"

# # Call the function to export sign-in logs to XML (and other formats)
# # Define the parameters to be splatted
$ExportAndProcessSignInLogsparams = @{
    ScriptRoot          = $PSscriptRoot
    ExportsFolderName   = $exportsFolderName
    ExportSubFolderName = $exportSubFolderName
    url                 = $url
    Headers             = $headers
}

# Call the function with splatted parameters
# Export-SignInLogs @params


# ExportAndProcessSignInLogs -ScriptRoot $PSscriptRoot -ExportsFolderName $exportsFolderName -ExportSubFolderName $exportSubFolderName -url $url -Headers $headers
ExportAndProcessSignInLogs @ExportAndProcessSignInLogsparams



# # Export to JSON for further processing
# $signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

# Write-EnhancedLog "Export complete. Check /usr/src/SignInLogs.json for results." -Color Green


# Write-EnhancedLog "Export complete. Check ""$PSscriptRoot/$exportsFolderName/$exportSubFolderName/SignInLogs_20240610143318.json"" for results." -Color Green


# # # Load the sign-in logs
# $json = Get-Content -Path "$PSscriptRoot/$exportsFolderName/$exportSubFolderName/SignInLogs_20240610143318.json" | ConvertFrom-Json





# $allSKUs = Get-MgSubscribedSku -Property SkuPartNumber, ServicePlans 
# $allSKUs | ForEach-Object {
#     Write-Host "Service Plan:" $_.SkuPartNumber
#     $_.ServicePlans | ForEach-Object {$_}
# }






















function Process-AllDevicesParallel {
    param (
        [Parameter(Mandatory = $true)]
        [array]$Json,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers,
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot
    )


$results = $Json | ForEach-Object -Parallel {
    # Import and install necessary modules


# Use $using: scope to pass $ScriptRoot
$localScriptRoot = $using:ScriptRoot

    #First, load secrets and create a credential object:
# Assuming secrets.json is in the same directory as your script
$secretsPath = Join-Path -Path $localScriptRoot -ChildPath "secrets.json"

# Load the secrets from the JSON file
$secrets = Get-Content -Path $secretsPath -Raw | ConvertFrom-Json

# Read configuration from the JSON file
# Assign values from JSON to variables

# Read configuration from the JSON file
$configPath = Join-Path -Path $localScriptRoot -ChildPath "config.json"
$env:MYMODULE_CONFIG_PATH = $configPath

$config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

#  Variables from JSON file
$tenantId = $secrets.tenantId
$clientId = $secrets.clientId

$certPath = Join-Path -Path $localScriptRoot -ChildPath 'graphcert.pfx'
$CertPassword = $secrets.CertPassword


function Initialize-Environment {
    param (
        [string]$WindowsModulePath = "EnhancedBoilerPlateAO\2.0.0\EnhancedBoilerPlateAO.psm1",
        [string]$LinuxModulePath = "/usr/src/code/Modules/EnhancedBoilerPlateAO/2.0.0/EnhancedBoilerPlateAO.psm1"
    )

    function Get-Platform {
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            return $PSVersionTable.Platform
        }
        else {
            return [System.Environment]::OSVersion.Platform
        }
    }

    function Setup-GlobalPaths {
        if ($env:DOCKER_ENV -eq $true) {
            $global:scriptBasePath = $env:SCRIPT_BASE_PATH
            $global:modulesBasePath = $env:MODULES_BASE_PATH
        }
        else {
            # $global:scriptBasePath = $PSScriptRoot
            # $global:modulesBasePath = "$PSScriptRoot\modules"

            $global:scriptBasePath = $localScriptRoot
            $global:modulesBasePath = "c:\code\modules"
        }
    }

    function Setup-WindowsEnvironment {
        # Get the base paths from the global variables
        Setup-GlobalPaths

        # Construct the paths dynamically using the base paths
        $global:modulePath = Join-Path -Path $modulesBasePath -ChildPath $WindowsModulePath
        $global:AOscriptDirectory = Join-Path -Path $scriptBasePath -ChildPath "Win32Apps-DropBox"
        $global:directoryPath = Join-Path -Path $scriptBasePath -ChildPath "Win32Apps-DropBox"
        $global:Repo_Path = $scriptBasePath
        $global:Repo_winget = "$Repo_Path\Win32Apps-DropBox"


        # Import the module using the dynamically constructed path
        Import-Module -Name $global:modulePath -Verbose -Force:$true -Global:$true

        # Log the paths to verify
        Write-Output "Module Path: $global:modulePath"
        Write-Output "Repo Path: $global:Repo_Path"
        Write-Output "Repo Winget Path: $global:Repo_winget"
    }

    function Setup-LinuxEnvironment {
        # Get the base paths from the global variables
        Setup-GlobalPaths

        # Import the module using the Linux path
        Import-Module $LinuxModulePath -Verbose

        # Convert paths from Windows to Linux format
        $global:AOscriptDirectory = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer"
        $global:directoryPath = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer\Win32Apps-DropBox"
        $global:Repo_Path = Convert-WindowsPathToLinuxPath -WindowsPath "C:\Users\Admin-Abdullah\AppData\Local\Intune-Win32-Deployer"
        $global:Repo_winget = "$global:Repo_Path\Win32Apps-DropBox"
    }

    $platform = Get-Platform
    if ($platform -eq 'Win32NT' -or $platform -eq [System.PlatformID]::Win32NT) {
        Setup-WindowsEnvironment
    }
    elseif ($platform -eq 'Unix' -or $platform -eq [System.PlatformID]::Unix) {
        Setup-LinuxEnvironment
    }
    else {
        throw "Unsupported operating system"
    }
}

# Call the function to initialize the environment
# Initialize-Environment


# Example usage of global variables outside the function
Write-Output "Global variables set by Initialize-Environment:"
Write-Output "scriptBasePath: $scriptBasePath"
Write-Output "modulesBasePath: $modulesBasePath"
Write-Output "modulePath: $modulePath"
Write-Output "AOscriptDirectory: $AOscriptDirectory"
Write-Output "directoryPath: $directoryPath"
Write-Output "Repo_Path: $Repo_Path"
Write-Output "Repo_winget: $Repo_winget"

#################################################################################################################################
################################################# END VARIABLES #################################################################
#################################################################################################################################

###############################################################################################################################
############################################### START MODULE LOADING ##########################################################
###############################################################################################################################





# # Define the paths
# $rootPath = 'C:\Code\Modules\EnhancedAO.Graph.SignInLogs\2.0.0'
# $privatePath = Join-Path -Path $rootPath -ChildPath 'Private'
# $publicPath = Join-Path -Path $rootPath -ChildPath 'Public'
# $outputFile = Join-Path -Path $rootPath -ChildPath 'CombinedScript.ps1'

# # Initialize the output file
# New-Item -Path $outputFile -ItemType File -Force

# # Function to combine files from a directory
# function Combine-Files {
#     param (
#         [string]$directory
#     )
#     Get-ChildItem -Path $directory -Filter *.ps1 | ForEach-Object {
#         Get-Content -Path $_.FullName | Add-Content -Path $outputFile
#         Add-Content -Path $outputFile -Value "`n"  # Add a new line for separation
#     }
# }

# # Combine files from Private and Public folders
# Combine-Files -directory $privatePath
# Combine-Files -directory $publicPath

# Write-Host "All files have been combined into $outputFile"


# Function to add result to the context
function Add-Result {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$Context,
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Item,
        [Parameter(Mandatory = $true)]
        [string]$DeviceId,
        [Parameter(Mandatory = $true)]
        [string]$DeviceState,
        [Parameter(Mandatory = $true)]
        [bool]$HasPremiumLicense
    )

    try {
        $deviceName = $Item.deviceDetail.displayName
        if ([string]::IsNullOrWhiteSpace($deviceName)) {
            $deviceName = "BYOD"
        }

        $Context.Results.Add([PSCustomObject]@{
            'DeviceName'             = $deviceName
            'UserName'               = $Item.userDisplayName
            'DeviceEntraID'          = $DeviceId
            'UserEntraID'            = $Item.userId
            'DeviceOS'               = $Item.deviceDetail.operatingSystem
            'DeviceComplianceStatus' = if ($Item.deviceDetail.isCompliant) { "Compliant" } else { "Non-Compliant" }
            'DeviceStateInIntune'    = $DeviceState
            'TrustType'              = $Item.deviceDetail.trustType
            'UserLicense'            = if ($HasPremiumLicense) { "Microsoft 365 Business Premium" } else { "Other" }
        })

        # Write-EnhancedLog -Message "Successfully added result for device: $deviceName for user: $($Item.userDisplayName)" -Level "INFO"
    } catch {
        Handle-Error -ErrorRecord $_
        Write-EnhancedLog -Message "Failed to add result for device: $($Item.deviceDetail.displayName)" -Level "ERROR"
    }
}


function Check-DeviceStateInIntune {
    param (
        [Parameter(Mandatory = $true)]
        [string]$entraDeviceId,
        [Parameter(Mandatory = $true)]
        [string]$username,
        [Parameter(Mandatory = $true)]
        [hashtable]$headers
    )

    if (-not [string]::IsNullOrWhiteSpace($entraDeviceId)) {
        # Write-EnhancedLog -Message "Checking device state in Intune for Entra Device ID: $entraDeviceId for username: $username "

        # Construct the Graph API URL to retrieve device details
        $graphApiUrl = "https://graph.microsoft.com/v1.0/deviceManagement/managedDevices?`$filter=azureADDeviceId eq '$entraDeviceId'"
        # Write-EnhancedLog -Message "Constructed Graph API URL: $graphApiUrl"

        # Send the request
        try {
            $response = Invoke-WebRequest -Uri $graphApiUrl -Headers $headers -Method Get
            $data = ($response.Content | ConvertFrom-Json).value

            if ($data -and $data.Count -gt 0) {
                # Write-EnhancedLog -Message "Device is present in Intune."
                return "Present"
            } else {
                # Write-EnhancedLog -Message "Device is absent in Intune."
                return "Absent"
            }
        } catch {
            Handle-Error -ErrorRecord $_
            return "Error"
        }
    } else {
        # Write-EnhancedLog -Message "Device ID is empty, considered as BYOD." #uncomment if verbose output is desired
        return "BYOD"
    }
}


function Fetch-UserLicense {
    param (
        [Parameter(Mandatory = $true)]
        [string]$UserId,
        [Parameter(Mandatory = $true)]
        [string]$Username,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    try {
        # Write-EnhancedLog -Message "Fetching licenses for user: $Username with ID: $UserId"
        $userLicenses = Get-UserLicenses -userId $UserId -username $Item.userDisplayName  -Headers $Headers
        return $userLicenses
    } catch {
        Handle-Error -ErrorRecord $_
        return $null
    }
}


# Function to initialize results and unique device IDs
function Initialize-Results {
    return @{
        Results = [System.Collections.Generic.List[PSCustomObject]]::new()
        UniqueDeviceIds = @{}
    }
}


# $functionInitializeResults = ${function:Initialize-Results}.ToString()
# $functionProcessDeviceItem = ${function:Process-DeviceItem}.ToString()
# $functionHandleError = ${function:Handle-Error}.ToString()

# function Process-AllDevicesParallel {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     # Initialize the results
#     $context = Initialize-Results

#     # Use ForEach-Object -Parallel to process items in parallel
#     $results = $Json | ForEach-Object -Parallel {
#         # Import the necessary functions
#         Invoke-Expression $using:functionInitializeResults
#         Invoke-Expression $using:functionProcessDeviceItem
#         Invoke-Expression $using:functionHandleError

#         # Initialize results in each runspace
#         $localContext = Initialize-Results

#         # Exclude "On-Premises Directory Synchronization Service Account" user
#         if ($using:item.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
#             return
#         }

#         try {
#             Process-DeviceItem -Item $_ -Context $localContext -Headers $using:Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }

#         return $localContext.Results
#     } -ArgumentList $Headers -ThrottleLimit 4

#     return $results
# }





# function Process-AllDevicesParallel {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionInitializeResults,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionProcessDeviceItem,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionHandleError
#     )

#     # Use ForEach-Object -Parallel to process items in parallel
#     $results = $Json | ForEach-Object -Parallel {
#         # Import the necessary functions
#         Invoke-Expression $using:FunctionInitializeResults
#         Invoke-Expression $using:FunctionProcessDeviceItem
#         Invoke-Expression $using:FunctionHandleError

#         # Initialize results in each runspace
#         $localContext = Initialize-Results

#         # Exclude "On-Premises Directory Synchronization Service Account" user
#         if ($_.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
#             return
#         }

#         try {
#             Process-DeviceItem -Item $_ -Context $localContext -Headers $using:Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }

#         return $localContext.Results
#     } -ThrottleLimit 4

#     return $results
# }




# function Process-AllDevicesParallel {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionInitializeResults,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionProcessDeviceItem,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionHandleError,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionImportLatestModulesLocalRepository,
#         [Parameter(Mandatory = $true)]
#         [string]$FunctionInstallAndImportModulesPSGallery,
#         [Parameter(Mandatory = $true)]
#         [string]$ModulesFolderPath,
#         [Parameter(Mandatory = $true)]
#         [string]$ModuleJsonPath
#     )

#     $results = $Json | ForEach-Object -Parallel {


     

#         # Define necessary functions within the parallel block
      
#         Invoke-Expression $using:FunctionImportLatestModulesLocalRepository
#         Invoke-Expression $using:FunctionInstallAndImportModulesPSGallery

#            # Import and install necessary modules
#            Import-LatestModulesLocalRepository -ModulesFolderPath $using:ModulesFolderPath
#            InstallAndImportModulesPSGallery -moduleJsonPath $using:ModuleJsonPath

#         Invoke-Expression $using:FunctionInitializeResults
#         Invoke-Expression $using:FunctionProcessDeviceItem
#         Invoke-Expression $using:FunctionHandleError

#         # Initialize results in each runspace
#         $localContext = Initialize-Results

#         if ($_.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
#             return
#         }

#         try {
#             Process-DeviceItem -Item $_ -Context $localContext -Headers $using:Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }

#         return $localContext.Results
#     } -ThrottleLimit 4

#     return $results
# }





# # Example JSON input and headers
# $jsonInput = @(
#     [pscustomobject]@{ userDisplayName = "User1"; DeviceId = 1 },
#     [pscustomobject]@{ userDisplayName = "User2"; DeviceId = 2 }
#     # Add more items as needed
# )
# $headers = @{ Authorization = "Bearer token" }

# # Process all devices in parallel
# $results = Process-AllDevices -Json $jsonInput -Headers $headers

# # Output the results
# $results




# function Process-AllDevicesParallel {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     $results = $Json | ForEach-Object -Parallel {
#         # Import and install necessary modules


      

#         # Initialize results in each runspace
#         $localContext = Initialize-Results

#         if ($_.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
#             return
#         }

#         try {
#             Process-DeviceItem -Item $_ -Context $localContext -Headers $using:Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }

#         return $localContext.Results
#     } -ThrottleLimit 4

#     return $results
# }



# Function to process each device item
function Process-DeviceItem {
    param (
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Item,
        [Parameter(Mandatory = $true)]
        [hashtable]$Context,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    $deviceId = $Item.deviceDetail.deviceId
    $deviceName = $Item.deviceDetail.displayName

    if ($deviceId -eq "{PII Removed}") {
        if (-not $Context.UniqueDeviceIds.ContainsKey($Item.userId)) {
            Write-EnhancedLog -Message "External Azure AD tenant detected for user: $($Item.userDisplayName)" -Level "INFO"
            Add-Result -Context $Context -Item $Item -DeviceId "N/A" -DeviceState "External" -HasPremiumLicense $false
            $Context.UniqueDeviceIds[$Item.userId] = $true
        }
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($deviceId)) {
        if (-not $Context.UniqueDeviceIds.ContainsKey($deviceId)) {
            $Context.UniqueDeviceIds[$deviceId] = $true
            $deviceState = Check-DeviceStateInIntune -entraDeviceId $deviceId -username $Item.userDisplayName -Headers $Headers

            try {
                $userLicenses = Fetch-UserLicense -UserId $Item.userId -Username $Item.userDisplayName -Headers $Headers
                $hasPremiumLicense = $false

                if ($null -ne $userLicenses -and $userLicenses.Count -gt 0) {
                    $hasPremiumLicense = $userLicenses.Contains("cbdc14ab-d96c-4c30-b9f4-6ada7cdc1d46")
                }

                Add-Result -Context $Context -Item $Item -DeviceId $deviceId -DeviceState $deviceState -HasPremiumLicense ($hasPremiumLicense -eq $true)
            } catch {
                Handle-Error -ErrorRecord $_
            }
        }
    } else {
        # Handle BYOD case
        if (-not $Context.UniqueDeviceIds.ContainsKey($Item.userId)) {
            Add-Result -Context $Context -Item $Item -DeviceId "N/A" -DeviceState "BYOD" -HasPremiumLicense $false
            $Context.UniqueDeviceIds[$Item.userId] = $true
        }
    }
}


function Export-SignInLogs {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$ExportsFolderName,
        [Parameter(Mandatory = $true)]
        [string]$ExportSubFolderName,
        [Parameter(Mandatory = $true)]
        [hashtable]$headers,
        [Parameter(Mandatory = $true)]
        [string]$url
    
    )

    # Ensure the exports folder is clean before exporting
    $exportFolder = Ensure-ExportsFolder -BasePath $ScriptRoot -ExportsFolderName $ExportsFolderName -ExportSubFolderName $ExportSubFolderName

    # Get the sign-in logs (assuming you have a way to fetch these logs)
    # $signInLogs = Get-SignInLogs # Replace with the actual command to get sign-in logs

    $signInLogs = Get-SignInLogs -url $url -Headers $headers

    # Check if there are no sign-in logs
    if ($signInLogs.Count -eq 0) {
        Write-EnhancedLog -Message "NO sign-in logs found." -Level "WARNING"
        return
    }

    # Generate a timestamp for the export
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $baseOutputPath = Join-Path -Path $exportFolder -ChildPath "SignInLogs_$timestamp"

    # Setup parameters for Export-Data using splatting
    $exportParams = @{
        Data             = $signInLogs
        BaseOutputPath   = $baseOutputPath
        # IncludeCSV       = $true
        IncludeJSON      = $true
        # IncludeXML       = $true
        # IncludePlainText = $true
        # IncludeExcel     = $true
        # IncludeYAML      = $true
    }

    # Call the Export-Data function with splatted parameters
    Export-Data @exportParams
    Write-EnhancedLog -Message "Data export completed successfully." -Level "INFO"
}


# # Define the root path where the scripts and exports are located
# $scriptRoot = "C:\MyScripts"

# # Optionally, specify the names for the exports folder and subfolder
# $exportsFolderName = "CustomExports"
# $exportSubFolderName = "CustomSignInLogs"

# # Call the function to export sign-in logs to XML (and other formats)
# Export-SignInLogsToXML -ScriptRoot $scriptRoot -ExportsFolderName $exportsFolderName -ExportSubFolderName $exportSubFolderName


# Main script logic
# function ExportAndProcessSignInLogs {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$ScriptRoot,
#         [Parameter(Mandatory = $true)]
#         [string]$ExportsFolderName,
#         [Parameter(Mandatory = $true)]
#         [string]$ExportSubFolderName,
#         [Parameter(Mandatory = $true)]
#         [string]$url,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     try {
#         $ExportSignInLogsparams = @{
#             ScriptRoot         = $ScriptRoot
#             ExportsFolderName  = $ExportsFolderName
#             ExportSubFolderName= $ExportSubFolderName
#             url                = $url
#             Headers            = $Headers
#         }
        
#         # Call the function with splatted parameters
#         # Export-SignInLogs @ExportSignInLogsparams (uncomment if you want to export fresh sign-in logs)

#         $subFolderPath = Join-Path -Path $ScriptRoot -ChildPath $ExportsFolderName
#         $subFolderPath = Join-Path -Path $subFolderPath -ChildPath $ExportSubFolderName

#         $latestJsonFile = Find-LatestJsonFile -Directory $subFolderPath

#         if ($latestJsonFile) {
#             $global:json = Load-SignInLogs -JsonFilePath $latestJsonFile
     
#             # Further processing of $json can go here...
#         } else {
#             Write-EnhancedLog -Message "No JSON file found to load sign-in logs." -Level "WARNING"
#         }
#     } catch {
#         Handle-Error -ErrorRecord $_
#     }
# }




# Example integration in the main script logic
function ExportAndProcessSignInLogs {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$ExportsFolderName,
        [Parameter(Mandatory = $true)]
        [string]$ExportSubFolderName,
        [Parameter(Mandatory = $true)]
        [string]$url,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    try {
        $ExportSignInLogsparams = @{
            ScriptRoot         = $ScriptRoot
            ExportsFolderName  = $ExportsFolderName
            ExportSubFolderName= $ExportSubFolderName
            url                = $url
            Headers            = $Headers
        }
        
        # Call the function with splatted parameters
        # Export-SignInLogs @ExportSignInLogsparams (uncomment if you want to export fresh sign-in logs to a new JSON file)

        $subFolderPath = Join-Path -Path $ScriptRoot -ChildPath $ExportsFolderName
        $subFolderPath = Join-Path -Path $subFolderPath -ChildPath $ExportSubFolderName

        $latestJsonFile = Find-LatestJsonFile -Directory $subFolderPath

        if ($latestJsonFile) {
            # $signInLogs = Load-SignInLogs -JsonFilePath $latestJsonFile
            $global:signInLogs = Load-SignInLogs -JsonFilePath $latestJsonFile
            if ($signInLogs.Count -gt 0) {
                # $results = Process-AllDevices -Json $signInLogs -Headers $Headers
                # Further processing of $results can go here...

                # Write-EnhancedLog -Message "sign-in logs found in $latestJsonFile. Starting to process it" -Level "INFO"
            } else {
                Write-EnhancedLog -Message "No sign-in logs found in $latestJsonFile." -Level "WARNING"
            }
        } else {
            Write-EnhancedLog -Message "No JSON file found to load sign-in logs." -Level "WARNING"
        }
    } catch {
        Handle-Error -ErrorRecord $_
    }
}



# Function to find the latest JSON file in the specified directory
function Find-LatestJsonFile {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Directory
    )

    $jsonFiles = Get-ChildItem -Path $Directory -Filter *.json | Sort-Object LastWriteTime -Descending

    if ($jsonFiles.Count -gt 0) {
        return $jsonFiles[0].FullName
    } else {
        Write-EnhancedLog -Message "No JSON files found in $Directory." -Level "ERROR"
        return $null
    }
}




# Function to generate reports based on user licenses
function Generate-LicenseReports {
    param (
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[PSCustomObject]]$Results,
        [Parameter(Mandatory = $true)]
        [string]$PSScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$ExportsFolderName
    )

    # Remove duplicates based on UserEntraID
    $uniqueResults = $Results | Sort-Object -Property UserEntraID -Unique

    # Generate reports for users with and without Business Premium licenses
    $premiumLicenses = $uniqueResults | Where-Object { $_.UserLicense -eq 'Microsoft 365 Business Premium' }
    $nonPremiumLicenses = $uniqueResults | Where-Object { $_.UserLicense -ne 'Microsoft 365 Business Premium' }

    $premiumLicenses | Export-Csv "$PSScriptRoot/$ExportsFolderName/Report_PremiumLicenses.csv" -NoTypeInformation
    $nonPremiumLicenses | Export-Csv "$PSScriptRoot/$ExportsFolderName/Report_NonPremiumLicenses.csv" -NoTypeInformation

    # Output totals to console
    Write-EnhancedLog -Message "Total users with Business Premium licenses: $($premiumLicenses.Count)" -Level "INFO"
    Write-EnhancedLog -Message "Total users without Business Premium licenses: $($nonPremiumLicenses.Count)" -Level "INFO"

    Write-EnhancedLog -Message "Generated reports for users with and without Business Premium licenses." -Level "INFO"
}

# # Example usage
# $Json = @() # Your JSON data here
# $Headers = @{} # Your actual headers
# $PSScriptRoot = "C:\Path\To\ScriptRoot" # Update to your script root path
# $ExportsFolderName = "CustomExports"

# $results = Process-AllDevices -Json $Json -Headers $Headers

# # Generate and export the reports
# Generate-LicenseReports -Results $results -PSScriptRoot $PSScriptRoot -ExportsFolderName $ExportsFolderName


# Function to generate a report for PII Removed cases
function Generate-PII-RemovedReport {
    param (
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[PSCustomObject]]$Results,
        [Parameter(Mandatory = $true)]
        [string]$PSScriptRoot,
        [Parameter(Mandatory = $true)]
        [string]$ExportsFolderName
    )

    # Filter results for PII Removed (external) cases
    $piiRemovedResults = $Results | Where-Object { $_.DeviceStateInIntune -eq 'External' }

    # Export the results to a CSV file
    $piiRemovedResults | Export-Csv "$PSScriptRoot/$ExportsFolderName/Report_PIIRemoved.csv" -NoTypeInformation

    # Output totals to console
    Write-EnhancedLog -Message "Total users with PII Removed (external Azure AD/Entra ID tenants): $($piiRemovedResults.Count)" -Level "Warning"
    Write-EnhancedLog -Message "Generated report for users with PII Removed (external Azure AD/Entra ID tenants." -Level "INFO"
}

# # Example usage
# $Json = @() # Your JSON data here
# $Headers = @{} # Your actual headers
# $PSScriptRoot = "C:\Path\To\ScriptRoot" # Update to your script root path
# $ExportsFolderName = "CustomExports"

# $results = Process-AllDevices -Json $Json -Headers $Headers

# # Generate and export the PII Removed report
# Generate-PII-RemovedReport -Results $results -PSScriptRoot $PSScriptRoot -ExportsFolderName $ExportsFolderName


# Function to fetch user licenses
function Get-UserLicenses {
    param (
        [Parameter(Mandatory = $true)]
        [string]$userId,
        [Parameter(Mandatory = $true)]
        [string]$username,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    $licenses = [System.Collections.Generic.List[string]]::new()
    $uri = "https://graph.microsoft.com/v1.0/users/$userId/licenseDetails"

    try {
        # Write-EnhancedLog -Message "Fetching licenses for user ID: $userId with username: $username" -Level "INFO"

        $response = Invoke-RestMethod -Uri $uri -Headers $Headers -Method Get

        if ($null -ne $response -and $null -ne $response.value) {
            foreach ($license in $response.value) {
                $licenses.Add($license.skuId)
            }
        } else {
            Write-EnhancedLog -Message "No license details found for user ID: $userId" -Level "WARNING"
        }
    } catch {
        Handle-Error -ErrorRecord $_
        throw
    }

    return $licenses
}







# # Function to load sign-in logs from the latest JSON file
# function Load-SignInLogs {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$JsonFilePath
#     )

#     try {
#         $json = Get-Content -Path $JsonFilePath | ConvertFrom-Json
#         Write-EnhancedLog -Message "Sign-in logs loaded successfully from $JsonFilePath." -Level "INFO"
#         return $json
#     } catch {
#         Handle-Error -ErrorRecord $_
        
#     }
# }



class SignInLog {
    [string] $userDisplayName
    [string] $userId
    [DeviceDetail] $deviceDetail

    SignInLog([string] $userDisplayName, [string] $userId, [DeviceDetail] $deviceDetail) {
        $this.userDisplayName = $userDisplayName
        $this.userId = $userId
        $this.deviceDetail = $deviceDetail
    }
}

class DeviceDetail {
    [string] $deviceId
    [string] $displayName
    [string] $operatingSystem
    [bool] $isCompliant
    [string] $trustType

    DeviceDetail([string] $deviceId, [string] $displayName, [string] $operatingSystem, [bool] $isCompliant, [string] $trustType) {
        $this.deviceId = $deviceId
        $this.displayName = $displayName
        $this.operatingSystem = $operatingSystem
        $this.isCompliant = $isCompliant
        $this.trustType = $trustType
    }
}




# Function to load sign-in logs from the latest JSON file
function Load-SignInLogs {
    param (
        [Parameter(Mandatory = $true)]
        [string]$JsonFilePath
    )

    $signInLogs = [System.Collections.Generic.List[SignInLog]]::new()
    $fileStream = [System.IO.File]::OpenRead($JsonFilePath)

    try {
        $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

        foreach ($element in $jsonDoc.RootElement.EnumerateArray()) {
            $deviceDetail = [DeviceDetail]::new(
                $element.GetProperty("deviceDetail").GetProperty("deviceId").GetString(),
                $element.GetProperty("deviceDetail").GetProperty("displayName").GetString(),
                $element.GetProperty("deviceDetail").GetProperty("operatingSystem").GetString(),
                $element.GetProperty("deviceDetail").GetProperty("isCompliant").GetBoolean(),
                $element.GetProperty("deviceDetail").GetProperty("trustType").GetString()
            )
            $signInLog = [SignInLog]::new(
                $element.GetProperty("userDisplayName").GetString(),
                $element.GetProperty("userId").GetString(),
                $deviceDetail
            )

            $signInLogs.Add($signInLog)
        }

        # Write-EnhancedLog -Message "Sign-in logs loaded successfully from $JsonFilePath." -Level "INFO"
    } catch {
        Handle-Error -ErrorRecord $_
    } finally {
        $fileStream.Dispose()
    }

    return $signInLogs
}










# # Function to load sign-in logs from the latest JSON file
# function Load-SignInLogs {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$JsonFilePath
#     )

#     $signInLogs = [System.Collections.Generic.List[PSCustomObject]]::new()
#     $fileStream = [System.IO.File]::OpenRead($JsonFilePath)

#     try {
#         $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

#         foreach ($element in $jsonDoc.RootElement.EnumerateArray()) {
#             $signInLog = [PSCustomObject]@{
#                 userDisplayName = $element.GetProperty("userDisplayName").GetString()
#                 userId = $element.GetProperty("userId").GetString()
#                 deviceDetail = [PSCustomObject]@{
#                     deviceId = $element.GetProperty("deviceDetail").GetProperty("deviceId").GetString()
#                     displayName = $element.GetProperty("deviceDetail").GetProperty("displayName").GetString()
#                     operatingSystem = $element.GetProperty("deviceDetail").GetProperty("operatingSystem").GetString()
#                     isCompliant = $element.GetProperty("deviceDetail").GetProperty("isCompliant").GetBoolean()
#                     trustType = $element.GetProperty("deviceDetail").GetProperty("trustType").GetString()
#                 }
#             }

#             $signInLogs.Add($signInLog)
#         }

#         Write-EnhancedLog -Message "Sign-in logs loaded successfully from $JsonFilePath." -Level "INFO"
#     } catch {
#         Handle-Error -ErrorRecord $_
#     } finally {
#         $fileStream.Dispose()
#     }

#     return $signInLogs
# }













# class SignInLog {
#     [string] $userDisplayName
#     [string] $deviceID
#     [datetime] $signInDateTime
#     # Add other relevant properties here
# }

# # Function to load and deserialize sign-in logs from the latest JSON file
# function Load-SignInLogs {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$JsonFilePath
#     )

#     try {
#         # Load the JSON file
#         $reader = [System.IO.StreamReader]::new($JsonFilePath)
#         $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
        
#         # Filter out specific users and deserialize to SignInLog class
#         $filteredLogs = $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]').ToObject[SignInLog]()
        
#         Write-EnhancedLog -Message "Sign-in logs loaded and filtered successfully from $JsonFilePath." -Level "INFO"
#         return $filteredLogs
#     } catch {
#         Handle-Error -ErrorRecord $_
#         # return $null
#     }
# }


# Main function to process all devices
# function Process-AllDevices {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     $context = Initialize-Results

#     foreach ($item in $Json) {
#         # Exclude "On-Premises Directory Synchronization Service Account" user
#         if ($item.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
#             # Write-EnhancedLog -Message "Skipping user: $($item.userDisplayName)" -Level "INFO"
#             continue
#         }

#         try {
#             Process-DeviceItem -Item $item -Context $context -Headers $Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }
#     }

#     return $context.Results
# }
# # Example usage
# $Json = @() # Your JSON data here
# $Headers = @{} # Your actual headers

# $results = Process-AllDevices -Json $Json -Headers $Headers






# Main function to process all devices
function Process-AllDevices {
    param (
        [Parameter(Mandatory = $true)]
        [array]$Json,
        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    $context = Initialize-Results

    foreach ($item in $Json) {
        # Exclude "On-Premises Directory Synchronization Service Account" user
        if ($item.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
            continue
        }

        try {
            Process-DeviceItem -Item $item -Context $context -Headers $Headers
        } catch {
            Handle-Error -ErrorRecord $_
        }
    }

    return $context.Results
}









# Main function to process all devices
# function Process-AllDevices {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$JsonFilePath,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     # Load the JSON file using Newtonsoft JSON library
#     $reader = [System.IO.StreamReader]::new($JsonFilePath)
#     $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)

#     # Filter out "On-Premises Directory Synchronization Service Account" user
#     $filteredJson = $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]')

#     $context = Initialize-Results

#     foreach ($item in $filteredJson) {
#         try {
#             Process-DeviceItem -Item $item -Context $context -Headers $Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }
#     }

#     return $context.Results
# }

# Example call to the function
# $Headers = @{"Authorization" = "Bearer <your_token>"}
# Process-AllDevices -JsonFilePath "path\to\your\jsonfile.json" -Headers $Headers







# Main function to process all devices
# function Process-AllDevices {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$JsonContent,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     # Parse JSON content using Newtonsoft JSON library
#     $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($JsonContent)

#     # Filter out "On-Premises Directory Synchronization Service Account" user
#     $filteredJson = $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]')

#     $context = Initialize-Results

#     foreach ($item in $filteredJson) {
#         try {
#             Process-DeviceItem -Item $item -Context $context -Headers $Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }
#     }

#     return $context.Results
# }

# # Read JSON content from file
# $JsonFilePath = "path\to\your\jsonfile.json"
# $JsonContent = Get-Content -Path $JsonFilePath -Raw

# # Example call to the function
# $Headers = @{"Authorization" = "Bearer <your_token>"}
# Process-AllDevices -JsonContent $JsonContent -Headers $Headers





# function Process-AllDevices {
#     param (
#         [Parameter(Mandatory = $true)]
#         [array]$Json,
#         [Parameter(Mandatory = $true)]
#         [hashtable]$Headers
#     )

#     # Filter out "On-Premises Directory Synchronization Service Account" user
#     $jarray = [Newtonsoft.Json.Linq.JArray]::FromObject($Json)
#     $filteredJson = $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]')

#     $context = Initialize-Results

#     foreach ($item in $filteredJson) {
#         try {
#             Process-DeviceItem -Item $item -Context $context -Headers $Headers
#         } catch {
#             Handle-Error -ErrorRecord $_
#         }
#     }

#     return $context.Results
# }


# # Define the paths
# $rootPath = 'C:\Code\Modules\EnhancedBoilerPlateAO\2.0.0'
# # $privatePath = Join-Path -Path $rootPath -ChildPath 'Private'
# $publicPath = Join-Path -Path $rootPath -ChildPath 'Public'
# $outputFile = Join-Path -Path $rootPath -ChildPath 'CombinedScript.ps1'

# # Initialize the output file
# New-Item -Path $outputFile -ItemType File -Force

# # Function to combine files from a directory
# function Combine-Files {
#     param (
#         [string]$directory
#     )
#     Get-ChildItem -Path $directory -Filter *.ps1 | ForEach-Object {
#         Get-Content -Path $_.FullName | Add-Content -Path $outputFile
#         Add-Content -Path $outputFile -Value "`n"  # Add a new line for separation
#     }
# }

# # Combine files from Private and Public folders
# # Combine-Files -directory $privatePath
# Combine-Files -directory $publicPath

# Write-Host "All files have been combined into $outputFile"



# ################################################################################################################################
# ################################################ CALLING AS SYSTEM (Uncomment for debugging) ###################################
# ################################################################################################################################

# Assuming Invoke-AsSystem and Write-EnhancedLog are already defined
# Update the path to your actual location of PsExec64.exe

# Write-EnhancedLog -Message "calling Test-RunningAsSystem" -Level "INFO"
# if (-not (Test-RunningAsSystem)) {
#     $privateFolderPath = Join-Path -Path $PSScriptRoot -ChildPath "private"

#     # Check if the private folder exists, and create it if it does not
#     if (-not (Test-Path -Path $privateFolderPath)) {
#         New-Item -Path $privateFolderPath -ItemType Directory | Out-Null
#     }
    
#     $PsExec64Path = Join-Path -Path $privateFolderPath -ChildPath "PsExec64.exe"
    

#     Write-EnhancedLog -Message "Current session is not running as SYSTEM. Attempting to invoke as SYSTEM..." -Level "INFO"

#     $ScriptToRunAsSystem = $MyInvocation.MyCommand.Path
#     Invoke-AsSystem -PsExec64Path $PsExec64Path -ScriptPath $ScriptToRunAsSystem -TargetFolder $privateFolderPath

# }
# else {
#     Write-EnhancedLog -Message "Session is already running as SYSTEM." -Level "INFO"
# }



# ################################################################################################################################
# ################################################ END CALLING AS SYSTEM (Uncomment for debugging) ###############################
# ################################################################################################################################



function Check-ModuleVersionStatus {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$ModuleNames
    )

    #the following modules PowerShellGet and PackageManagement has to be either automatically imported or manually imported into C:\windows\System32\WindowsPowerShell\v1.0\Modules

    Import-Module -Name PowerShellGet -ErrorAction SilentlyContinue
    # Import-Module 'C:\Program Files (x86)\WindowsPowerShell\Modules\PowerShellGet\PSModule.psm1' -ErrorAction SilentlyContinue
    # Import-Module 'C:\windows\System32\WindowsPowerShell\v1.0\Modules\PowerShellGet\PSModule.psm1' -ErrorAction SilentlyContinue
    # Import-Module 'C:\Program Files (x86)\WindowsPowerShell\Modules\PackageManagement\PackageProviderFunctions.psm1' -ErrorAction SilentlyContinue
    # Import-Module 'C:\windows\System32\WindowsPowerShell\v1.0\Modules\PackageManagement\PackageProviderFunctions.psm1' -ErrorAction SilentlyContinue
    # Import-Module 'C:\Program Files (x86)\WindowsPowerShell\Modules\PackageManagement\PackageManagement.psm1' -ErrorAction SilentlyContinue

    $results = New-Object System.Collections.Generic.List[PSObject]  # Initialize a List to hold the results

    foreach ($ModuleName in $ModuleNames) {
        try {

            Write-Host 'Checking module '$ModuleName
            $installedModule = Get-Module -ListAvailable -Name $ModuleName | Sort-Object Version -Descending | Select-Object -First 1
            # $installedModule = Check-SystemWideModule -ModuleName 'Pester'
            $latestModule = Find-Module -Name $ModuleName -ErrorAction SilentlyContinue

            if ($installedModule -and $latestModule) {
                if ($installedModule.Version -lt $latestModule.Version) {
                    $results.Add([PSCustomObject]@{
                        ModuleName = $ModuleName
                        Status = "Outdated"
                        InstalledVersion = $installedModule.Version
                        LatestVersion = $latestModule.Version
                    })
                } else {
                    $results.Add([PSCustomObject]@{
                        ModuleName = $ModuleName
                        Status = "Up-to-date"
                        InstalledVersion = $installedModule.Version
                        LatestVersion = $installedModule.Version
                    })
                }
            } elseif (-not $installedModule) {
                $results.Add([PSCustomObject]@{
                    ModuleName = $ModuleName
                    Status = "Not Installed"
                    InstalledVersion = $null
                    LatestVersion = $null
                })
            } else {
                $results.Add([PSCustomObject]@{
                    ModuleName = $ModuleName
                    Status = "Not Found in Gallery"
                    InstalledVersion = $null
                    LatestVersion = $null
                })
            }
        } catch {
            Write-Error "An error occurred checking module '$ModuleName': $_"
        }
    }

    return $results
}

# Example usage:
# $versionStatuses = Check-ModuleVersionStatus -ModuleNames @('Pester', 'AzureRM', 'PowerShellGet')
# $versionStatuses | Format-Table -AutoSize  # Display the results in a table format for readability




function Ensure-LoggingFunctionExists {
    if (Get-Command Write-EnhancedLog -ErrorAction SilentlyContinue) {
        Write-EnhancedLog -Message "Logging works" -Level "INFO"
    }
    else {
        throw "Write-EnhancedLog function not found. Terminating script."
    }
}


function Get-ModulesFolderPath {
    param (
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath,
        [Parameter(Mandatory = $true)]
        [string]$UnixPath
    )

    # Auxiliary function to detect OS and set the Modules folder path
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        if ($PSVersionTable.Platform -eq 'Win32NT') {
            return $WindowsPath
        }
        elseif ($PSVersionTable.Platform -eq 'Unix') {
            return $UnixPath
        }
        else {
            throw "Unsupported operating system"
        }
    }
    else {
        $os = [System.Environment]::OSVersion.Platform
        if ($os -eq [System.PlatformID]::Win32NT) {
            return $WindowsPath
        }
        elseif ($os -eq [System.PlatformID]::Unix) {
            return $UnixPath
        }
        else {
            throw "Unsupported operating system"
        }
    }
}


# function Get-ModulesScriptPathsAndVariables {
    

#     <#
#     .SYNOPSIS
#     Dot-sources all PowerShell scripts in the 'Modules' folder relative to the script root.
    
#     .DESCRIPTION
#     This function finds all PowerShell (.ps1) scripts in a 'Modules' folder located in the script root directory and dot-sources them. It logs the process, including any errors encountered, with optional color coding.
    
#     .EXAMPLE
#     Dot-SourceModulesScripts
    
#     Dot-sources all scripts in the 'Modules' folder and logs the process.
    
#     .NOTES
#     Ensure the Write-EnhancedLog function is defined before using this function for logging purposes.
#     #>
#         param (
#             [string]$BaseDirectory
#         )
    
#         try {
#             $ModulesFolderPath = Join-Path -Path $BaseDirectory -ChildPath "Modules"
            
#             if (-not (Test-Path -Path $ModulesFolderPath)) {
#                 throw "Modules folder path does not exist: $ModulesFolderPath"
#             }
    
#             # Construct and return a PSCustomObject
#             return [PSCustomObject]@{
#                 BaseDirectory     = $BaseDirectory
#                 ModulesFolderPath = $ModulesFolderPath
#             }
#         }
#         catch {
#             Write-Host "Error in finding Modules script files: $_"
#             # Optionally, you could return a PSCustomObject indicating an error state
#             # return [PSCustomObject]@{ Error = $_.Exception.Message }
#         }
#     }


function Import-LatestModulesLocalRepository {

    <#
.SYNOPSIS
    Imports the latest version of all modules found in the specified Modules directory.

.DESCRIPTION
    This function scans the Modules directory for module folders, identifies the latest version of each module,
    and attempts to import the module. If a module file is not found or if importing fails, appropriate error
    messages are logged.

.PARAMETER None
    This function does not take any parameters.

.NOTES
    This function assumes the presence of a custom function 'Import-ModuleWithRetry' for retrying module imports.

.EXAMPLE
    ImportLatestModules
    This example imports the latest version of all modules found in the Modules directory.
#>

    [CmdletBinding()]
    param (
        $ModulesFolderPath
    )

    Begin {
        # Get the path to the Modules directory
        # $modulesDir = Join-Path -Path $PSScriptRoot -ChildPath "Modules"
        # $modulesDir = "C:\code\Modules"

        # Get all module directories
        $moduleDirectories = Get-ChildItem -Path $ModulesFolderPath -Directory

        Write-Host "moduleDirectories is $moduleDirectories"

        # Log the number of discovered module directories
        write-host "Discovered module directories: $($moduleDirectories.Count)" 
    }

    Process {
        foreach ($moduleDir in $moduleDirectories) {
            # Get the latest version directory for the current module
            $latestVersionDir = Get-ChildItem -Path $moduleDir.FullName -Directory | Sort-Object Name -Descending | Select-Object -First 1

            if ($null -eq $latestVersionDir) {
                write-host "No version directories found for module: $($moduleDir.Name)"
                continue
            }

            # Construct the path to the module file
            $modulePath = Join-Path -Path $latestVersionDir.FullName -ChildPath "$($moduleDir.Name).psm1"

            # Check if the module file exists
            if (Test-Path -Path $modulePath) {
                # Import the module with retry logic
                try {
                    Import-ModuleWithRetry -ModulePath $modulePath
                    # Import-Module $ModulePath -ErrorAction Stop -Verbose
                    write-host "Successfully imported module: $($moduleDir.Name) from version: $($latestVersionDir.Name)" 
                }
                catch {
                    write-host "Failed to import module: $($moduleDir.Name) from version: $($latestVersionDir.Name). Error: $_" 
                }
            }
            else {
                write-host  "Module file not found: $modulePath"
            }
        }
    }

    End {
        write-host "Module import process completed using Import-LatestModulesLocalRepository from $moduleDirectories"
    }
}


function Import-Modules {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )
    
    foreach ($module in $Modules) {
        if (Get-Module -ListAvailable -Name $module) {
            # Import-Module -Name $module -Force -Verbose
            Import-Module -Name $module -Force:$true -Global:$true
            Write-EnhancedLog -Message "Module '$module' imported." -Level "INFO"
        }
        else {
            Write-EnhancedLog -Message "Module '$module' not found. Cannot import." -Level "ERROR"
        }
    }
}


function Import-ModuleWithRetry {
    <#
    .SYNOPSIS
    Imports a PowerShell module with retries on failure.

    .DESCRIPTION
    This function attempts to import a specified PowerShell module, retrying the import process up to a specified number of times upon failure. It also checks if the module path exists before attempting to import.

    .PARAMETER ModulePath
    The path to the PowerShell module file (.psm1) that should be imported.

    .PARAMETER MaxRetries
    The maximum number of retries to attempt if importing the module fails. Default is 3.

    .PARAMETER WaitTimeSeconds
    The number of seconds to wait between retry attempts. Default is 2 seconds.

    .EXAMPLE
    $modulePath = "C:\Modules\MyPowerShellModule.psm1"
    Import-ModuleWithRetry -ModulePath $modulePath

    Tries to import the module located at "C:\Modules\MyPowerShellModule.psm1", with up to 3 retries, waiting 2 seconds between each retry.

    .NOTES
    This function requires the `Write-EnhancedLog` function to be defined in the script for logging purposes.

    .LINK
    Write-EnhancedLog
    #>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$ModulePath,

        [int]$MaxRetries = 3,

        [int]$WaitTimeSeconds = 2
    )

    Begin {
        $retryCount = 0
        $isModuleLoaded = $false
        Write-Host "Starting to import module from path: $ModulePath"
        
        # Check if the module file exists before attempting to load it
        if (-not (Test-Path -Path $ModulePath -PathType Leaf)) {
            Write-Host "The module path '$ModulePath' does not exist."
            return
        }
    }

    Process {
        while (-not $isModuleLoaded -and $retryCount -lt $MaxRetries) {
            try {
                # Import-Module $ModulePath -ErrorAction Stop -Verbose -Global
                Import-Module $ModulePath -ErrorAction Stop -Global
                # Import-Module $ModulePath -ErrorAction Stop
                $isModuleLoaded = $true
                write-host "Module: $ModulePath imported successfully."
            }
            catch {
                $errorMsg = $_.Exception.Message
                Write-Host "Attempt $retryCount to load module failed: $errorMsg Waiting $WaitTimeSeconds seconds before retrying."
                Write-Host "Attempt $retryCount to load module failed with error: $errorMsg"
                Start-Sleep -Seconds $WaitTimeSeconds
            }
            finally {
                $retryCount++
            }

            if ($retryCount -eq $MaxRetries -and -not $isModuleLoaded) {
                Write-Host "Failed to import module after $MaxRetries retries."
                Write-Host "Failed to import module after $MaxRetries retries with last error: $errorMsg"
                break
            }
        }
    }

    End {
        if ($isModuleLoaded) {
            write-host "Module: $ModulePath loaded successfully."
        }
        else {
            Write-Host -Message "Failed to load module $ModulePath within the maximum retry limit."
        }
    }
}


function Install-Modules {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )

    # Check if running in PowerShell 5 or in a Windows environment
    if ($PSVersionTable.PSVersion.Major -eq 5 -or ($PSVersionTable.Platform -eq 'Win32NT' -or [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT)) {
        # Install the NuGet package provider if the condition is met
        # Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force -Scope CurrentUser
    }

    
    foreach ($module in $Modules) {
        if (-not (Get-Module -ListAvailable -Name $module)) {
            # Install-Module -Name $module -Force -Scope AllUsers
            Install-Module -Name $module -Force -Scope CurrentUser
            Write-EnhancedLog -Message "Module '$module' installed." -Level "INFO"
        }
        else {
            Write-EnhancedLog -Message "Module '$module' is already installed." -Level "INFO"
        }
    }
}


# function Install-RequiredModules {

#     [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

#     # $requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Authentication")
#     $requiredModules = @("Microsoft.Graph.Authentication")

#     foreach ($module in $requiredModules) {
#         if (!(Get-Module -ListAvailable -Name $module)) {

#             Write-EnhancedLog -Message "Installing module: $module" -Level "INFO"
#             Install-Module -Name $module -Force
#             Write-EnhancedLog -Message "Module: $module has been installed" -Level "INFO"
#         }
#         else {
#             Write-EnhancedLog -Message "Module $module is already installed" -Level "INFO"
#         }
#     }


#     $ImportedModules = @("Microsoft.Graph.Identity.DirectoryManagement", "Microsoft.Graph.Authentication")
    
#     foreach ($Importedmodule in $ImportedModules) {
#         if ((Get-Module -ListAvailable -Name $Importedmodule)) {
#             Write-EnhancedLog -Message "Importing module: $Importedmodule" -Level "INFO"
#             Import-Module -Name $Importedmodule
#             Write-EnhancedLog -Message "Module: $Importedmodule has been Imported" -Level "INFO"
#         }
#     }


# }


function InstallAndImportModulesPSGallery {

    <#
.SYNOPSIS
    Validates, installs, and imports required PowerShell modules specified in a JSON file.

.DESCRIPTION
    This function reads the 'modules.json' file from the script's directory, validates the existence of the required modules,
    installs any that are missing, and imports the specified modules into the current session.

.PARAMETER None
    This function does not take any parameters.

.NOTES
    This function relies on a properly formatted 'modules.json' file in the script's root directory.
    The JSON file should have 'requiredModules' and 'importedModules' arrays defined.

.EXAMPLE
    InstallAndImportModules
    This example reads the 'modules.json' file, installs any missing required modules, and imports the specified modules.
#>

    # Define the path to the modules.json file
    # $moduleJsonPath = "$PSScriptRoot/modules.json"

    param (
        [Parameter(Mandatory = $true)]
        [string]$moduleJsonPath
    )
    
    if (Test-Path -Path $moduleJsonPath) {
        try {
            # Read and convert JSON data from the modules.json file
            $moduleData = Get-Content -Path $moduleJsonPath | ConvertFrom-Json
            $requiredModules = $moduleData.requiredModules
            $importedModules = $moduleData.importedModules

            # Validate, Install, and Import Modules
            if ($requiredModules) {
                Install-Modules -Modules $requiredModules
            }
            if ($importedModules) {
                Import-Modules -Modules $importedModules
            }

            Write-EnhancedLog -Message "Modules installed and imported successfully." -Level "INFO"
        }
        catch {
            Write-EnhancedLog -Message "Error processing modules.json: $_" -Level "ERROR"
        }
    }
    else {
        Write-EnhancedLog -Message "modules.json file not found." -Level "ERROR"
    }
}


function Generate-JWTAssertion {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$jwtHeader,
        [Parameter(Mandatory = $true)]
        [hashtable]$jwtPayload,
        [Parameter(Mandatory = $true)]
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$cert
    )

    $jwtHeaderJson = ($jwtHeader | ConvertTo-Json -Compress)
    $jwtPayloadJson = ($jwtPayload | ConvertTo-Json -Compress)
    $jwtHeaderEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($jwtHeaderJson)).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $jwtPayloadEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($jwtPayloadJson)).TrimEnd('=').Replace('+', '-').Replace('/', '_')

    $dataToSign = "$jwtHeaderEncoded.$jwtPayloadEncoded"
    $sha256 = [Security.Cryptography.SHA256]::Create()
    $hash = $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($dataToSign))

    $rsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($cert)
    $signature = [Convert]::ToBase64String($rsa.SignHash($hash, [Security.Cryptography.HashAlgorithmName]::SHA256, [Security.Cryptography.RSASignaturePadding]::Pkcs1)).TrimEnd('=').Replace('+', '-').Replace('/', '_')

    return "$dataToSign.$signature"
}


function Get-UnixTime {
    param (
        [Parameter(Mandatory = $true)]
        [int]$offsetMinutes
    )

    return [int]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + ($offsetMinutes * 60))
}


function Send-TokenRequest {
    param (
        [Parameter(Mandatory = $true)]
        [string]$tokenEndpoint,
        [Parameter(Mandatory = $true)]
        [string]$clientId,
        [Parameter(Mandatory = $true)]
        [string]$clientAssertion
    )

    $body = @{
        client_id = $clientId
        scope = "https://graph.microsoft.com/.default"
        client_assertion = $clientAssertion
        client_assertion_type = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
        grant_type = "client_credentials"
    }

    try {
        Write-EnhancedLog -Message "Sending request to token endpoint: $tokenEndpoint" -Level "INFO"
        $response = Invoke-RestMethod -Method Post -Uri $tokenEndpoint -ContentType "application/x-www-form-urlencoded" -Body $body
        Write-EnhancedLog -Message "Successfully obtained access token." -Level "INFO"
        return $response.access_token
    }
    catch {
        Write-EnhancedLog -Message "Error obtaining access token: $_"
        throw $_
    }
}


function Update-ApplicationPermissions {
    param (
        [string]$appId,
        [string]$permissionsFile
    )

    $resourceAppId = "00000003-0000-0000-c000-000000000000"  # Microsoft Graph

    # Load permissions from the JSON file
    if (Test-Path -Path $permissionsFile) {
        $permissions = Get-Content -Path $permissionsFile | ConvertFrom-Json
    }
    else {
        Write-EnhancedLog -Message "Permissions file not found: $permissionsFile" -Level "ERROR"
        throw "Permissions file not found: $permissionsFile"
    }

    # Retrieve the existing application (optional, uncomment if needed)
    # $app = Get-MgApplication -ApplicationId $appId

    # Prepare the required resource access
    $requiredResourceAccess = @(
        @{
            ResourceAppId = $resourceAppId
            ResourceAccess = $permissions
        }
    )

    # Update the application
    try {
        Update-MgApplication -ApplicationId $appId -RequiredResourceAccess $requiredResourceAccess
        Write-EnhancedLog -Message "Successfully updated application permissions for appId: $appId" -Level "INFO"
    }
    catch {
        Write-EnhancedLog -Message "Failed to update application permissions for appId: $appId. Error: $_" -Level "ERROR"
        throw $_
    }
}


# Associate certificate with App Registration
function Add-KeyCredentialToApp {
    param (
        [Parameter(Mandatory = $true)]
        [string]$AppId,

        [Parameter(Mandatory = $true)]
        [string]$CertPath
    )

    # Read the certificate file using the constructor
    $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($CertPath)
    $certBytes = $cert.RawData
    $base64Cert = [System.Convert]::ToBase64String($certBytes)

    # Convert certificate dates to DateTime and adjust for time zone
    $startDate = [datetime]::Parse($cert.NotBefore.ToString("o"))
    $endDate = [datetime]::Parse($cert.NotAfter.ToString("o"))

    # Adjust the start and end dates to ensure they are valid and in UTC
    $startDate = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId($startDate, [System.TimeZoneInfo]::Local.Id, 'UTC')
    $endDate = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId($endDate, [System.TimeZoneInfo]::Local.Id, 'UTC')

    # Adjust end date by subtracting one day to avoid potential end date issues
    $endDate = $endDate.AddDays(-1)

    # Prepare the key credential parameters
    $keyCredentialParams = @{
        CustomKeyIdentifier = [System.Convert]::FromBase64String([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($cert.Thumbprint.Substring(0, 32))))
        DisplayName = "GraphCert"
        EndDateTime = $endDate
        StartDateTime = $startDate
        KeyId = [Guid]::NewGuid().ToString()
        Type = "AsymmetricX509Cert"
        Usage = "Verify"
        Key = $certBytes
    }

    # Create the key credential object
    $keyCredential = [Microsoft.Graph.PowerShell.Models.MicrosoftGraphKeyCredential]::new()
    $keyCredential.CustomKeyIdentifier = $keyCredentialParams.CustomKeyIdentifier
    $keyCredential.DisplayName = $keyCredentialParams.DisplayName
    $keyCredential.EndDateTime = $keyCredentialParams.EndDateTime
    $keyCredential.StartDateTime = $keyCredentialParams.StartDateTime
    $keyCredential.KeyId = $keyCredentialParams.KeyId
    $keyCredential.Type = $keyCredentialParams.Type
    $keyCredential.Usage = $keyCredentialParams.Usage
    $keyCredential.Key = $keyCredentialParams.Key

    # Update the application with the new key credential
    try {
        Update-MgApplication -ApplicationId $AppId -KeyCredentials @($keyCredential)
        Write-Host "Key credential added successfully to the application."
    } catch {
        Write-Host "An error occurred: $_"
    }
}




function Connect-GraphWithCert {
    param (
        [Parameter(Mandatory = $true)]
        [string]$tenantId,
        [Parameter(Mandatory = $true)]
        [string]$clientId,
        [Parameter(Mandatory = $true)]
        [string]$certPath,
        [Parameter(Mandatory = $true)]
        [string]$certPassword
    )

    # Log the certificate path
    Log-Params -Params @{certPath = $certPath}

    # Load the certificate from the PFX file
    $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($certPath, $certPassword)

    # Define the splat for Connect-MgGraph
    $GraphParams = @{
        ClientId    = $clientId
        TenantId    = $tenantId
        Certificate = $cert
    }

    # Log the parameters
    Log-Params -Params $GraphParams

    # Obtain access token (if needed separately)
    $accessToken = Get-MsGraphAccessTokenCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword
    Log-Params -Params @{accessToken = $accessToken}

    # Connect to Microsoft Graph
    Write-EnhancedLog -message 'Calling Connect-MgGraph with client certificate file path and password' -Level 'INFO'
    Connect-MgGraph @GraphParams -NoWelcome

    # # Example command after connection (optional)
    # $organization = Get-MgOrganization
    # Write-Output $organization





    
  # # Graph Connect 

        # Define the parameters for non-interactive connection

        $IntuneGraphconnectionParams = @{
            clientId     = $clientId
            tenantID     = $tenantId
            # ClientSecret = $secrets.ClientSecret
            Clientcert = $cert
        }

    # Call the Connect-MSIntuneGraph function with splatted parameters
    Write-EnhancedLog -Message "calling Connect-MSIntuneGraph with connectionParams " -Level "WARNING"

    $Session = Connect-MSIntuneGraph @IntuneGraphconnectionParams

    Write-EnhancedLog -Message "connecting to Graph using Connect-MSIntuneGraph - done" -Level "INFO"



    return $accessToken
}

# # Example usage
# $clientId = '8230c33e-ff30-419c-a1fc-4caf98f069c9'
# $tenantId = 'b5dae566-ad8f-44e1-9929-5669f1dbb343'
# $certPath = Join-Path -Path $PSScriptRoot -ChildPath 'graphcert.pfx'
# $certPassword = "somepassword"

# Connect-GraphWithCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword


function Convert-EntraDeviceIdToIntuneDeviceId {
    param (
        [Parameter(Mandatory = $true)]
        [string]$entraDeviceId,
        [hashtable]$headers
    )

    Write-EnhancedLog -Message "Converting Entra Device ID: $entraDeviceId to Intune Device ID" -Level "INFO"

    try {
        # Construct the Graph API URL to retrieve device details
        $graphApiUrl = "https://graph.microsoft.com/v1.0/deviceManagement/managedDevices?`$filter=azureADDeviceId eq '$entraDeviceId'"
        Write-Output "Constructed Graph API URL: $graphApiUrl"

        # Send the request
        $response = Invoke-WebRequest -Uri $graphApiUrl -Headers $headers -Method Get
        $data = ($response.Content | ConvertFrom-Json).value

        if ($data -and $data.Count -gt 0) {
            $intuneDeviceId = $data[0].id
            Write-EnhancedLog -Message "Converted Entra Device ID: $entraDeviceId to Intune Device ID: $intuneDeviceId" -Level "INFO"
            return $intuneDeviceId
        } else {
            Write-EnhancedLog -Message "No Intune Device found for Entra Device ID: $entraDeviceId" -Level "WARN"
            return $null
        }
    } catch {
        Write-EnhancedLog -Message "Error converting Entra Device ID to Intune Device ID: $_" -Level "ERROR"
        return $null
    }
}

# # Example usage
# $headers = @{ Authorization = "Bearer your-access-token" }
# $entraDeviceId = "73e94a92-fc5a-45b6-bf6c-90ce8a353c44"

# $intuneDeviceId = Convert-EntraDeviceIdToIntuneDeviceId -entraDeviceId $entraDeviceId -Headers $headers
# Write-Output "Intune Device ID: $intuneDeviceId"


function Convert-WindowsPathToLinuxPath {
    <#
.SYNOPSIS
    Converts a Windows file path to a Linux file path.

.DESCRIPTION
    This function takes a Windows file path as input and converts it to a Linux file path.
    It replaces backslashes with forward slashes and handles the drive letter.

.PARAMETER WindowsPath
    The full file path in Windows format that needs to be converted.

.EXAMPLE
    PS> Convert-WindowsPathToLinuxPath -WindowsPath 'C:\Code\CB\Entra\ARH\Get-EntraConnectSyncErrorsfromEntra copy.ps1'
    Returns '/mnt/c/Code/CB/Entra/ARH/Get-EntraConnectSyncErrorsfromEntra copy.ps1'

#>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    Begin {
        Write-Host "Starting the path conversion process..."
    }

    Process {
        try {
            Write-Host "Input Windows Path: $WindowsPath"
            
            # Replace backslashes with forward slashes
            $linuxPath = $WindowsPath -replace '\\', '/'

            # Handle drive letter by converting "C:" to "/mnt/c"
            if ($linuxPath -match '^[A-Za-z]:') {
                $driveLetter = $linuxPath.Substring(0, 1).ToLower()
                $linuxPath = "/mnt/$driveLetter" + $linuxPath.Substring(2)
            }

            Write-Host "Converted Linux Path: $linuxPath"
            return $linuxPath
        }
        catch {
            Write-Host "Error during conversion: $_"
            throw
        }
    }

    End {
        Write-Host "Path conversion completed."
    }
}

# # Example usage
# $windowsPath = 'C:\Code\Unified365toolbox\Graph\graphcert.pfx'
# $linuxPath = Convert-WindowsPathToLinuxPath -WindowsPath $windowsPath
# Write-Host "Linux path: $linuxPath"


function Create-SelfSignedCert {
    param (
        [string]$CertName,
        [string]$CertStoreLocation = "Cert:\CurrentUser\My",
        [string]$TenantName,
        [string]$AppId
    )

    $cert = New-SelfSignedCertificate -CertStoreLocation $CertStoreLocation `
        -Subject "CN=$CertName, O=$TenantName, OU=$AppId" `
        -KeyLength 2048 `
        -NotAfter (Get-Date).AddDays(30)

    if ($null -eq $cert) {
        Write-EnhancedLog -Message "Failed to create certificate" -Level "ERROR"
        throw "Certificate creation failed"
    }
    Write-EnhancedLog -Message "Certificate created successfully" -Level "INFO"
    return $cert
}


    # Define the function
    function ExportCertificatetoFile {
        param (
            [Parameter(Mandatory = $true)]
            [string]$CertThumbprint,

            [Parameter(Mandatory = $true)]
            [string]$ExportDirectory
        )

        try {
            # Get the certificate from the current user's personal store
            $cert = Get-Item -Path "Cert:\CurrentUser\My\$CertThumbprint"
        
            # Ensure the export directory exists
            if (-not (Test-Path -Path $ExportDirectory)) {
                New-Item -ItemType Directory -Path $ExportDirectory -Force
            }

            # Dynamically create a file name using the certificate subject name and current timestamp
            $timestamp = (Get-Date).ToString("yyyyMMddHHmmss")
            $subjectName = $cert.SubjectName.Name -replace "[^a-zA-Z0-9]", "_"
            $fileName = "${subjectName}_$timestamp"

            # Set the export file path
            $certPath = Join-Path -Path $ExportDirectory -ChildPath "$fileName.cer"
        
            # Export the certificate to a file (DER encoded binary format with .cer extension)
            $cert | Export-Certificate -FilePath $certPath -Type CERT -Force | Out-Null

            # Output the export file path
            Write-EnhancedLog -Message "Certificate exported to: $certPath"

            # Return the export file path
            return $certPath
        }
        catch {
            Write-Host "Failed to export certificate: $_"
        }
    }


function Get-AppInfoFromJson {
    param (
        [Parameter(Mandatory = $true)]
        [string]$jsonPath
    )

    # Check if the file exists
    if (-Not (Test-Path -Path $jsonPath)) {
        Write-Error "The file at path '$jsonPath' does not exist."
        return
    }

    # Read the JSON content from the file
    $jsonContent = Get-Content -Path $jsonPath -Raw

    # Convert the JSON content to a PowerShell object
    $appData = ConvertFrom-Json -InputObject $jsonContent

    # Extract the required information
    $extractedData = $appData | ForEach-Object {
        [PSCustomObject]@{
            Id              = $_.Id
            DisplayName     = $_.DisplayName
            AppId           = $_.AppId
            SignInAudience  = $_.SignInAudience
            PublisherDomain = $_.PublisherDomain
        }
    }

    # Return the extracted data
    return $extractedData
}


# Function to read the application name from app.json and append a timestamp
function Get-AppName {
    param (
        [string]$AppJsonFile
    )

    if (-Not (Test-Path $AppJsonFile)) {
        Write-EnhancedLog -Message "App JSON file not found: $AppJsonFile" -Level "ERROR"
        throw "App JSON file missing"
    }

    $appConfig = Get-Content -Path $AppJsonFile | ConvertFrom-Json
    $baseAppName = $appConfig.AppName
    $timestamp = (Get-Date).ToString("yyyyMMddHHmmss")
    $uniqueAppName = "$baseAppName-$timestamp"

    Write-EnhancedLog -Message "Generated unique app name: $uniqueAppName" -Level "INFO"
    return $uniqueAppName
}


#need to the test the following first

function Get-FriendlyNamesForPermissions {
    param (
        [string]$tenantId,
        [string]$clientId,
        [string]$clientSecret,
        [string]$permissionsFile
    )

    # Function to get access token
    function Get-MsGraphAccessToken {
        param (
            [string]$tenantId,
            [string]$clientId,
            [string]$clientSecret
        )

        $body = @{
            grant_type    = "client_credentials"
            client_id     = $clientId
            client_secret = $clientSecret
            scope         = "https://graph.microsoft.com/.default"
        }

        $response = Invoke-RestMethod -Method Post -Uri "https://login.microsoftonline.com/$tenantId/oauth2/v2.0/token" -ContentType "application/x-www-form-urlencoded" -Body $body
        return $response.access_token
    }

    # Load permissions from the JSON file
    if (Test-Path -Path $permissionsFile) {
        $permissions = Get-Content -Path $permissionsFile | ConvertFrom-Json
    }
    else {
        Write-Error "Permissions file not found: $permissionsFile"
        throw "Permissions file not found: $permissionsFile"
    }

    # Get access token
    $accessToken = Get-MsGraphAccessToken -tenantId $tenantId -clientId $clientId -clientSecret $clientSecret

    # Create header for Graph API requests
    $headers = @{
        Authorization = "Bearer $accessToken"
    }

    # Translate IDs to friendly names
    foreach ($permission in $permissions) {
        $id = $permission.Id
        $url = "https://graph.microsoft.com/v1.0/servicePrincipals?$filter=appRoles/id eq '$id' or oauth2PermissionScopes/id eq '$id'&$select=displayName"
        $response = Invoke-RestMethod -Method Get -Uri $url -Headers $headers
        $friendlyName = $response.value[0].displayName
        $permission | Add-Member -MemberType NoteProperty -Name FriendlyName -Value $friendlyName
    }

    return $permissions
}

# # Example usage
# $tenantId = "your-tenant-id"
# $clientId = "your-client-id"
# $clientSecret = "your-client-secret"
# $permissionsFilePath = Join-Path -Path $PSScriptRoot -ChildPath "permissions.json"

# $friendlyPermissions = Get-FriendlyNamesForPermissions -tenantId $tenantId -clientId $clientId -clientSecret $clientSecret -permissionsFile $permissionsFilePath
# $friendlyPermissions | Format-Table -AutoSize


function Get-MsGraphAccessToken {
    param (
        [string]$tenantId,
        [string]$clientId,
        [string]$clientSecret
    )

    $tokenEndpoint = "https://login.microsoftonline.com/$tenantId/oauth2/v2.0/token"
    $body = @{
        client_id     = $clientId
        scope         = "https://graph.microsoft.com/.default"
        client_secret = $clientSecret
        grant_type    = "client_credentials"
    }

    $httpClient = New-Object System.Net.Http.HttpClient
    $bodyString = ($body.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join '&'

    try {
        $content = New-Object System.Net.Http.StringContent($bodyString, [System.Text.Encoding]::UTF8, "application/x-www-form-urlencoded")
        $response = $httpClient.PostAsync($tokenEndpoint, $content).Result

        if (-not $response.IsSuccessStatusCode) {
            Write-EnhancedLog -Message "HTTP request failed with status code: $($response.StatusCode)" -Level "ERROR"
            return $null
        }

        $responseContent = $response.Content.ReadAsStringAsync().Result
        $accessToken = (ConvertFrom-Json $responseContent).access_token

        if ($accessToken) {
            Write-EnhancedLog -Message "Access token retrieved successfully" -Level "INFO"
            return $accessToken
        }
        else {
            Write-EnhancedLog -Message "Failed to retrieve access token, response was successful but no token was found." -Level "ERROR"
            return $null
        }
    }
    catch {
        Write-EnhancedLog -Message "Failed to execute HTTP request or process results: $_" -Level "ERROR"
        return $null
    }
}


function Get-MsGraphAccessTokenCert {
    param (
        [Parameter(Mandatory = $true)]
        [string]$tenantId,
        [Parameter(Mandatory = $true)]
        [string]$clientId,
        [Parameter(Mandatory = $true)]
        [string]$certPath,
        [Parameter(Mandatory = $true)]
        [string]$certPassword
    )

    $tokenEndpoint = "https://login.microsoftonline.com/$tenantId/oauth2/v2.0/token"

    # Load the certificate
    $cert = Load-Certificate -certPath $certPath -certPassword $certPassword

    # Create JWT header
    $jwtHeader = @{
        alg = "RS256"
        typ = "JWT"
        x5t = [Convert]::ToBase64String($cert.GetCertHash())
    }

    $now = [System.DateTime]::UtcNow
    Write-EnhancedLog -Message "Current UTC Time: $now"

    # Get nbf and exp times
    $nbfTime = Get-UnixTime -offsetMinutes -5  # nbf is 5 minutes ago
    $expTime = Get-UnixTime -offsetMinutes 55  # exp is 55 minutes from now

    Write-EnhancedLog -Message "nbf (not before) time: $nbfTime"
    Write-EnhancedLog -Message "exp (expiration) time: $expTime"

    # Create JWT payload
    $jwtPayload = @{
        aud = $tokenEndpoint
        exp = $expTime
        iss = $clientId
        jti = [guid]::NewGuid().ToString()
        nbf = $nbfTime
        sub = $clientId
    }

    Write-EnhancedLog -Message "JWT Payload: $(ConvertTo-Json $jwtPayload -Compress)"

    # Generate JWT assertion
    $clientAssertion = Generate-JWTAssertion -jwtHeader $jwtHeader -jwtPayload $jwtPayload -cert $cert

    # Send token request
    return Send-TokenRequest -tokenEndpoint $tokenEndpoint -clientId $clientId -clientAssertion $clientAssertion
}


# # Example usage of Get-MsGraphAccessTokenCert
# $tenantId = "b5dae566-ad8f-44e1-9929-5669f1dbb343"
# $clientId = "8230c33e-ff30-419c-a1fc-4caf98f069c9"
# $certPath = "C:\Code\appgallery\Intune-Win32-Deployer\apps-winget-repo\PR4B_ExportVPNtoSPO-v1\PR4B-ExportVPNtoSPO-v2\graphcert.pfx"
# $certPassword = "somepassword"
# $accessToken = Get-MsGraphAccessTokenCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword
# Write-Host "Access Token: $accessToken"


function Get-Secrets {
    <#
.SYNOPSIS
Loads secrets from a JSON file.

.DESCRIPTION
This function reads a JSON file containing secrets and returns an object with these secrets.

.PARAMETER SecretsPath
The path to the JSON file containing secrets. If not provided, the default is "secrets.json" in the same directory as the script.

.EXAMPLE
$secrets = Get-Secrets -SecretsPath "C:\Path\To\secrets.json"

This example loads secrets from the specified JSON file.

.NOTES
If the SecretsPath parameter is not provided, the function assumes the JSON file is named "secrets.json" and is located in the same directory as the script.
#>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $false)]
        # [string]$SecretsPath = (Join-Path -Path $PSScriptRoot -ChildPath "secrets.json")
        [string]$SecretsPath
    )

    try {
        Write-EnhancedLog -Message "Attempting to load secrets from path: $SecretsPath" -Level "INFO"

        # Check if the secrets file exists
        if (-not (Test-Path -Path $SecretsPath)) {
            Write-EnhancedLog -Message "Secrets file not found at path: $SecretsPath" -Level "ERROR"
            throw "Secrets file not found at path: $SecretsPath"
        }

        # Load and parse the secrets file
        $secrets = Get-Content -Path $SecretsPath -Raw | ConvertFrom-Json
        Write-EnhancedLog -Message "Successfully loaded secrets from path: $SecretsPath" -Level "INFO"
        
        return $secrets
    }
    catch {
        Write-EnhancedLog -Message "Error loading secrets from path: $SecretsPath. Error: $_" -Level "ERROR"
        throw $_
    }
}


function Get-SignInLogs {
    param (
        [string]$url,
        [hashtable]$headers
    )

    $allLogs = @()

    while ($url) {
        try {
            Write-EnhancedLog -Message "Requesting URL: $url" -Level "INFO"
            # Make the API request
            $response = Invoke-WebRequest -Uri $url -Headers $headers -Method Get
            $data = ($response.Content | ConvertFrom-Json)

            # Collect the logs
            $allLogs += $data.value

            # Check for pagination
            $url = $data.'@odata.nextLink'
        } catch {
            Write-EnhancedLog -Message "Error: $($_.Exception.Message)" -Level "ERROR"
            break
        }
    }

    return $allLogs
}


function Get-TenantDetails {
    # Retrieve the organization details
    $organization = Get-MgOrganization

    # Extract the required details
    $tenantName = $organization.DisplayName
    $tenantId = $organization.Id
    $tenantDomain = $organization.VerifiedDomains[0].Name

    # Output tenant summary
    Write-EnhancedLog -Message "Tenant Name: $tenantName" -Level "INFO"
    Write-EnhancedLog -Message "Tenant ID: $tenantId" -Level "INFO"
    Write-EnhancedLog -Message "Tenant Domain: $tenantDomain" -Level "INFO"
}

# Example usage
# Get-TenantDetails


function Invoke-GraphRequest {
    param (
        [string]$Method,
        [string]$Uri,
        [string]$AccessToken,
        [string]$Body = $null
    )

    $httpClient = New-Object System.Net.Http.HttpClient
    $httpClient.DefaultRequestHeaders.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", $AccessToken)

    try {
        if ($Body) {
            $content = New-Object System.Net.Http.StringContent($Body, [System.Text.Encoding]::UTF8, "application/json")
        }

        $response = $null
        switch ($Method) {
            "GET" { $response = $httpClient.GetAsync($Uri).Result }
            "POST" { $response = $httpClient.PostAsync($Uri, $content).Result }
            "PATCH" { $response = $httpClient.PatchAsync($Uri, $content).Result }
            "DELETE" { $response = $httpClient.DeleteAsync($Uri).Result }
        }

        if (-not $response) {
            throw "No response received from the server."
        }

        $responseContent = $response.Content.ReadAsStringAsync().Result
        $responseStatus = $response.IsSuccessStatusCode

        # Define the directory for response JSON files
        $responseDir = Join-Path -Path $PSScriptRoot -ChildPath "responses"
        if (-not (Test-Path -Path $responseDir)) {
            New-Item -ItemType Directory -Path $responseDir
        }

        # Log the full request and response in JSON format
        $logEntry = @{
            RequestUri    = $Uri
            RequestMethod = $Method
            RequestBody   = $Body
            Response      = $responseContent
            IsSuccess     = $responseStatus
            TimeStamp     = Get-Date -Format "yyyyMMddHHmmssfff"
        }

        $logFile = Join-Path -Path $responseDir -ChildPath ("Response_$($logEntry.TimeStamp).json")
        $logEntry | ConvertTo-Json | Set-Content -Path $logFile

        Write-EnhancedLog -Message "Response logged to $logFile" -Level "INFO"

        if ($response.IsSuccessStatusCode) {
            Write-EnhancedLog -Message "Successfully executed $Method request to $Uri." -Level "INFO"
            return $responseContent
        }
        else {
            $errorContent = $responseContent
            Write-EnhancedLog -Message "HTTP request failed with status code: $($response.StatusCode). Error content: $errorContent" -Level "ERROR"
            return $null
        }
    }
    catch {
        Write-EnhancedLog -Message "Failed to execute $Method request to $Uri $_" -Level "ERROR"
        return $null
    }
    finally {
        $httpClient.Dispose()
    }
}


function Load-Certificate {
    param (
        [Parameter(Mandatory = $true)]
        [string]$CertPath,

        [Parameter(Mandatory = $true)]
        [string]$CertPassword
    )

    try {
        Write-EnhancedLog -Message "Attempting to load certificate from path: $CertPath" -Level "INFO"

        # Validate certificate path before loading
        $certExistsBefore = Validate-Certificate -CertPath $CertPath
        if (-not $certExistsBefore) {
            throw "Certificate path does not exist: $CertPath"
        }

        # Check the OS and convert the certificate path if running on Linux
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            if ($PSVersionTable.Platform -eq 'Unix') {
                $CertPath = Convert-WindowsPathToLinuxPath -WindowsPath $CertPath
            }
        } else {
            $os = [System.Environment]::OSVersion.Platform
            if ($os -eq [System.PlatformID]::Unix) {
                $CertPath = Convert-WindowsPathToLinuxPath -WindowsPath $CertPath
            }
        }

        # Load the certificate directly from the file
        $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($CertPath, $CertPassword)
        
        Write-EnhancedLog -Message "Successfully loaded certificate from path: $CertPath" -Level "INFO"

        # Validate certificate path after loading
        $certExistsAfter = Validate-Certificate -CertPath $CertPath
        if ($certExistsAfter) {
            Write-EnhancedLog -Message "Certificate path still exists after loading: $CertPath" -Level "INFO"
        } else {
            Write-EnhancedLog -Message "Certificate path does not exist after loading: $CertPath" -Level "WARNING"
        }

        return $cert
    }
    catch {
        Write-EnhancedLog -Message "Error loading certificate from path: $CertPath. Error: $_" -Level "ERROR"
        throw $_
    }
}



function Log-Params {
    param (
        [hashtable]$Params
    )

    foreach ($key in $Params.Keys) {
        Write-EnhancedLog "$key $($Params[$key])" -Color Yellow
    }
}


# Output secrets to console and file
function Output-Secrets {
    param (
        [string]$AppDisplayName,
        [string]$ApplicationID,
        [string]$Thumbprint,
        [string]$TenantID,
        [string]$SecretsFile = "$PSScriptRoot/secrets.json"
    )

    $secrets = @{
        AppDisplayName         = $AppDisplayName
        ApplicationID_ClientID = $ApplicationID
        Thumbprint             = $Thumbprint
        TenantID               = $TenantID
    }

    $secrets | ConvertTo-Json | Set-Content -Path $SecretsFile

    Write-Host "================ Secrets ================"
    Write-Host "`$AppDisplayName        = $($AppDisplayName)"
    Write-Host "`$ApplicationID_ClientID          = $($ApplicationID)"
    Write-Host "`$Thumbprint     = $($Thumbprint)"
    Write-Host "`$TenantID        = $TenantID"
    Write-Host "================ Secrets ================"
    Write-Host "    SAVE THESE IN A SECURE LOCATION     "
}


function Remove-AppListJson {
    param (
        [Parameter(Mandatory = $true)]
        [string]$jsonPath
    )

    # Check if the file exists
    if (Test-Path -Path $jsonPath) {
        try {
            # Remove the file
            Remove-Item -Path $jsonPath -Force
            Write-EnhancedLog -Message "The applist.json file has been removed successfully."
        }
        catch {
            Write-EnhancedLog -Message "An error occurred while removing the file: $_"
            throw $_
        }
    }
    else {
        Write-EnhancedLog -Message "The file at path '$jsonPath' does not exist."
    }
}


# # Import the module
# Import-Module Microsoft.Graph.Applications

# # Connect to Microsoft Graph
# Connect-MgGraph -Scopes "Application.ReadWrite.All"

# # Get and remove applications starting with 'GraphApp-Test001'
# Get-MgApplication -Filter "startswith(displayName, 'GraphApp-Test001')" | ForEach-Object {
#     Remove-MgApplication -ApplicationId $_.Id -Confirm:$false
# }

# # Disconnect the session
# Disconnect-MgGraph



function Run-DumpAppListToJSON {
    param (
        [string]$JsonPath
    )

    $scriptContent = @"
function Dump-AppListToJSON {
    param (
        [string]`$JsonPath
    )


    Disconnect-MgGraph

    # Connect to Graph interactively
    Connect-MgGraph -Scopes 'Application.ReadWrite.All'

    # Retrieve all application objects
    `$allApps = Get-MgApplication

    # Export to JSON
    `$allApps | ConvertTo-Json -Depth 10 | Out-File -FilePath `$JsonPath
}

# Dump application list to JSON
Dump-AppListToJSON -JsonPath `"$JsonPath`"
"@

    # Write the script content to a temporary file
    $tempScriptPath = [System.IO.Path]::Combine($PSScriptRoot, "DumpAppListTemp.ps1")
    Set-Content -Path $tempScriptPath -Value $scriptContent

    # Start a new PowerShell session to run the script and wait for it to complete
    $process = Start-Process pwsh -ArgumentList "-NoProfile", "-NoLogo", "-File", $tempScriptPath -PassThru
    $process.WaitForExit()

    # Remove the temporary script file after execution
    Remove-Item -Path $tempScriptPath
}


function Update-ApplicationPermissions {
    param (
        [string]$appId,
        [string]$permissionsFile
    )

    $resourceAppId = "00000003-0000-0000-c000-000000000000"  # Microsoft Graph

    # Load permissions from the JSON file
    if (Test-Path -Path $permissionsFile) {
        $permissions = Get-Content -Path $permissionsFile | ConvertFrom-Json
    }
    else {
        Write-EnhancedLog -Message "Permissions file not found: $permissionsFile" -Level "ERROR"
        throw "Permissions file not found: $permissionsFile"
    }

    # Retrieve the existing application (optional, uncomment if needed)
    # $app = Get-MgApplication -ApplicationId $appId

    # Prepare the required resource access
    $requiredResourceAccess = @(
        @{
            ResourceAppId = $resourceAppId
            ResourceAccess = $permissions
        }
    )

    # Update the application
    try {
        Update-MgApplication -ApplicationId $appId -RequiredResourceAccess $requiredResourceAccess
        Write-EnhancedLog -Message "Successfully updated application permissions for appId: $appId" -Level "INFO"
    }
    catch {
        Write-EnhancedLog -Message "Failed to update application permissions for appId: $appId. Error: $_" -Level "ERROR"
        throw $_
    }
}


function Validate-AppCreation {
    param (
        [string]$AppName,
        [string]$JsonPath
    )

    # Call the function to run the script in its own instance of pwsh
    

    # Example usage
    # $jsonPath = "C:\path\to\your\jsonfile.json"
    # $appInfo = Get-AppInfoFromJson -jsonPath $jsonPath

   
    write-enhancedlog -Message "validating AppName $AppName from $JsonPath"

    try {
        # Import application objects from JSON using Get-AppInfoFromJson function
        $allApps = Get-AppInfoFromJson -jsonPath $JsonPath

         # Output the extracted data
        # $allApps | Format-Table -AutoSize

        # # List all applications
        # write-enhancedlog -Message "Listing all applications:"
        # $allApps | Format-Table Id, DisplayName, AppId, SignInAudience, PublisherDomain -AutoSize

        # Filter the applications to find the one with the specified display name
        $app = $allApps | Where-Object { $_.DisplayName -eq $AppName }

        # Debug output
        # write-enhancedlog -Message "Filtered applications count: $($app.Count)"
        if ($app.Count -eq 0) {
            write-enhancedlog -Message "No applications found with the name $AppName"
        }
        else {
            # write-enhancedlog -Message "Filtered applications details:"
            # $app | Format-Table Id, DisplayName, AppId, SignInAudience, PublisherDomain -AutoSize
        }

        # Log the parameters and the retrieved application object
        $params = @{
            AppName    = $AppName
            AppCount   = ($app | Measure-Object).Count
            AppDetails = $app
        }
        Log-Params -Params $params

        # Check if the application object is not null and has items
        if ($null -ne $app -and ($app | Measure-Object).Count -gt 0) {
            write-enhancedlog -Message "Application found."
            return $true
        }
        write-enhancedlog -Message "Application not found."
        return $false
    }
    catch {
        write-enhancedlog -Message "An error occurred: $_"
        throw $_
    }
}


function Validate-AppCreationWithRetry {
    param (
        [Parameter(Mandatory = $true)]
        [string]$AppName,
        [Parameter(Mandatory = $true)]
        [string]$JsonPath
    )

    $maxDuration = 120  # Maximum duration in seconds (2 minutes)
    $interval = 2       # Interval in seconds
    $elapsed = 0        # Elapsed time counter

    while ($elapsed -lt $maxDuration) {
        try {
            # Validate the app creation
            Write-EnhancedLog -Message 'second validation'
            Remove-AppListJson -jsonPath $jsonPath
            # Start-Sleep -Seconds 30
            Run-DumpAppListToJSON -JsonPath $JsonPath
            $appExists = Validate-AppCreation -AppName $AppName -JsonPath $JsonPath
            if (-not $appExists) {
                Write-EnhancedLog -Message "App creation validation failed" -Level "ERROR"
                throw "App creation validation failed"
            }

            # If the app validation passes, exit the loop
            break
        }
        catch {
            Write-EnhancedLog -Message "An error occurred during app creation validation: $_" -Level "ERROR"
            Start-Sleep -Seconds $interval
            $elapsed += $interval
        }
    }

    if ($elapsed -ge $maxDuration) {
        Write-EnhancedLog -Message "App creation validation failed after multiple retries" -Level "ERROR"
        throw "App creation validation failed after multiple retries"
    }
}


function Validate-Certificate {
    param (
        [Parameter(Mandatory = $true)]
        [string]$CertPath
    )

    try {
        if (Test-Path -Path $CertPath) {
            Write-EnhancedLog -Message "Certificate path exists: $CertPath" -Level "INFO"
            return $true
        } else {
            Write-EnhancedLog -Message "Certificate path does not exist: $CertPath" -Level "WARNING"
            return $false
        }
    }
    catch {
        Write-EnhancedLog -Message "Error validating certificate path: $CertPath. Error: $_" -Level "ERROR"
        throw $_
    }
}


function Validate-UriAccess {
    param (
        [string]$uri,
        [hashtable]$headers
    )

    Write-EnhancedLog -Message "Validating access to URI: $uri" -Level "INFO"
    try {
        $response = Invoke-WebRequest -Uri $uri -Headers $headers -Method Get
        if ($response.StatusCode -eq 200) {
            Write-EnhancedLog -Message "Access to $uri PASS" -Level "INFO"
            return $true
        } else {
            Write-EnhancedLog -Message "Access to $uri FAIL" -Level "ERROR"
            return $false
        }
    } catch {
        Write-EnhancedLog -Message "Access to $uri FAIL - $_" -Level "ERROR"
        return $false
    }
}


#Unique Tracking ID: ff04d7f9-5cac-43a8-8602-c2d45228bcfa, Timestamp: 2024-03-20 12:25:26
# Assign values from JSON to variables
$LoggingDeploymentName = $config.LoggingDeploymentName
    
function Initialize-ScriptAndLogging {
    $ErrorActionPreference = 'SilentlyContinue'
    $deploymentName = "$LoggingDeploymentName" # Replace this with your actual deployment name
    $scriptPath_1001 = "C:\code\$deploymentName"
    # $hadError = $false
    
    try {
        if (-not (Test-Path -Path $scriptPath_1001)) {
            New-Item -ItemType Directory -Path $scriptPath_1001 -Force | Out-Null
            Write-Host "Created directory: $scriptPath_1001"
        }
    
        $computerName = $env:COMPUTERNAME
        $Filename = "$LoggingDeploymentName"
        $logDir = Join-Path -Path $scriptPath_1001 -ChildPath "exports\Logs\$computerName"
        $logPath = Join-Path -Path $logDir -ChildPath "$(Get-Date -Format 'yyyy-MM-dd-HH-mm-ss')"
            
        if (!(Test-Path $logPath)) {
            Write-Host "Did not find log file at $logPath"
            Write-Host "Creating log file at $logPath"
            $createdLogDir = New-Item -ItemType Directory -Path $logPath -Force -ErrorAction Stop
            Write-Host "Created log file at $logPath"
        }
            
        $logFile = Join-Path -Path $logPath -ChildPath "$Filename-Transcript.log"
        Start-Transcript -Path $logFile -ErrorAction Stop | Out-Null
    
        # $CSVDir_1001 = Join-Path -Path $scriptPath_1001 -ChildPath "exports\CSV"
        # $CSVFilePath_1001 = Join-Path -Path $CSVDir_1001 -ChildPath "$computerName"
            
        # if (!(Test-Path $CSVFilePath_1001)) {
        #     Write-Host "Did not find CSV file at $CSVFilePath_1001"
        #     Write-Host "Creating CSV file at $CSVFilePath_1001"
        #     $createdCSVDir = New-Item -ItemType Directory -Path $CSVFilePath_1001 -Force -ErrorAction Stop
        #     Write-Host "Created CSV file at $CSVFilePath_1001"
        # }
    
        return @{
            ScriptPath  = $scriptPath_1001
            Filename    = $Filename
            LogPath     = $logPath
            LogFile     = $logFile
            CSVFilePath = $CSVFilePath_1001
        }
    
    }
    catch {
        Write-Error "An error occurred while initializing script and logging: $_"
    }
}
# $initializationInfo = Initialize-ScriptAndLogging
    
    
    
# Script Execution and Variable Assignment
# After the function Initialize-ScriptAndLogging is called, its return values (in the form of a hashtable) are stored in the variable $initializationInfo.
    
# Then, individual elements of this hashtable are extracted into separate variables for ease of use:
    
# $ScriptPath_1001: The path of the script's main directory.
# $Filename: The base name used for log files.
# $logPath: The full path of the directory where logs are stored.
# $logFile: The full path of the transcript log file.
# $CSVFilePath_1001: The path of the directory where CSV files are stored.
# This structure allows the script to have a clear organization regarding where logs and other files are stored, making it easier to manage and maintain, especially for logging purposes. It also encapsulates the setup logic in a function, making the main script cleaner and more focused on its primary tasks.
    
    
# $ScriptPath_1001 = $initializationInfo['ScriptPath']
# $Filename = $initializationInfo['Filename']
# $logPath = $initializationInfo['LogPath']
# $logFile = $initializationInfo['LogFile']
# $CSVFilePath_1001 = $initializationInfo['CSVFilePath']


#Unique Tracking ID: 737397f0-c74e-4087-9b99-279b520b7448, Timestamp: 2024-03-20 12:25:26
function AppendCSVLog {
    param (
        [string]$Message,
        [string]$CSVFilePath_1001
           
    )
    
    $csvData = [PSCustomObject]@{
        TimeStamp    = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
        ComputerName = $env:COMPUTERNAME
        Message      = $Message
    }
    
    $csvData | Export-Csv -Path $CSVFilePath_1001 -Append -NoTypeInformation -Force
}


#Unique Tracking ID: 4362313d-3c19-4d0c-933d-99438a6da297, Timestamp: 2024-03-20 12:25:26

function CreateEventSourceAndLog {
    param (
        [string]$LogName,
        [string]$EventSource
    )
    
    
    # Validate parameters
    if (-not $LogName) {
        Write-Warning "LogName is required."
        return
    }
    if (-not $EventSource) {
        Write-Warning "Source is required."
        return
    }
    
    # Function to create event log and source
    function CreateEventLogSource($logName, $EventSource) {
        try {
            if ($PSVersionTable.PSVersion.Major -lt 6) {
                New-EventLog -LogName $logName -Source $EventSource
            }
            else {
                [System.Diagnostics.EventLog]::CreateEventSource($EventSource, $logName)
            }
            Write-Host "Event source '$EventSource' created in log '$logName'"
        }
        catch {
            Write-Warning "Error creating the event log. Make sure you run PowerShell as an Administrator."
        }
    }
    
    # Check if the event log exists
    if (-not (Get-WinEvent -ListLog $LogName -ErrorAction SilentlyContinue)) {
        # CreateEventLogSource $LogName $EventSource
    }
    # Check if the event source exists
    elseif (-not ([System.Diagnostics.EventLog]::SourceExists($EventSource))) {
        # Unregister the source if it's registered with a different log
        $existingLogName = (Get-WinEvent -ListLog * | Where-Object { $_.LogName -contains $EventSource }).LogName
        if ($existingLogName -ne $LogName) {
            Remove-EventLog -Source $EventSource -ErrorAction SilentlyContinue
        }
        # CreateEventLogSource $LogName $EventSource
    }
    else {
        Write-Host "Event source '$EventSource' already exists in log '$LogName'"
    }
}
    
# $LogName = (Get-Date -Format "HHmmss") + "_$LoggingDeploymentName"
# $EventSource = (Get-Date -Format "HHmmss") + "_$LoggingDeploymentName"
    
# Call the Create-EventSourceAndLog function
# CreateEventSourceAndLog -LogName $LogName -EventSource $EventSource
    
# Call the Write-CustomEventLog function with custom parameters and level
# Write-CustomEventLog -LogName $LogName -EventSource $EventSource -EventMessage "Outlook Signature Restore completed with warnings." -EventID 1001 -Level 'WARNING'


#Unique Tracking ID: c00ecaca-dd4b-4c7c-b80e-566b2f627e32, Timestamp: 2024-03-20 12:25:26
function Export-EventLog {
    param (
        [Parameter(Mandatory = $true)]
        [string]$LogName,
        [Parameter(Mandatory = $true)]
        [string]$ExportPath
    )
    
    try {
        wevtutil epl $LogName $ExportPath
    
        if (Test-Path $ExportPath) {
            Write-EnhancedLog -Message "Event log '$LogName' exported to '$ExportPath'" -Level "INFO"
        }
        else {
            Write-EnhancedLog -Message "Event log '$LogName' not exported: File does not exist at '$ExportPath'" -Level "WARNING"
        }
    }
    catch {
        Write-EnhancedLog -Message "Error exporting event log '$LogName': $($_.Exception.Message)" -Level "ERROR"
    }
}
    
# # Example usage
# $LogName = '$LoggingDeploymentNameLog'
# # $ExportPath = 'Path\to\your\exported\eventlog.evtx'
# $ExportPath = "C:\code\$LoggingDeploymentName\exports\Logs\$logname.evtx"
# Export-EventLog -LogName $LogName -ExportPath $ExportPath


#Unique Tracking ID: 132a90f9-6ba2-49cd-878b-279deadb8e22, Timestamp: 2024-03-20 12:25:26

function Write-EventLogMessage {
    param (
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,
    
        [string]$LogName = "$LoggingDeploymentName",
        [string]$EventSource,
    
        [int]$EventID = 1000  # Default event ID
    )
    
    $ErrorActionPreference = 'SilentlyContinue'
    $hadError = $false
    
    try {
        if (-not $EventSource) {
            throw "EventSource is required."
        }
    
        if ($PSVersionTable.PSVersion.Major -lt 6) {
            # PowerShell version is less than 6, use Write-EventLog
            Write-EventLog -LogName $logName -Source $EventSource -EntryType Information -EventId $EventID -Message $Message
        }
        else {
            # PowerShell version is 6 or greater, use System.Diagnostics.EventLog
            $eventLog = New-Object System.Diagnostics.EventLog($logName)
            $eventLog.Source = $EventSource
            $eventLog.WriteEntry($Message, [System.Diagnostics.EventLogEntryType]::Information, $EventID)
        }
    
        # Write-host "Event log entry created: $Message" 
    }
    catch {
        Write-host "Error creating event log entry: $_" 
        $hadError = $true
    }
    
    if (-not $hadError) {
        # Write-host "Event log message writing completed successfully."
    }
}
    


# param (
#     [string]$ExportFolderName = "IPV4Scan-v1"
#     # [string]$LogFileName = "exports-EnabledMemberDuplicatesExcludingGuests-v7Log"
# )

# # Global setup for paths


# $timestamp = Get-Date -Format "yyyyMMddHHmmss"
# $exportFolder = Join-Path -Path $PSScriptRoot -ChildPath $ExportFolderName_$timestamp
# # $logPath = Join-Path -Path $exportFolder -ChildPath "${LogFileName}_$timestamp.txt"

# # Ensure the exports folder and log file are ready
# if (-not (Test-Path -Path $exportFolder)) {
#     New-Item -ItemType Directory -Path $exportFolder | Out-Null
# }

# Define the Write-Log function
# function Write-Log {
#     param (
#         [Parameter(Mandatory = $true)]
#         [string]$Message,
#         [ConsoleColor]$Color = 'White'
#     )
#     Write-Host $Message -ForegroundColor $Color
#     Add-Content -Path $logPath -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message"
# }


# $configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
# $env:MYMODULE_CONFIG_PATH = $configPath


# <#
# .SYNOPSIS
# Dot-sources all PowerShell scripts in the 'private' folder relative to the script root.

# .DESCRIPTION
# This function finds all PowerShell (.ps1) scripts in a 'private' folder located in the script root directory and dot-sources them. It logs the process, including any errors encountered, with optional color coding.

# .EXAMPLE
# Dot-SourcePrivateScripts

# Dot-sources all scripts in the 'private' folder and logs the process.

# .NOTES
# Ensure the Write-EnhancedLog function is defined before using this function for logging purposes.
# #>

# function Get-PrivateScriptPathsAndVariables {
#     param (
#         [string]$BaseDirectory
#     )

#     try {
#         $privateFolderPath = Join-Path -Path $BaseDirectory -ChildPath "private"
    
#         if (-not (Test-Path -Path $privateFolderPath)) {
#             throw "Private folder path does not exist: $privateFolderPath"
#         }

#         # Construct and return a PSCustomObject
#         return [PSCustomObject]@{
#             BaseDirectory     = $BaseDirectory
#             PrivateFolderPath = $privateFolderPath
#         }
#     }
#     catch {
#         Write-Host "Error in finding private script files: $_"
#         # Optionally, you could return a PSCustomObject indicating an error state
#         # return [PSCustomObject]@{ Error = $_.Exception.Message }
#     }
# }



# # Retrieve script paths and related variables
# $DotSourcinginitializationInfo = Get-PrivateScriptPathsAndVariables -BaseDirectory $PSScriptRoot

# # $DotSourcinginitializationInfo
# $DotSourcinginitializationInfo | Format-List


# function Import-ModuleWithRetry {

#     <#
# .SYNOPSIS
# Imports a PowerShell module with retries on failure.

# .DESCRIPTION
# This function attempts to import a specified PowerShell module, retrying the import process up to a specified number of times upon failure. It waits for a specified delay between retries. The function uses advanced logging to provide detailed feedback about the import process.

# .PARAMETER ModulePath
# The path to the PowerShell module file (.psm1) that should be imported.

# .PARAMETER MaxRetries
# The maximum number of retries to attempt if importing the module fails. Default is 30.

# .PARAMETER WaitTimeSeconds
# The number of seconds to wait between retry attempts. Default is 2 seconds.

# .EXAMPLE
# $modulePath = "C:\Modules\MyPowerShellModule.psm1"
# Import-ModuleWithRetry -ModulePath $modulePath

# Tries to import the module located at "C:\Modules\MyPowerShellModule.psm1", with up to 30 retries, waiting 2 seconds between each retry.

# .NOTES
# This function requires the `Write-EnhancedLog` function to be defined in the script for logging purposes.

# .LINK
# Write-EnhancedLog

# #>

#     [CmdletBinding()]
#     param (
#         [Parameter(Mandatory)]
#         [string]$ModulePath,

#         [int]$MaxRetries = 30,

#         [int]$WaitTimeSeconds = 2
#     )

#     Begin {
#         $retryCount = 0
#         $isModuleLoaded = $false
#         # Write-EnhancedLog "Starting to import module from path: $ModulePath" -Level "INFO"
#         Write-host "Starting to import module from path: $ModulePath"
#     }

#     Process {
#         while (-not $isModuleLoaded -and $retryCount -lt $MaxRetries) {
#             try {
#                 Import-Module $ModulePath -ErrorAction Stop
#                 $isModuleLoaded = $true
#                 Write-EnhancedLog "Module $ModulePath imported successfully." -Level "INFO"
#             }
#             catch {
#                 # Write-EnhancedLog "Attempt $retryCount to load module failed. Waiting $WaitTimeSeconds seconds before retrying." -Level "WARNING"
#                 Write-host "Attempt $retryCount to load module failed. Waiting $WaitTimeSeconds seconds before retrying."
#                 Start-Sleep -Seconds $WaitTimeSeconds
#             }
#             finally {
#                 $retryCount++
#             }

#             if ($retryCount -eq $MaxRetries -and -not $isModuleLoaded) {
#                 # Write-EnhancedLog "Failed to import module after $MaxRetries retries." -Level "ERROR"
#                 Write-host "Failed to import module after $MaxRetries retries."
#                 break
#             }
#         }
#     }

#     End {
#         if ($isModuleLoaded) {
#             Write-EnhancedLog "Module $ModulePath loaded successfully." -Level "INFO"
#         }
#         else {
#             # Write-EnhancedLog "Failed to load module $ModulePath within the maximum retry limit." -Level "CRITICAL"
#             Write-host "Failed to load module $ModulePath within the maximum retry limit."
#         }
#     }
# }

# Example of how to use the function
# $PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
# $LoggingmodulePath = Join-Path -Path $PSScriptRoot -ChildPath "Private\EnhancedLoggingAO\2.0.0\EnhancedLoggingAO.psm1"
# $ModuleUpdatermodulePath = Join-Path -Path $PSScriptRoot -ChildPath "Private\EnhancedModuleUpdaterAO\1.0.0\EnhancedModuleUpdaterAO.psm1"

# Call the function to import the module with retry logic
# Import-ModuleWithRetry -ModulePath $LoggingmodulePath
# Import-ModuleWithRetry -ModulePath $ModuleUpdatermodulePath
# Import-ModuleWithRetry -ModulePath 'C:\Program Files (x86)\WindowsPowerShell\Modules\PowerShellGet\PSModule.psm1'
# Import-ModuleWithRetry -ModulePath 'C:\Program Files (x86)\WindowsPowerShell\Modules\PackageManagement\PackageProviderFunctions.psm1'

# function Install-MissingModules {
#     <#
# .SYNOPSIS
# Installs missing PowerShell modules from a given list of module names.

# .DESCRIPTION
# The Install-MissingModules function checks a list of PowerShell module names and installs any that are not already installed on the system. This function requires administrative privileges to install modules for all users.

# .PARAMETER RequiredModules
# An array of module names that you want to ensure are installed on the system.

# .EXAMPLE
# PS> $modules = @('ImportExcel', 'powershell-yaml')
# PS> Install-MissingModules -RequiredModules $modules

# This example checks for the presence of the 'ImportExcel' and 'powershell-yaml' modules and installs them if they are not already installed.
# #>

#     [CmdletBinding()]
#     param (
#         [Parameter(Mandatory = $true)]
#         [string[]]$RequiredModules
#     )

#     Begin {
#         Write-Verbose "Starting to check and install required modules..."
#     }

#     Process {
#         foreach ($module in $RequiredModules) {
#             if (-not (Get-Module -ListAvailable -Name $module)) {
#                 Write-Host "Module '$module' is not installed. Attempting to install..."
#                 try {
#                     Install-Module -Name $module -Force -Scope AllUsers
#                     Write-Host "Module '$module' installed successfully."
#                 }
#                 catch {
#                     Write-Error "Failed to install module '$module'. Error: $_"
#                 }
#             }
#             else {
#                 Write-Host "Module '$module' is already installed."
#             }
#         }
#     }

#     End {
#         Write-Verbose "Completed checking and installing modules."
#     }
# }

# # Example usage
# # $modules = @('ImportExcel', 'powershell-yaml' , 'PSWriteHTML')
# # Install-MissingModules -RequiredModules $modules -Verbose

function Export-Data {
    <#
.SYNOPSIS
Exports data to various formats including CSV, JSON, XML, HTML, PlainText, Excel, PDF, Markdown, and YAML.

.DESCRIPTION
The Export-Data function exports provided data to multiple file formats based on switches provided. It supports CSV, JSON, XML, GridView (for display only), HTML, PlainText, Excel, PDF, Markdown, and YAML formats. This function is designed to work with any PSObject.

.PARAMETER Data
The data to be exported. This parameter accepts input of type PSObject.

.PARAMETER BaseOutputPath
The base path for output files without file extension. This path is used to generate filenames for each export format.

.PARAMETER IncludeCSV
Switch to include CSV format in the export.

.PARAMETER IncludeJSON
Switch to include JSON format in the export.

.PARAMETER IncludeXML
Switch to include XML format in the export.

.PARAMETER IncludeGridView
Switch to display the data in a GridView.

.PARAMETER IncludeHTML
Switch to include HTML format in the export.

.PARAMETER IncludePlainText
Switch to include PlainText format in the export.

.PARAMETER IncludePDF
Switch to include PDF format in the export. Requires intermediate HTML to PDF conversion.

.PARAMETER IncludeExcel
Switch to include Excel format in the export.

.PARAMETER IncludeMarkdown
Switch to include Markdown format in the export. Custom or use a module if available.

.PARAMETER IncludeYAML
Switch to include YAML format in the export. Requires 'powershell-yaml' module.

.EXAMPLE
PS> $data = Get-Process | Select-Object -First 10
PS> Export-Data -Data $data -BaseOutputPath "C:\exports\mydata" -IncludeCSV -IncludeJSON

This example exports the first 10 processes to CSV and JSON formats.
#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [psobject]$Data,

        [Parameter(Mandatory = $true)]
        [string]$BaseOutputPath,

        [switch]$IncludeCSV,
        [switch]$IncludeJSON,
        [switch]$IncludeXML,
        [switch]$IncludeGridView,
        [switch]$IncludeHTML,
        [switch]$IncludePlainText,
        [switch]$IncludePDF, # Requires intermediate HTML to PDF conversion
        [switch]$IncludeExcel,
        [switch]$IncludeMarkdown, # Custom or use a module if available
        [switch]$IncludeYAML  # Requires 'powershell-yaml' module
    )

    Begin {




        # $modules = @('ImportExcel', 'powershell-yaml' , 'PSWriteHTML')
        # Install-MissingModules -RequiredModules $modules -Verbose


        # Setup the base path without extension
        Write-Host "BaseOutputPath before change: '$BaseOutputPath'"
        $basePathWithoutExtension = [System.IO.Path]::ChangeExtension($BaseOutputPath, $null)

        # Remove extension manually if it exists
        $basePathWithoutExtension = if ($BaseOutputPath -match '\.') {
            $BaseOutputPath.Substring(0, $BaseOutputPath.LastIndexOf('.'))
        }
        else {
            $BaseOutputPath
        }

        # Ensure no trailing periods
        $basePathWithoutExtension = $basePathWithoutExtension.TrimEnd('.')
    }

    Process {
        try {
            if ($IncludeCSV) {
                $csvPath = "$basePathWithoutExtension.csv"
                $Data | Export-Csv -Path $csvPath -NoTypeInformation
            }

            if ($IncludeJSON) {
                $jsonPath = "$basePathWithoutExtension.json"
                $Data | ConvertTo-Json -Depth 10 | Set-Content -Path $jsonPath
            }

            if ($IncludeXML) {
                $xmlPath = "$basePathWithoutExtension.xml"
                $Data | Export-Clixml -Path $xmlPath
            }

            if ($IncludeGridView) {
                $Data | Out-GridView -Title "Data Preview"
            }

            if ($IncludeHTML) {
                # Assumes $Data is the dataset you want to export to HTML
                # and $basePathWithoutExtension is prepared earlier in your script
                
                $htmlPath = "$basePathWithoutExtension.html"
                
                # Convert $Data to HTML using PSWriteHTML
                New-HTML -Title "Data Export Report" -FilePath $htmlPath -ShowHTML {
                    New-HTMLSection -HeaderText "Data Export Details" -Content {
                        New-HTMLTable -DataTable $Data -ScrollX -HideFooter
                    }
                }
            
                Write-Host "HTML report generated: '$htmlPath'"
            }
            

            if ($IncludePlainText) {
                $txtPath = "$basePathWithoutExtension.txt"
                $Data | Out-String | Set-Content -Path $txtPath
            }

            if ($IncludeExcel) {
                $excelPath = "$basePathWithoutExtension.xlsx"
                $Data | Export-Excel -Path $excelPath
            }

            # Assuming $Data holds the objects you want to serialize to YAML
            if ($IncludeYAML) {
                $yamlPath = "$basePathWithoutExtension.yaml"
    
                # Check if the powershell-yaml module is loaded
                if (Get-Module -ListAvailable -Name powershell-yaml) {
                    Import-Module powershell-yaml

                    # Process $Data to handle potentially problematic properties
                    $processedData = $Data | ForEach-Object {
                        $originalObject = $_
                        $properties = $_ | Get-Member -MemberType Properties
                        $clonedObject = New-Object -TypeName PSObject

                        foreach ($prop in $properties) {
                            try {
                                $clonedObject | Add-Member -MemberType NoteProperty -Name $prop.Name -Value $originalObject.$($prop.Name) -ErrorAction Stop
                            }
                            catch {
                                # Optionally handle or log the error. Skipping problematic property.
                                $clonedObject | Add-Member -MemberType NoteProperty -Name $prop.Name -Value "Error serializing property" -ErrorAction SilentlyContinue
                            }
                        }

                        return $clonedObject
                    }

                    # Convert the processed data to YAML and save it with UTF-16 LE encoding
                    $processedData | ConvertTo-Yaml | Set-Content -Path $yamlPath -Encoding Unicode
                    Write-Host "YAML export completed successfully: $yamlPath"
                }
                else {
                    Write-Warning "The 'powershell-yaml' module is not installed. YAML export skipped."
                }
            }

            if ($IncludeMarkdown) {
                # You'll need to implement or find a ConvertTo-Markdown function or use a suitable module
                $markdownPath = "$basePathWithoutExtension.md"
                $Data | ConvertTo-Markdown | Set-Content -Path $markdownPath
            }

            if ($IncludePDF) {
                # Convert HTML to PDF using external tool
                # This is a placeholder for the process. You will need to generate HTML first and then convert it.
                $pdfPath = "$basePathWithoutExtension.pdf"
                # Assuming you have a Convert-HtmlToPdf function or a similar mechanism
                $htmlPath = "$basePathWithoutExtension.html"
                $Data | ConvertTo-Html | Convert-HtmlToPdf -OutputPath $pdfPath
            }

        }
        catch {
            Write-Error "An error occurred during export: $_"
        }
    }

    End {
        Write-Verbose "Export-Data function execution completed."
    }
}


function Handle-Error {
    param (
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )
    
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        $fullErrorDetails = Get-Error -InputObject $ErrorRecord | Out-String
    } else {
        $fullErrorDetails = $ErrorRecord.Exception | Format-List * -Force | Out-String
    }

    Write-EnhancedLog -Message "Exception Message: $($ErrorRecord.Exception.Message)" -Level "ERROR"
    Write-EnhancedLog -Message "Full Exception: $fullErrorDetails" -Level "ERROR"
}

# # Example usage
# try {
#     # Your code that might throw an error
#     Throw "This is a test error"
# }
# catch {
#     Handle-Error -ErrorRecord $_
# }


#Unique Tracking ID: 275d6fc2-003c-4da0-9e66-16cfa045f901, Timestamp: 2024-03-20 12:25:26
# Read configuration from the JSON file
# $configPath = Join-Path -Path (Join-Path -Path $PSScriptRoot -ChildPath "..") -ChildPath "config.json"
# $config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

function Write-EnhancedLog {
    param (
        [string]$Message,
        [string]$Level = 'INFO',
        # [ConsoleColor]$ForegroundColor = [ConsoleColor]::White,
        # [string]$CSVFilePath_1001 = "$scriptPath_1001\exports\CSV\$(Get-Date -Format 'yyyy-MM-dd')-Log.csv",
        # [string]$CentralCSVFilePath = "$scriptPath_1001\exports\CSV\$Filename.csv",
        [switch]$UseModule = $false,
        [string]$Caller = (Get-PSCallStack)[0].Command
    )
    
    # Add timestamp, computer name, and log level to the message
    $formattedMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($env:COMPUTERNAME): [$Level] [$Caller] $Message"
    
    # Set foreground color based on log level
    # switch ($Level) {
    #     'DEBUG' { $ForegroundColor = [ConsoleColor]::Gray } # Added level
    #     'INFO' { $ForegroundColor = [ConsoleColor]::Green }
    #     'NOTICE' { $ForegroundColor = [ConsoleColor]::Cyan } # Added level
    #     'WARNING' { $ForegroundColor = [ConsoleColor]::Yellow }
    #     'ERROR' { $ForegroundColor = [ConsoleColor]::Red }
    #     'CRITICAL' { $ForegroundColor = [ConsoleColor]::Magenta } # Added level
    #     default { $ForegroundColor = [ConsoleColor]::White } # Default case for unknown levels




        
    # }
    
    # Write the message with the specified colors
    # $currentForegroundColor = $Host.UI.RawUI.ForegroundColor
    # $Host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Host $formattedMessage
    # $Host.UI.RawUI.ForegroundColor = $currentForegroundColor
    
    # Append to CSV file
    # AppendCSVLog -Message $formattedMessage -CSVFilePath $CSVFilePath_1001
    # AppendCSVLog -Message $formattedMessage -CSVFilePath $CentralCSVFilePath
    
    # Potential place for Write to event log (optional)
    # Depending on how you implement `Write-CustomEventLog`, you might want to handle it differently for various levels.


    # Write to event log (optional)
    # Write-CustomEventLog -EventMessage $formattedMessage -Level $Level

    
    # Adjust this line in your script where you call the function
    # Write-EventLogMessage -LogName $LogName -EventSource $EventSource -Message $formattedMessage -EventID 1001
}

# Note: Make sure the `AppendCSVLog` function is defined in your script or module.
# It should handle the CSV file appending logic.

    
#################################################################################################################################
################################################# END LOGGING ###################################################################
#################################################################################################################################



function Download-PsExec {
    [CmdletBinding()]
    param(
        # [string]$TargetFolder = "$PSScriptRoot\private"
        [string]$TargetFolder
    )

    Begin {

        Remove-ExistingPsExec -TargetFolder $TargetFolder
    }



    process {

        # Define the URL for PsExec download
        $url = "https://download.sysinternals.com/files/PSTools.zip"
    
        # Ensure the target folder exists
        if (-Not (Test-Path -Path $TargetFolder)) {
            New-Item -Path $TargetFolder -ItemType Directory
        }
  
        # Full path for the downloaded file
        $zipPath = Join-Path -Path $TargetFolder -ChildPath "PSTools.zip"
  
        try {
            # Download the PSTools.zip file containing PsExec
            Write-EnhancedLog -Message "Downloading PSTools.zip from: $url to: $zipPath"
            Invoke-WebRequest -Uri $url -OutFile $zipPath
  
            # Extract PsExec64.exe from the zip file
            Expand-Archive -Path $zipPath -DestinationPath "$TargetFolder\PStools" -Force
  
            # Specific extraction of PsExec64.exe
            $extractedFolderPath = Join-Path -Path $TargetFolder -ChildPath "PSTools"
            $PsExec64Path = Join-Path -Path $extractedFolderPath -ChildPath "PsExec64.exe"
            $finalPath = Join-Path -Path $TargetFolder -ChildPath "PsExec64.exe"
  
            # Move PsExec64.exe to the desired location
            if (Test-Path -Path $PsExec64Path) {
  
                Write-EnhancedLog -Message "Moving PSExec64.exe from: $PsExec64Path to: $finalPath"
                Move-Item -Path $PsExec64Path -Destination $finalPath
  
                # Remove the downloaded zip file and extracted folder
                Remove-Item -Path $zipPath -Force
                Remove-Item -Path $extractedFolderPath -Recurse -Force
  
                Write-EnhancedLog -Message "PsExec64.exe has been successfully downloaded and moved to: $finalPath"
            }
        }
        catch {
            # Handle any errors during the process
            Write-Error "An error occurred: $_"
        }
    }


  

}



function MyRegisterScheduledTask {


    <#
.SYNOPSIS
Registers a scheduled task with the system.

.DESCRIPTION
This function creates a new scheduled task with the specified parameters, including the name, description, VBScript path, and PowerShell script path. It sets up a basic daily trigger and runs the task as the SYSTEM account with the highest privileges. Enhanced logging is used for status messages and error handling to manage potential issues.

.PARAMETER schtaskName
The name of the scheduled task to register.

.PARAMETER schtaskDescription
A description for the scheduled task.

.PARAMETER Path_vbs
The path to the VBScript file used to run the PowerShell script.

.PARAMETER Path_PSscript
The path to the PowerShell script to execute.

.EXAMPLE
MyRegisterScheduledTask -schtaskName "MyTask" -schtaskDescription "Performs automated checks" -Path_vbs "C:\Scripts\run-hidden.vbs" -Path_PSscript "C:\Scripts\myScript.ps1"

This example registers a new scheduled task named "MyTask" that executes "myScript.ps1" using "run-hidden.vbs".
#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$schtaskName,

        [Parameter(Mandatory = $true)]
        [string]$schtaskDescription,

        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_ -File})]
        [string]$Path_vbs,

        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_ -File})]
        [string]$Path_PSscript,

        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_ -File})]
        [string]$PackageExecutionContext
    )

    try {
        Write-EnhancedLog -Message "Registering scheduled task: $schtaskName" -Level "INFO"

        $startTime = (Get-Date).AddMinutes(1).ToString("HH:mm")

        
        # $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "`"$Path_vbs`" `"$Path_PSscript`""
        # $argList = "-NoExit -ExecutionPolicy Bypass -File"
        # $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "`"$argList`" `"$Path_PSscript`""



        # Define the path to the PowerShell script
        # $Path_PSscript = "C:\Path\To\Your\Script.ps1"

        # Define the arguments for the PowerShell executable
        # $argList = "-NoExit -ExecutionPolicy Bypass -File `"$Path_PSscript`""

        # # Create the scheduled task action
        # $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument $argList








        # # Load the configuration from config.json
        # $configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
        # $config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

        # # Define the principal for the task
        # $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
        # Write-EnhancedLog -Message "Principal for the task defined." -Level "INFO"

        # Define the action based on the provided options in the config.json
        if ($config.UsePSADT) {
            Write-EnhancedLog -Message "setting up Schedule Task action for Service UI and PSADT" -Level "INFO"

            # Define the path to the PowerShell Application Deployment Toolkit executable
            # $ToolkitExecutable = "$PSScriptRoot\Private\PSAppDeployToolkit\Toolkit\Deploy-Application.exe"
            $ToolkitExecutable = "$Path_PR\Private\PSAppDeployToolkit\Toolkit\Deploy-Application.exe"

            # Define the path to the ServiceUI executable
            # $ServiceUIExecutable = "$PSScriptRoot\Private\ServiceUI.exe"
            $ServiceUIExecutable = "$Path_PR\Private\ServiceUI.exe"

            # Define the deployment type
            $DeploymentType = "install"

            # Define the arguments for ServiceUI.exe
            $argList = "-process:explorer.exe `"$ToolkitExecutable`" -DeploymentType $DeploymentType"

            # Create the scheduled task action
            $action = New-ScheduledTaskAction -Execute $ServiceUIExecutable -Argument $argList
        }
        else {
            Write-EnhancedLog -Message "Setting up Scheduled Task action for wscript and VBS" -Level "INFO"

            # Define the arguments for wscript.exe
            $argList = "`"$Path_vbs`" `"$Path_PSscript`""

            # Create the scheduled task action for wscript and VBS
            $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\wscript.exe" -Argument $argList
        }


        # Write-EnhancedLog -Message "Scheduled Task '$($config.TaskName)' created successfully." -Level "INFO"

        


















        #option 1 - NO PSADT but rather Wscript and VBS

        # $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\wscript.exe" -Argument "`"$Path_vbs`" `"$Path_PSscript`""




        # #option 2 - ServiceUI calling PSADT in the SYSTEM context
        # Write-EnhancedLog -Message "setting up Schedule Task action for Service UI and PSADT" -Level "INFO"

        # # Define the path to the PowerShell Application Deployment Toolkit executable
        # # $ToolkitExecutable = "$PSScriptRoot\Private\PSAppDeployToolkit\Toolkit\Deploy-Application.exe"
        # $ToolkitExecutable = "$Path_PR\Private\PSAppDeployToolkit\Toolkit\Deploy-Application.exe"

        # # Define the path to the ServiceUI executable
        # # $ServiceUIExecutable = "$PSScriptRoot\Private\ServiceUI.exe"
        # $ServiceUIExecutable = "$Path_PR\Private\ServiceUI.exe"

        # # Define the deployment type
        # $DeploymentType = "install"

        # # Define the arguments for ServiceUI.exe
        # $argList = "-process:explorer.exe `"$ToolkitExecutable`" -DeploymentType $DeploymentType"

        # # Create the scheduled task action
        # $action = New-ScheduledTaskAction -Execute $ServiceUIExecutable -Argument $argList



        #option 1: Trigger - Daily Frequency

        # $trigger = New-ScheduledTaskTrigger -Daily -At $startTime

        #option 2: Trigger On logon of user defaultuser0 (OOBE)




        # Load the configuration from config.json
        # $configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
        # $config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

        # Define the trigger based on the TriggerType
        if ($config.TriggerType -eq "Daily") {
            $trigger = New-ScheduledTaskTrigger -Daily -At $startTime
            Write-EnhancedLog -Message "Trigger set to Daily at $startTime" -Level "INFO"
        }
        elseif ($config.TriggerType -eq "Logon") {
            if (-not $config.LogonUserId) {
                throw "LogonUserId must be specified for Logon trigger type."
            }
            # $trigger = New-ScheduledTaskTrigger -AtLogOn -User $config.LogonUserId
            $trigger = New-ScheduledTaskTrigger -AtLogOn
            Write-EnhancedLog -Message "Trigger set to logon of user $($config.LogonUserId)" -Level "INFO"
        }
        else {
            throw "Invalid TriggerType specified in the configuration."
        }

        $principal = New-ScheduledTaskPrincipal -UserId "NT AUTHORITY\SYSTEM" -LogonType ServiceAccount -RunLevel Highest

        # $task = Register-ScheduledTask -TaskName $schtaskName -Action $action -Trigger $trigger -Principal $principal -Description $schtaskDescription -Force


  

        # Check if the task should run on demand (Zero triggers defined)
        if ($config.RunOnDemand -eq $true) {
            Write-EnhancedLog -Message "calling Register-ScheduledTask with RunOnDemand set to $($config.RunOnDemand)"
            # Task to run on demand; no trigger defined
            $task = Register-ScheduledTask -TaskName $schtaskName -Action $action -Principal $principal -Description $schtaskDescription -Force

            $task = Get-ScheduledTask -TaskName $schtaskName
        }
        else {
            # Define your trigger here
            Write-EnhancedLog -Message "calling Register-ScheduledTask with RunOnDemand set to $($config.RunOnDemand)"
            $task = Register-ScheduledTask -TaskName $schtaskName -Action $action -Trigger $trigger -Principal $principal -Description $schtaskDescription -Force
            # $DBG

            Write-EnhancedLog -Message "calling Register-ScheduledTask done"

            Write-EnhancedLog -Message "calling Get-ScheduledTask"
            $task = Get-ScheduledTask -TaskName $schtaskName
            Write-EnhancedLog -Message "calling Get-ScheduledTask done"
            
            $task.Triggers[0].Repetition.Interval = $RepetitionInterval
            $task | Set-ScheduledTask
        }



        # Updating the task to include repetition with a 5-minute interval
        

        # Check the execution context specified in the config
        if ($PackageExecutionContext -eq "User") {
            # This code block will only execute if ExecutionContext is set to "User"

            # Connect to the Task Scheduler service
            $ShedService = New-Object -ComObject 'Schedule.Service'
            $ShedService.Connect()

            # Get the folder where the task is stored (root folder in this case)
            $taskFolder = $ShedService.GetFolder("\")
    
            # Get the existing task by name
            $Task = $taskFolder.GetTask("$schtaskName")

            # Update the task with a new definition
            $taskFolder.RegisterTaskDefinition("$schtaskName", $Task.Definition, 6, 'Users', $null, 4)  # 6 is TASK_CREATE_OR_UPDATE
        }
        else {
            Write-Host "Execution context is not set to 'User', skipping this block."
        }



        Write-EnhancedLog -Message "Scheduled task $schtaskName registered successfully." -Level "INFO"
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while registering the scheduled task: $_" -Level "ERROR"
        throw $_
    }
}



function SetupNewTaskEnvironment {

    <#
.SYNOPSIS
Sets up a new task environment for scheduled task execution.

.DESCRIPTION
This function prepares the environment for a new scheduled task. It creates a specified directory, determines the PowerShell script path based on the script mode, generates a VBScript to run the PowerShell script hidden, and finally registers the scheduled task with the provided parameters. It utilizes enhanced logging for feedback and error handling to manage potential issues.

.PARAMETER Path_PR
The path where the task's scripts and support files will be stored.

.PARAMETER schtaskName
The name of the scheduled task to be created.

.PARAMETER schtaskDescription
A description for the scheduled task.

.PARAMETER ScriptMode
Determines the script type to be executed ("Remediation" or "PackageName").

.EXAMPLE
SetupNewTaskEnvironment -Path_PR "C:\Tasks\MyTask" -schtaskName "MyScheduledTask" -schtaskDescription "This task does something important" -ScriptMode "Remediation"

This example sets up the environment for a scheduled task named "MyScheduledTask" with a specific description, intended for remediation purposes.
#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_ -PathType 'Container'})]
        [string]$Path_PR,

        [Parameter(Mandatory = $true)]
        [string]$schtaskName,

        [Parameter(Mandatory = $true)]
        [string]$schtaskDescription,

        [Parameter(Mandatory = $true)]
        [ValidateSet("Remediation", "PackageName")]
        [string]$ScriptMode,

        [Parameter(Mandatory = $true)]
        [string]$PackageExecutionContext

    )

    try {
        Write-EnhancedLog -Message "Setting up new task environment at $Path_PR." -Level "INFO"

        # New-Item -Path $Path_PR -ItemType Directory -Force | Out-Null
        # Write-EnhancedLog -Message "Created new directory at $Path_PR" -Level "INFO"

        $Path_PSscript = switch ($ScriptMode) {
            "Remediation" { Join-Path $Path_PR "remediation.ps1" }
            "PackageName" { Join-Path $Path_PR "$PackageName.ps1" }
            Default { throw "Invalid ScriptMode: $ScriptMode. Expected 'Remediation' or 'PackageName'." }
        }

        # $Path_vbs = Create-VBShiddenPS -Path_local $Path_PR
        $Path_vbs = $global:Path_VBShiddenPS

        $scheduledTaskParams = @{
            schtaskName             = $schtaskName
            schtaskDescription      = $schtaskDescription
            Path_vbs                = $Path_vbs
            Path_PSscript           = $Path_PSscript
            PackageExecutionContext = $PackageExecutionContext
        }


        Log-params @{scheduledTaskParams = $scheduledTaskParams}

        MyRegisterScheduledTask @scheduledTaskParams

        Write-EnhancedLog -Message "Scheduled task $schtaskName with description '$schtaskDescription' registered successfully." -Level "INFO"
    }
    catch {
        Write-EnhancedLog -Message "An error occurred during setup of new task environment: $_" -Level "ERROR"
        throw $_
    }
}



function Check-ExistingTask {

    <#
.SYNOPSIS
Checks for the existence of a specified scheduled task.

.DESCRIPTION
This function searches for a scheduled task by name and optionally filters it by version. It returns $true if a task matching the specified criteria exists, otherwise $false.

.PARAMETER taskName
The name of the scheduled task to search for.

.PARAMETER version
The version of the scheduled task to match. The task's description must start with "Version" followed by this parameter value.

.EXAMPLE
$exists = Check-ExistingTask -taskName "MyTask" -version "1"
This example checks if a scheduled task named "MyTask" with a description starting with "Version 1" exists.

#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$taskName,

        [string]$version
    )

    try {
        Write-EnhancedLog -Message "Checking for existing scheduled task: $taskName" -Level "INFO"
        $task_existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($null -eq $task_existing) {
            Write-EnhancedLog -Message "No existing task named $taskName found." -Level "INFO"
            return $false
        }

        if ($null -ne $version) {
            $versionMatch = $task_existing.Description -like "Version $version*"
            if ($versionMatch) {
                Write-EnhancedLog -Message "Found matching task with version: $version" -Level "INFO"
            }
            else {
                Write-EnhancedLog -Message "No matching version found for task: $taskName" -Level "INFO"
            }
            return $versionMatch
        }

        return $true
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while checking for the scheduled task: $_" -Level "ERROR"
        throw $_
    }
}




function CheckAndElevate {

    <#
.SYNOPSIS
Elevates the script to run with administrative privileges if not already running as an administrator.

.DESCRIPTION
The CheckAndElevate function checks if the current PowerShell session is running with administrative privileges. If it is not, the function attempts to restart the script with elevated privileges using the 'RunAs' verb. This is useful for scripts that require administrative privileges to perform their tasks.

.EXAMPLE
CheckAndElevate

Checks the current session for administrative privileges and elevates if necessary.

.NOTES
This function will cause the script to exit and restart if it is not already running with administrative privileges. Ensure that any state or data required after elevation is managed appropriately.
#>
    [CmdletBinding()]
    param (
        # Advanced parameters could be added here if needed. For this function, parameters aren't strictly necessary,
        # but you could, for example, add parameters to control logging behavior or to specify a different method of elevation.
        # [switch]$Elevated
    )

    begin {
        try {
            $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
            $isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

            Write-EnhancedLog -Message "Checking for administrative privileges..." -Level "INFO" 
        }
        catch {
            Write-EnhancedLog -Message "Error determining administrative status: $_" -Level "ERROR"
            throw $_
        }
    }

    process {
        if (-not $isAdmin) {
            try {
                Write-EnhancedLog -Message "The script is not running with administrative privileges. Attempting to elevate..." -Level "WARNING"
                
                $arguments = "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$PSCommandPath`" $args"
                Start-Process PowerShell -Verb RunAs -ArgumentList $arguments

                # Invoke-AsSystem -PsExec64Path $PsExec64Path
                
                Write-EnhancedLog -Message "Script re-launched with administrative privileges. Exiting current session." -Level "INFO"
                exit
            }
            catch {
                Write-EnhancedLog -Message "Failed to elevate privileges: $_" -Level "ERROR"
                throw $_
            }
        }
        else {
            Write-EnhancedLog -Message "Script is already running with administrative privileges." -Level "INFO"
        }
    }

    end {
        # This block is typically used for cleanup. In this case, there's nothing to clean up,
        # but it's useful to know about this structure for more complex functions.
    }
}



function CheckAndExecuteTask {

<#
.SYNOPSIS
Checks for an existing scheduled task and executes tasks based on conditions.

.DESCRIPTION
This function checks if a scheduled task with the specified name and version exists. If it does, it proceeds to execute detection and remediation scripts. If not, it sets up a new task environment and registers the task. It uses enhanced logging for status messages and error handling to manage potential issues.

.PARAMETER schtaskName
The name of the scheduled task to check and potentially execute.

.PARAMETER Version
The version of the task to check for. This is used to verify if the correct task version is already scheduled.

.PARAMETER Path_PR
The path to the directory containing the detection and remediation scripts, used if the task needs to be executed.

.EXAMPLE
CheckAndExecuteTask -schtaskName "MyScheduledTask" -Version 1 -Path_PR "C:\Tasks\MyTask"

This example checks for an existing scheduled task named "MyScheduledTask" of version 1. If it exists, it executes the associated tasks; otherwise, it sets up a new environment and registers the task.
#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$schtaskName,

        [Parameter(Mandatory = $true)]
        [int]$Version,

        [Parameter(Mandatory = $true)]
        [string]$Path_PR,

        [Parameter(Mandatory = $true)]
        [string]$ScriptMode, # Adding ScriptMode as a parameter

        [Parameter(Mandatory = $true)]
        [string]$PackageExecutionContext

    )

    try {
        Write-EnhancedLog -Message "Checking for existing task: $schtaskName" -Level "INFO"

        $taskExists = Check-ExistingTask -taskName $schtaskName -version $Version
        if ($taskExists) {
            Write-EnhancedLog -Message "Existing task found. Executing detection and remediation scripts." -Level "INFO"
            Execute-DetectionAndRemediation -Path_PR $Path_PR
        }
        else {
            Write-EnhancedLog -Message "No existing task found. Setting up new task environment." -Level "INFO"
            SetupNewTaskEnvironment -Path_PR $Path_PR -schtaskName $schtaskName -schtaskDescription $schtaskDescription -ScriptMode $ScriptMode -PackageExecutionContext $PackageExecutionContext
        }
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while checking and executing the task: $_" -Level "ERROR"
        throw $_
    }
}


function Copy-FilesToPath {
    <#
.SYNOPSIS
Copies all files and folders in the specified source directory to the specified destination path.

.DESCRIPTION
This function copies all files and folders located in the specified source directory to the specified destination path. It can be used to bundle necessary files and folders with the script for distribution or deployment.

.PARAMETER SourcePath
The source path from where the files and folders will be copied. If not provided, the default will be the directory of the calling script.

.PARAMETER DestinationPath
The destination path where the files and folders will be copied.

.EXAMPLE
Copy-FilesToPath -SourcePath "C:\Source" -DestinationPath "C:\Temp"

This example copies all files and folders in the "C:\Source" directory to the "C:\Temp" directory.

.EXAMPLE
Copy-FilesToPath -DestinationPath "C:\Temp"

This example copies all files and folders in the same directory as the calling script to the "C:\Temp" directory.
#>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $false)]
        # [string]$SourcePath = $PSScriptRoot,
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    begin {
        Write-EnhancedLog -Message "Starting the copy process from the Source Path $SourcePath to $DestinationPath" -Level "INFO"
        
        # Ensure the destination directory exists
        if (-not (Test-Path -Path $DestinationPath)) {
            New-Item -Path $DestinationPath -ItemType Directory | Out-Null
        }
    }

    process {
        try {
            # Copy all items from the source directory to the destination, including subdirectories
            Copy-Item -Path "$SourcePath\*" -Destination $DestinationPath -Recurse -Force -ErrorAction Stop

            Write-EnhancedLog -Message "All items copied successfully from the Source Path $SourcePath to $DestinationPath." -Level "INFO"
        }
        catch {
            Write-EnhancedLog -Message "Error occurred during the copy process: $_" -Level "ERROR"
            throw $_
        }
    }

    end {
        Write-EnhancedLog -Message "Copy process completed." -Level "INFO"
    }
}




function Create-VBShiddenPS {

    <#
.SYNOPSIS
Creates a VBScript file to run a PowerShell script hidden from the user interface.

.DESCRIPTION
This function generates a VBScript (.vbs) file designed to execute a PowerShell script without displaying the PowerShell window. It's particularly useful for running background tasks or scripts that do not require user interaction. The path to the PowerShell script is taken as an argument, and the VBScript is created in a specified directory within the global path variable.

.EXAMPLE
$Path_VBShiddenPS = Create-VBShiddenPS

This example creates the VBScript file and returns its path.
#>


    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$Path_local,

        [string]$DataFolder = "Data",

        [string]$FileName = "run-ps-hidden.vbs"
    )

    try {
        # Construct the full path for DataFolder and validate it manually
        $fullDataFolderPath = Join-Path -Path $Path_local -ChildPath $DataFolder
        if (-not (Test-Path -Path $fullDataFolderPath -PathType Container)) {
            throw "DataFolder does not exist or is not a directory: $fullDataFolderPath"
        }

        # Log message about creating VBScript
        Write-EnhancedLog -Message "Creating VBScript to hide PowerShell window..." -Level "INFO"

        $scriptBlock = @"
Dim shell,fso,file

Set shell=CreateObject("WScript.Shell")
Set fso=CreateObject("Scripting.FileSystemObject")

strPath=WScript.Arguments.Item(0)

If fso.FileExists(strPath) Then
    set file=fso.GetFile(strPath)
    strCMD="powershell -nologo -executionpolicy ByPass -command " & Chr(34) & "&{" & file.ShortPath & "}" & Chr(34)
    shell.Run strCMD,0
End If
"@

        # Combine paths to construct the full path for the VBScript
        $folderPath = $fullDataFolderPath
        $Path_VBShiddenPS = Join-Path -Path $folderPath -ChildPath $FileName

        # Write the script block to the VBScript file
        $scriptBlock | Out-File -FilePath (New-Item -Path $Path_VBShiddenPS -Force) -Force

        # Validate the VBScript file creation
        if (Test-Path -Path $Path_VBShiddenPS) {
            Write-EnhancedLog -Message "VBScript created successfully at $Path_VBShiddenPS" -Level "INFO"
        }
        else {
            throw "Failed to create VBScript at $Path_VBShiddenPS"
        }

        return $Path_VBShiddenPS
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while creating VBScript: $_" -Level "ERROR"
        throw $_
    }
}


function Download-And-Install-ServiceUI {
    [CmdletBinding()]
    param(
        [string]$TargetFolder = "$PSScriptRoot\private"
    )

    Begin {
        try {
            Remove-ExistingServiceUI -TargetFolder $TargetFolder
        }
        catch {
            Write-EnhancedLog -Message "Error during Remove-ExistingServiceUI: $_" -Level "ERROR"
            throw $_
        }
    }

    Process {
        # Define the URL for MDT download
        $url = "https://download.microsoft.com/download/3/3/9/339BE62D-B4B8-4956-B58D-73C4685FC492/MicrosoftDeploymentToolkit_x64.msi"
        
        # Path for the downloaded MSI file
        $msiPath = Join-Path -Path $([System.IO.Path]::GetTempPath()) -ChildPath "MicrosoftDeploymentToolkit_x64.msi"
        
        try {
            # Download the MDT MSI file
            Write-EnhancedLog -Message "Downloading MDT MSI from: $url to: $msiPath" -Level "INFO"
            Invoke-WebRequest -Uri $url -OutFile $msiPath

            # Install the MSI silently
            Write-EnhancedLog -Message "Installing MDT MSI from: $msiPath" -Level "INFO"
            Start-Process msiexec.exe -ArgumentList "/i", "`"$msiPath`"", "/quiet", "/norestart" -Wait

            # Path to the installed ServiceUI.exe
            $installedServiceUIPath = "C:\Program Files\Microsoft Deployment Toolkit\Templates\Distribution\Tools\x64\ServiceUI.exe"
            $finalPath = Join-Path -Path $TargetFolder -ChildPath "ServiceUI.exe"

            # Move ServiceUI.exe to the desired location
            if (Test-Path -Path $installedServiceUIPath) {
                Write-EnhancedLog -Message "Copying ServiceUI.exe from: $installedServiceUIPath to: $finalPath" -Level "INFO"
                Copy-Item -Path $installedServiceUIPath -Destination $finalPath

                Write-EnhancedLog -Message "ServiceUI.exe has been successfully copied to: $finalPath" -Level "INFO"
            }
            else {
                throw "ServiceUI.exe not found at: $installedServiceUIPath"
            }

            # Remove the downloaded MSI file
            Remove-Item -Path $msiPath -Force
        }
        catch {
            # Handle any errors during the process
            Write-Error "An error occurred: $_"
            Write-EnhancedLog -Message "An error occurred: $_" -Level "ERROR"
        }
    }

    End {
        Write-EnhancedLog -Message "Download-And-Install-ServiceUI function execution completed." -Level "INFO"
    }
}



function Download-PSAppDeployToolkit {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$GithubRepository,

        [Parameter(Mandatory)]
        [string]$FilenamePatternMatch,

        [Parameter(Mandatory)]
        [string]$ZipExtractionPath
    )

    begin {
        try {
            # Log the beginning of the function
            Write-EnhancedLog -Message "Starting Download-PSAppDeployToolkit function." -Level "INFO"

            # Set the URI to get the latest release information from the GitHub repository
            $psadtReleaseUri = "https://api.github.com/repos/$GithubRepository/releases/latest"
            Write-EnhancedLog -Message "GitHub release URI: $psadtReleaseUri" -Level "INFO"
        }
        catch {
            Write-EnhancedLog -Message "Error in begin block: $_" -Level "ERROR"
            throw $_
        }
    }

    process {
        try {
            # Get the download URL for the matching filename pattern
            Write-EnhancedLog -Message "Fetching the latest release information from GitHub." -Level "INFO"
            $psadtDownloadUri = (Invoke-RestMethod -Method GET -Uri $psadtReleaseUri).assets | Where-Object { $_.name -like $FilenamePatternMatch } | Select-Object -ExpandProperty browser_download_url
            
            if (-not $psadtDownloadUri) {
                throw "No matching file found for pattern: $FilenamePatternMatch"
            }
            Write-EnhancedLog -Message "Found matching download URL: $psadtDownloadUri" -Level "INFO"
            
            # Set the path for the temporary download location
            $zipTempDownloadPath = Join-Path -Path $([System.IO.Path]::GetTempPath()) -ChildPath (Split-Path -Path $psadtDownloadUri -Leaf)
            Write-EnhancedLog -Message "Temporary download path: $zipTempDownloadPath" -Level "INFO"

            # Download the file to the temporary location
            Write-EnhancedLog -Message "Downloading file from $psadtDownloadUri to $zipTempDownloadPath" -Level "INFO"
            Invoke-WebRequest -Uri $psadtDownloadUri -OutFile $zipTempDownloadPath

            # Unblock the downloaded file if necessary
            Write-EnhancedLog -Message "Unblocking file at $zipTempDownloadPath" -Level "INFO"
            Unblock-File -Path $zipTempDownloadPath

            # Extract the contents of the zip file to the specified extraction path
            Write-EnhancedLog -Message "Extracting file from $zipTempDownloadPath to $ZipExtractionPath" -Level "INFO"
            Expand-Archive -Path $zipTempDownloadPath -DestinationPath $ZipExtractionPath -Force
        }
        catch {
            Write-EnhancedLog -Message "Error in process block: $_" -Level "ERROR"
            throw $_
        }
    }

    end {
        try {
            Write-Host ("File: {0} extracted to Path: {1}" -f $psadtDownloadUri, $ZipExtractionPath)
            Write-EnhancedLog -Message "File extracted successfully to $ZipExtractionPath" -Level "INFO"
        }
        catch {
            Write-EnhancedLog -Message "Error in end block: $_" -Level "ERROR"
            throw $_
        }
    }
}


function Ensure-RunningAsSystem {
    param (
        [Parameter(Mandatory = $true)]
        [string]$PsExec64Path,
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$TargetFolder
    )

     Write-EnhancedLog -Message "Calling Test-RunningAsSystem" -Level "INFO"
    
    if (-not (Test-RunningAsSystem)) {
        # Check if the target folder exists, and create it if it does not
        if (-not (Test-Path -Path $TargetFolder)) {
            New-Item -Path $TargetFolder -ItemType Directory | Out-Null
        }

        $PsExec64Path = Join-Path -Path $TargetFolder -ChildPath "PsExec64.exe"
        
         Write-EnhancedLog -Message "Current session is not running as SYSTEM. Attempting to invoke as SYSTEM..." -Level "INFO"

        Invoke-AsSystem -PsExec64Path $PsExec64Path -ScriptPath $ScriptPath -TargetFolder $TargetFolder
    }
    else {
         Write-EnhancedLog -Message "Session is already running as SYSTEM." -Level "INFO"
    }
}

# # Example usage
# $privateFolderPath = Join-Path -Path $PSScriptRoot -ChildPath "private"
# $PsExec64Path = Join-Path -Path $privateFolderPath -ChildPath "PsExec64.exe"
# $ScriptToRunAsSystem = $MyInvocation.MyCommand.Path

# Ensure-RunningAsSystem -PsExec64Path $PsExec64Path -ScriptPath $ScriptToRunAsSystem -TargetFolder $privateFolderPath




function Ensure-ScriptPathsExist {


    <#
.SYNOPSIS
Ensures that all necessary script paths exist, creating them if they do not.

.DESCRIPTION
This function checks for the existence of essential script paths and creates them if they are not found. It is designed to be called after initializing script variables to ensure the environment is correctly prepared for the script's operations.

.PARAMETER Path_local
The local path where the script's data will be stored. This path varies based on the execution context (system vs. user).

.PARAMETER Path_PR
The specific path for storing package-related files, constructed based on the package name and unique GUID.

.EXAMPLE
Ensure-ScriptPathsExist -Path_local $global:Path_local -Path_PR $global:Path_PR

This example ensures that the paths stored in the global variables $Path_local and $Path_PR exist, creating them if necessary.
#>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$Path_local,

        [Parameter(Mandatory = $true)]
        [string]$Path_PR
    )

    try {
        # Ensure Path_local exists
        if (-not (Test-Path -Path $Path_local)) {
            New-Item -Path $Path_local -ItemType Directory -Force | Out-Null
            Write-EnhancedLog -Message "Created directory: $Path_local" -Level "INFO"
        }

        # Ensure Path_PR exists
        if (-not (Test-Path -Path $Path_PR)) {
            New-Item -Path $Path_PR -ItemType Directory -Force | Out-Null
            Write-EnhancedLog -Message "Created directory: $Path_PR" -Level "INFO"
        }
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while ensuring script paths exist: $_" -Level "ERROR"
    }
}



function Execute-DetectionAndRemediation {

    <#
.SYNOPSIS
Executes detection and remediation scripts located in a specified directory.

.DESCRIPTION
This function navigates to the specified directory and executes the detection script. If the detection script exits with a non-zero exit code, indicating a positive detection, the remediation script is then executed. The function uses enhanced logging for status messages and error handling to manage any issues that arise during execution.

.PARAMETER Path_PR
The path to the directory containing the detection and remediation scripts.

.EXAMPLE
Execute-DetectionAndRemediation -Path_PR "C:\Scripts\MyTask"
This example executes the detection and remediation scripts located in "C:\Scripts\MyTask".
#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_ -PathType 'Container'})]
        [string]$Path_PR
    )

    try {
        Write-EnhancedLog -Message "Executing detection and remediation scripts in $Path_PR..." -Level "INFO"
        Set-Location -Path $Path_PR

        # Execution of the detection script
        & .\detection.ps1
        if ($LASTEXITCODE -ne 0) {
            Write-EnhancedLog -Message "Detection positive, remediation starts now." -Level "INFO"
            & .\remediation.ps1
        }
        else {
            Write-EnhancedLog -Message "Detection negative, no further action needed." -Level "INFO"
        }
    }
    catch {
        Write-EnhancedLog -Message "An error occurred during detection and remediation execution: $_" -Level "ERROR"
        throw $_
    }
}




function Initialize-ScriptVariables {


    <#
.SYNOPSIS
Initializes global script variables and defines the path for storing related files.

.DESCRIPTION
This function initializes global script variables such as PackageName, PackageUniqueGUID, Version, and ScriptMode. Additionally, it constructs the path where related files will be stored based on the provided parameters.

.PARAMETER PackageName
The name of the package being processed.

.PARAMETER PackageUniqueGUID
The unique identifier for the package being processed.

.PARAMETER Version
The version of the package being processed.

.PARAMETER ScriptMode
The mode in which the script is being executed (e.g., "Remediation", "PackageName").

.EXAMPLE
Initialize-ScriptVariables -PackageName "MyPackage" -PackageUniqueGUID "1234-5678" -Version 1 -ScriptMode "Remediation"

This example initializes the script variables with the specified values.

#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$PackageName,

        [Parameter(Mandatory = $true)]
        [string]$PackageUniqueGUID,

        [Parameter(Mandatory = $true)]
        [int]$Version,

        [Parameter(Mandatory = $true)]
        [string]$ScriptMode,

        [Parameter(Mandatory = $true)]
        [string]$PackageExecutionContext
    )

    # Assuming Set-LocalPathBasedOnContext and Test-RunningAsSystem are defined elsewhere
    # $global:Path_local = Set-LocalPathBasedOnContext

    # Default logic for $Path_local if not set by Set-LocalPathBasedOnContext
    if (-not $Path_local) {
        if (Test-RunningAsSystem) {
            # $Path_local = "$ENV:ProgramFiles\_MEM"
            $Path_local = "c:\_MEM"
        }
        else {
            $Path_local = "$ENV:LOCALAPPDATA\_MEM"
        }
    }

    $Path_PR = "$Path_local\Data\$PackageName-$PackageUniqueGUID"
    $schtaskName = "$PackageName - $PackageUniqueGUID"
    $schtaskDescription = "Version $Version"

    try {
        # Assuming Write-EnhancedLog is defined elsewhere
        Write-EnhancedLog -Message "Initializing script variables..." -Level "INFO"

        # Returning a hashtable of all the important variables
        return @{
            PackageName             = $PackageName
            PackageUniqueGUID       = $PackageUniqueGUID
            Version                 = $Version
            ScriptMode              = $ScriptMode
            Path_local              = $Path_local
            Path_PR                 = $Path_PR
            schtaskName             = $schtaskName
            schtaskDescription      = $schtaskDescription
            PackageExecutionContext = $PackageExecutionContext
        }
    }
    catch {
        Write-Error "An error occurred while initializing script variables: $_"
    }
}


function Invoke-AsSystem {
    <#
.SYNOPSIS
Executes a PowerShell script under the SYSTEM context, similar to Intune's execution context.

.DESCRIPTION
The Invoke-AsSystem function executes a PowerShell script using PsExec64.exe to run under the SYSTEM context. This method is useful for scenarios requiring elevated privileges beyond the current user's capabilities.

.PARAMETER PsExec64Path
Specifies the full path to PsExec64.exe. If not provided, it assumes PsExec64.exe is in the same directory as the script.

.EXAMPLE
Invoke-AsSystem -PsExec64Path "C:\Tools\PsExec64.exe"

Executes PowerShell as SYSTEM using PsExec64.exe located at "C:\Tools\PsExec64.exe".

.NOTES
Ensure PsExec64.exe is available and the script has the necessary permissions to execute it.

.LINK
https://docs.microsoft.com/en-us/sysinternals/downloads/psexec
#>
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$PsExec64Path,
        [string]$ScriptPathAsSYSTEM,  # Path to the PowerShell script you want to run as SYSTEM
        [string]$TargetFolder  # Path to the PowerShell script you want to run as SYSTEM
    )

    begin {
        CheckAndElevate
        # Define the arguments for PsExec64.exe to run PowerShell as SYSTEM with the script
        $argList = "-accepteula -i -s -d powershell.exe -NoExit -ExecutionPolicy Bypass -File `"$ScriptPathAsSYSTEM`""
        Write-EnhancedLog -Message "Preparing to execute PowerShell as SYSTEM using PsExec64 with the script: $ScriptPathAsSYSTEM" -Level "INFO"

        Log-Params -Params @{PsExec64Path = $PsExec64Path}

        # Download-PsExec -targetFolder $PsExec64Path
        Download-PsExec -targetFolder $TargetFolder
    }

    process {
        try {
            # Ensure PsExec64Path exists
            if (-not (Test-Path -Path $PsExec64Path)) {
                $errorMessage = "PsExec64.exe not found at path: $PsExec64Path"
                Write-EnhancedLog -Message $errorMessage -Level "ERROR"
                throw $errorMessage
            }

            # Run PsExec64.exe with the defined arguments to execute the script as SYSTEM
            $executingMessage = "Executing PsExec64.exe to start PowerShell as SYSTEM running script: $ScriptPathAsSYSTEM"
            Write-EnhancedLog -Message $executingMessage -Level "INFO"
            Start-Process -FilePath "$PsExec64Path" -ArgumentList $argList -Wait -NoNewWindow
            
            Write-EnhancedLog -Message "SYSTEM session started. Closing elevated session..." -Level "INFO"
            exit

        }
        catch {
            Write-EnhancedLog -Message "An error occurred: $_" -Level "ERROR"
        }
    }
}



function Remove-ExistingPsExec {
    [CmdletBinding()]
    param(
        # [string]$TargetFolder = "$PSScriptRoot\private"
        [string]$TargetFolder
    )

    # Full path for PsExec64.exe
    $PsExec64Path = Join-Path -Path $TargetFolder -ChildPath "PsExec64.exe"

    try {
        # Check if PsExec64.exe exists
        if (Test-Path -Path $PsExec64Path) {
            Write-EnhancedLog -Message "Removing existing PsExec64.exe from: $TargetFolder"
            # Remove PsExec64.exe
            Remove-Item -Path $PsExec64Path -Force
            Write-EnhancedLog -Message "PsExec64.exe has been removed from: $TargetFolder"
        }
        else {
            Write-EnhancedLog -Message "No PsExec64.exe file found in: $TargetFolder"
        }
    }
    catch {
        # Handle any errors during the removal
        Write-Error "An error occurred while trying to remove PsExec64.exe: $_"
    }
}



function Remove-ExistingServiceUI {
    [CmdletBinding()]
    param(
        [string]$TargetFolder = "$PSScriptRoot\private"
    )

    # Full path for ServiceUI.exe
    $ServiceUIPath = Join-Path -Path $TargetFolder -ChildPath "ServiceUI.exe"

    try {
        # Check if ServiceUI.exe exists
        if (Test-Path -Path $ServiceUIPath) {
            Write-EnhancedLog -Message "Removing existing ServiceUI.exe from: $TargetFolder" -Level "INFO"
            # Remove ServiceUI.exe
            Remove-Item -Path $ServiceUIPath -Force
            Write-Output "ServiceUI.exe has been removed from: $TargetFolder"
        }
        else {
            Write-EnhancedLog -Message "No ServiceUI.exe file found in: $TargetFolder" -Level "INFO"
        }
    }
    catch {
        # Handle any errors during the removal
        Write-Error "An error occurred while trying to remove ServiceUI.exe: $_"
        Write-EnhancedLog -Message "An error occurred while trying to remove ServiceUI.exe: $_" -Level "ERROR"
    }
}


function Remove-ScheduledTaskFilesWithLogging {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    try {
        # Validate before removal
        $existsBefore = Validate-PathExistsWithLogging -Path $Path

        if ($existsBefore) {
            Write-EnhancedLog -Message "Calling Remove-Item for path: $Path" -Level "INFO"
            Remove-Item -Path $Path -Recurse -Force
            Write-EnhancedLog -Message "Remove-Item done for path: $Path" -Level "INFO"
        } else {
            Write-EnhancedLog -Message "Path $Path does not exist. No action taken." -Level "WARNING"
        }

        # Validate after removal
        $existsAfter = Validate-PathExistsWithLogging -Path $Path

        if ($existsAfter) {
            Write-EnhancedLog -Message "Path $Path still exists after attempting to remove. Manual intervention may be required." -Level "ERROR"
        } else {
            Write-EnhancedLog -Message "Path $Path successfully removed." -Level "INFO"
        }
    }
    catch {
        Write-EnhancedLog -Message "Error during Remove-Item for path: $Path. Error: $_" -Level "ERROR"
        throw $_
    }
}


function Set-LocalPathBasedOnContext {
    Write-EnhancedLog -Message "Checking running context..." -Level "INFO"
    if (Test-RunningAsSystem) {
        Write-EnhancedLog -Message "Running as system, setting path to Program Files" -Level "INFO"
        # return "$ENV:Programfiles\_MEM"
        return "C:\_MEM"
    }
    else {
        Write-EnhancedLog -Message "Running as user, setting path to Local AppData" -Level "INFO"
        return "$ENV:LOCALAPPDATA\_MEM"
    }
}


function Start-ServiceUIWithAppDeploy {
    [CmdletBinding()]
    param (
        [string]$PSADTExecutable = "$PSScriptRoot\Private\PSAppDeployToolkit\Toolkit\Deploy-Application.exe",
        [string]$ServiceUIExecutable = "$PSScriptRoot\Private\ServiceUI.exe",
        [string]$DeploymentType = "Install",
        [string]$DeployMode = "Interactive"
    )

    try {
        # Verify if the ServiceUI executable exists
        if (-not (Test-Path -Path $ServiceUIExecutable)) {
            throw "ServiceUI executable not found at path: $ServiceUIExecutable"
        }

        # Verify if the PSAppDeployToolkit executable exists
        if (-not (Test-Path -Path $PSADTExecutable)) {
            throw "PSAppDeployToolkit executable not found at path: $PSADTExecutable"
        }

        # Log the start of the process
        Write-EnhancedLog -Message "Starting ServiceUI.exe with Deploy-Application.exe" -Level "INFO"

        # Define the arguments to pass to ServiceUI.exe
        $arguments = "-process:explorer.exe `"$PSADTExecutable`" -DeploymentType $DeploymentType -Deploymode $Deploymode"

        # Start the ServiceUI.exe process with the specified arguments
        Start-Process -FilePath $ServiceUIExecutable -ArgumentList $arguments -Wait -WindowStyle Hidden

        # Log successful completion
        Write-EnhancedLog -Message "ServiceUI.exe started successfully with Deploy-Application.exe" -Level "INFO"
    }
    catch {
        # Handle any errors during the process
        Write-Error "An error occurred: $_"
        Write-EnhancedLog -Message "An error occurred: $_" -Level "ERROR"
    }
}


function Test-RunningAsSystem {
    $systemSid = New-Object System.Security.Principal.SecurityIdentifier "S-1-5-18"
    $currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User

    return $currentSid -eq $systemSid
}



function Unregister-ScheduledTaskWithLogging {
    param (
        [Parameter(Mandatory = $true)]
        [string]$TaskName
    )

    try {
        Write-EnhancedLog -Message "Checking if task '$TaskName' exists before attempting to unregister." -Level "INFO"
        $taskExistsBefore = Check-ExistingTask -taskName $TaskName
        
        if ($taskExistsBefore) {
            Write-EnhancedLog -Message "Task '$TaskName' found. Proceeding to unregister." -Level "INFO"
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-EnhancedLog -Message "Unregister-ScheduledTask done for task: $TaskName" -Level "INFO"
        } else {
            Write-EnhancedLog -Message "Task '$TaskName' not found. No action taken." -Level "INFO"
        }

        Write-EnhancedLog -Message "Checking if task '$TaskName' exists after attempting to unregister." -Level "INFO"
        $taskExistsAfter = Check-ExistingTask -taskName $TaskName
        
        if ($taskExistsAfter) {
            Write-EnhancedLog -Message "Task '$TaskName' still exists after attempting to unregister. Manual intervention may be required." -Level "ERROR"
        } else {
            Write-EnhancedLog -Message "Task '$TaskName' successfully unregistered." -Level "INFO"
        }
    }
    catch {
        Write-EnhancedLog -Message "Error during Unregister-ScheduledTask for task: $TaskName. Error: $_" -Level "ERROR"
        throw $_
    }
}


function Validate-PathExistsWithLogging {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $exists = Test-Path -Path $Path

    if ($exists) {
        Write-EnhancedLog -Message "Path exists: $Path" -Level "INFO"
    } else {
        Write-EnhancedLog -Message "Path does not exist: $Path" -Level "WARNING"
    }

    return $exists
}



function Verify-CopyOperation {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $false)]
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    begin {
        Write-EnhancedLog -Message "Verifying copy operation..." -Level "INFO"
        $sourceItems = Get-ChildItem -Path $SourcePath -Recurse
        $destinationItems = Get-ChildItem -Path $DestinationPath -Recurse

        # Use a generic list for better performance compared to using an array with +=
        $verificationResults = New-Object System.Collections.Generic.List[Object]
    }

    process {
        try {
            foreach ($item in $sourceItems) {
                $relativePath = $item.FullName.Substring($SourcePath.Length)
                $correspondingPath = Join-Path -Path $DestinationPath -ChildPath $relativePath

                if (-not (Test-Path -Path $correspondingPath)) {
                    $verificationResults.Add([PSCustomObject]@{
                            Status       = "Missing"
                            SourcePath   = $item.FullName
                            ExpectedPath = $correspondingPath
                        })
                }
            }

            foreach ($item in $destinationItems) {
                $relativePath = $item.FullName.Substring($DestinationPath.Length)
                $correspondingPath = Join-Path -Path $SourcePath -ChildPath $relativePath

                if (-not (Test-Path -Path $correspondingPath)) {
                    $verificationResults.Add([PSCustomObject]@{
                            Status     = "Extra"
                            SourcePath = $correspondingPath
                            ActualPath = $item.FullName
                        })
                }
            }
        }
        catch {
            Write-EnhancedLog -Message "Error during verification process: $_" -Level "ERROR"
        }
    }

    end {
        if ($verificationResults.Count -gt 0) {
            Write-EnhancedLog -Message "Discrepancies found. See detailed log." -Level "WARNING"
            $verificationResults | Format-Table -AutoSize | Out-String | ForEach-Object { Write-EnhancedLog -Message $_ -Level "INFO" }
        }
        else {
            Write-EnhancedLog -Message "All items verified successfully. No discrepancies found." -Level "INFO"
        }

        Write-EnhancedLog -Message ("Total items in source: " + $sourceItems.Count) -Level "INFO"
        Write-EnhancedLog -Message ("Total items in destination: " + $destinationItems.Count) -Level "INFO"
    }
}


function Clean-ExportsFolder {
    param (
        [Parameter(Mandatory = $true)]
        [string]$FolderPath
    )

    if (Test-Path -Path $FolderPath) {
        # Get all files in the folder
        $files = Get-ChildItem -Path "$FolderPath\*" -Recurse

        # Remove each file and log its name
        foreach ($file in $files) {
            try {
                Remove-Item -Path $file.FullName -Recurse -Force
                Write-EnhancedLog -Message "Deleted file: $($file.FullName)" -Level "INFO"
            } catch {
                Write-EnhancedLog -Message "Failed to delete file: $($file.FullName) - Error: $_" -Level "ERROR"
            }
        }

        Write-EnhancedLog -Message "Cleaned up existing folder at: $FolderPath" -Level "INFO"
    } else {
        # Create the folder if it does not exist
        New-Item -ItemType Directory -Path $FolderPath | Out-Null
        Write-EnhancedLog -Message "Created folder at: $FolderPath" -Level "INFO"
    }
}

# Example usage
# $folderPath = "C:\path\to\exports"
# Clean-ExportsFolder -FolderPath $folderPath -LogFunction Write-EnhancedLog


function Validate-FileExists {
    param (
        [Parameter(Mandatory = $true)]
        [string]$FilePath
    )

    if (-Not (Test-Path -Path $FilePath)) {
        Write-EnhancedLog -Message "File '$FilePath' does not exist." -Level "ERROR"
        throw "File '$FilePath' does not exist."
    }
}




function Validate-FileUpload {
    param (
        [Parameter(Mandatory = $true)]
        [string]$DocumentDriveId,

        [Parameter(Mandatory = $true)]
        [string]$FolderName,

        [Parameter(Mandatory = $true)]
        [string]$FileName,

        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    $url = "https://graph.microsoft.com/v1.0/drives/$DocumentDriveId/root:/$FolderName/$FileName"
    try {
        $response = Invoke-RestMethod -Headers $Headers -Uri $url -Method GET
        if ($response) {
            Write-EnhancedLog -Message "File '$FileName' exists in '$FolderName' after upload." -Level "INFO"
        } else {
            Write-EnhancedLog -Message "File '$FileName' does not exist in '$FolderName' after upload." -Level "ERROR"
            throw "File '$FileName' does not exist in '$FolderName' after upload."
        }
    }
    catch {
        Write-EnhancedLog -Message "Failed to validate file '$FileName' in '$FolderName': $_" -Level "ERROR"
        throw $_
    }
}


function Ensure-ExportsFolder {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$ExportsFolderName,
        [Parameter(Mandatory = $true)]
        [string]$ExportSubFolderName
    )

    # Construct the full path to the exports folder
    $ExportsBaseFolderPath = Join-Path -Path $BasePath -ChildPath $ExportsFolderName
    $ExportsFolderPath = Join-Path -Path $ExportsBaseFolderPath -ChildPath $ExportSubFolderName

    # Check if the base exports folder exists
    if (-Not (Test-Path -Path $ExportsBaseFolderPath)) {
        # Create the base exports folder
        New-Item -ItemType Directory -Path $ExportsBaseFolderPath | Out-Null
        Write-EnhancedLog -Message "Created base exports folder at: $ExportsBaseFolderPath" -Level "INFO"
    }

    # Ensure the subfolder is clean
    # Clean-ExportsFolder -FolderPath $ExportsFolderPath

    # Return the full path of the exports folder
    return $ExportsFolderPath
}


function Get-SharePointDocumentDriveId {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$SiteObjectId,

        [Parameter(Mandatory = $true)]
        [string]$DocumentDriveName,

        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    try {
        # Get the subsite ID
        $url = "https://graph.microsoft.com/v1.0/groups/$SiteObjectId/sites/root"
        $subsiteID = (Invoke-RestMethod -Headers $Headers -Uri $url -Method GET).id
        Write-EnhancedLog -Message "Retrieved subsite ID: $subsiteID" -Level "INFO"

        # Get the drives
        $url = "https://graph.microsoft.com/v1.0/sites/$subsiteID/drives"
        $drives = Invoke-RestMethod -Headers $Headers -Uri $url -Method GET
        Write-EnhancedLog -Message "Retrieved drives for subsite ID: $subsiteID" -Level "INFO"

        # Find the document drive ID
        $documentDriveId = ($drives.value | Where-Object { $_.name -eq $DocumentDriveName }).id

        if ($documentDriveId) {
            Write-EnhancedLog -Message "Found document drive ID: $documentDriveId" -Level "INFO"
            return $documentDriveId
        } else {
            Write-EnhancedLog -Message "Document drive '$DocumentDriveName' not found." -Level "ERROR"
            throw "Document drive '$DocumentDriveName' not found."
        }
    }
    catch {
        Write-EnhancedLog -Message "Failed to get document drive ID: $_" -Level "ERROR"
        throw $_
    }
}


# # Example usage
# $headers = @{
#     "Authorization" = "Bearer YOUR_ACCESS_TOKEN"
#     "Content-Type"  = "application/json"
# }

# $siteObjectId = "your_site_object_id"
# $documentDriveName = "Documents"

# Get-SharePointDocumentDriveId -SiteObjectId $siteObjectId -DocumentDriveName $documentDriveName -Headers $headers


function New-SharePointFolder {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$DocumentDriveId,

        [Parameter(Mandatory = $true)]
        [string]$ParentFolderPath,

        [Parameter(Mandatory = $true)]
        [string]$FolderName,

        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    try {
        # Check if the folder already exists
        $checkUrl = "https://graph.microsoft.com/v1.0/drives/" + $DocumentDriveId + "/root:/" + $ParentFolderPath + ":/children"
        $existingFolders = Invoke-RestMethod -Headers $Headers -Uri $checkUrl -Method GET
        $existingFolder = $existingFolders.value | Where-Object { $_.name -eq $FolderName -and $_.folder }

        if ($existingFolder) {
            Write-EnhancedLog -Message "Folder '$FolderName' already exists in '$ParentFolderPath'. Skipping folder creation." -Level "INFO"
            return $existingFolder
        }
    }
    catch {
        Write-EnhancedLog -Message "Folder '$FolderName' not found in '$ParentFolderPath'. Proceeding with folder creation." -Level "INFO"
    }

    try {
        # If the folder does not exist, create it
        $url = "https://graph.microsoft.com/v1.0/drives/" + $DocumentDriveId + "/root:/" + $ParentFolderPath + ":/children"
        $body = @{
            "@microsoft.graph.conflictBehavior" = "fail"
            "name"                              = $FolderName
            "folder"                            = @{}
        }

        Write-EnhancedLog -Message "Creating folder '$FolderName' in '$ParentFolderPath'..." -Level "INFO"
        $createdFolder = Invoke-RestMethod -Headers $Headers -Uri $url -Body ($body | ConvertTo-Json) -Method POST
        Write-EnhancedLog -Message "Folder created successfully." -Level "INFO"
        return $createdFolder
    }
    catch {
        Write-EnhancedLog -Message "Failed to create folder '$FolderName' in '$ParentFolderPath': $_" -Level "ERROR"
        throw $_
    }
}

# # Example usage
# $headers = @{
#     "Authorization" = "Bearer YOUR_ACCESS_TOKEN"
#     "Content-Type"  = "application/json"
# }

# $documentDriveId = "your_document_drive_id"
# $parentFolderPath = "your/parent/folder/path"
# $folderName = "NewFolder"

# New-SharePointFolder -DocumentDriveId $documentDriveId -ParentFolderPath $parentFolderPath -FolderName $folderName -Headers $headers


function Upload-FileToSharePoint {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$DocumentDriveId,

        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string]$FolderName,

        [Parameter(Mandatory = $true)]
        [hashtable]$Headers
    )

    try {
        # Validate file existence before upload
        Validate-FileExists -FilePath $FilePath

        # Read the file content
        $content = Get-Content -Path $FilePath -Raw
        $filename = (Get-Item -Path $FilePath).Name

        # Construct the PUT URL
        $putUrl = "https://graph.microsoft.com/v1.0/drives/$DocumentDriveId/root:/$FolderName/$($filename):/content"

        # Upload the file
        Write-EnhancedLog -Message "Uploading file '$filename' to folder '$FolderName'..." -Level "INFO"
        $uploadResponse = Invoke-RestMethod -Headers $Headers -Uri $putUrl -Body $content -Method PUT
        Write-EnhancedLog -Message "File '$filename' uploaded successfully." -Level "INFO"

        # Validate file existence after upload
        Validate-FileUpload -DocumentDriveId $DocumentDriveId -FolderName $FolderName -FileName $filename -Headers $Headers

        return $uploadResponse
    }
    catch {
        Write-EnhancedLog -Message "Failed to upload file '$filename' to folder '$FolderName': $_" -Level "ERROR"
        throw $_
    }
}


function Export-VPNConnectionsToXML {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$ExportFolder
    )

    # Get the list of current VPN connections
    $vpnConnections = Get-VpnConnection

    # Check if there are no VPN connections
    if ($vpnConnections.Count -eq 0) {
        Write-EnhancedLog -Message "NO VPN connections found." -Level "WARNING"
        return
    }

    # Generate a timestamp for the export
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $baseOutputPath = Join-Path -Path $ExportFolder -ChildPath "VPNExport_$timestamp"

    # Setup parameters for Export-Data using splatting
    $exportParams = @{
        Data             = $vpnConnections
        BaseOutputPath   = $baseOutputPath
        IncludeCSV       = $true
        IncludeJSON      = $true
        IncludeXML       = $true
        # IncludeHTML      = $true
        IncludePlainText = $true
        IncludeExcel     = $true
        IncludeYAML      = $true
        # IncludeGridView  = $true  # Note: GridView displays data but doesn't export/save it
    }

    # Call the Export-Data function with splatted parameters
    Export-Data @exportParams
    Write-EnhancedLog -Message "Data export completed successfully." -Level "INFO"
}


function New-VPNConnection {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$ConnectionName,

        [Parameter(Mandatory = $true)]
        [string]$ServerAddress,

        [Parameter(Mandatory = $true)]
        # [string]$TunnelType = "Pptp"  # Default to Pptp, can be changed to Ikev2, L2tp, etc.
        [string]$TunnelType
    )

    try {
        # Validate if VPN connection already exists
        if (Test-VPNConnection -ConnectionName $ConnectionName) {
            Write-EnhancedLog -Message "VPN connection '$ConnectionName' already exists." -Level "INFO"
            return
        }

        # Create the VPN connection
        Add-VpnConnection -Name $ConnectionName -ServerAddress $ServerAddress -TunnelType $TunnelType -AuthenticationMethod MSChapv2 -EncryptionLevel Optional -Force

        # Validate if VPN connection was created successfully
        if (Test-VPNConnection -ConnectionName $ConnectionName) {
            Write-EnhancedLog -Message "VPN connection '$ConnectionName' created successfully." -Level "INFO"
        } else {
            Write-EnhancedLog -Message "Failed to create VPN connection '$ConnectionName'." -Level "ERROR"
            throw "Failed to create VPN connection '$ConnectionName'."
        }
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while creating VPN connection '$ConnectionName': $_" -Level "ERROR"
        throw $_
    }
}


function Test-VPNConnection {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$ConnectionName
    )

    try {
        # Check if the VPN connection exists
        $vpnConnection = Get-VpnConnection -Name $ConnectionName -ErrorAction SilentlyContinue
        if ($vpnConnection) {
            Write-EnhancedLog -Message "VPN connection '$ConnectionName' exists." -Level "INFO"
            return $true
        } else {
            Write-EnhancedLog -Message "VPN connection '$ConnectionName' does not exist." -Level "INFO"
            return $false
        }
    }
    catch {
        Write-EnhancedLog -Message "An error occurred while checking VPN connection '$ConnectionName': $_" -Level "ERROR"
        throw $_
    }
}


function Add-GuidToPs1Files {


    <#
.SYNOPSIS
Adds a unique GUID and timestamp to the top of each .ps1 file in a specified directory.

.DESCRIPTION
This function searches for PowerShell script files (.ps1) within a specified subdirectory of a given root directory. It then prepends a unique GUID and a timestamp to each file for tracking purposes. This is useful for marking scripts in bulk operations or deployments.

.PARAMETER AOscriptDirectory
The root directory under which the target program folder resides.

.PARAMETER programfoldername
The name of the subdirectory containing the .ps1 files to be modified.

.EXAMPLE
Add-GuidToPs1Files -AOscriptDirectory "d:\Scripts" -programfoldername "MyProgram"

Adds a tracking GUID and timestamp to all .ps1 files under "d:\Scripts\apps-winget\MyProgram".

.NOTES
Author: Your Name
Date: Get the current date
Version: 1.0

#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        # [ValidateScript({Test-Path $_})]
        [string]$AOscriptDirectory,

        [Parameter(Mandatory = $true)]
        [string]$programfoldername
    )

    # Helper function for logging
    Begin {
        Write-EnhancedLog -Message "Starting to modify PowerShell files." -Level "INFO"
    }

    Process {
        $targetFolder = Join-Path -Path $AOscriptDirectory -ChildPath "apps-winget\$programfoldername"

        if (-Not (Test-Path -Path $targetFolder)) {
            Write-EnhancedLog -Message "The target folder does not exist: $targetFolder" -Level "ERROR"
            return
        }

        $ps1Files = Get-ChildItem -Path $targetFolder -Filter *.ps1 -Recurse
        if ($ps1Files.Count -eq 0) {
            Write-EnhancedLog -Message "No PowerShell files (.ps1) found in $targetFolder" -Level "WARNING"
            return
        }

        foreach ($file in $ps1Files) {
            try {
                $content = Get-Content -Path $file.FullName -ErrorAction Stop
                $pattern = '^#Unique Tracking ID: .+'
                $content = $content | Where-Object { $_ -notmatch $pattern }

                $guid = [guid]::NewGuid().ToString("D").ToLower()
                $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                $lineToAdd = "#Unique Tracking ID: $guid, Timestamp: $timestamp"
                $newContent = $lineToAdd, $content

                Set-Content -Path $file.FullName -Value $newContent -ErrorAction Stop
                Write-EnhancedLog -Message "Modified file: $($file.FullName)" -Level "VERBOSE"
            }
            catch {
                Write-EnhancedLog -Message "Failed to modify file: $($file.FullName). Error: $($_.Exception.Message)" -Level "ERROR"
            }
        }
    }

    End {
        Write-EnhancedLog -Message "Completed modifications." -Level "INFO"
    }
}


# Example usage:
# Add-GuidToPs1Files -AOscriptDirectory $AOscriptDirectory


function Compile-Win32_intunewin {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Prg,

        [Parameter(Mandatory)]
        [string]$Repo_winget,

        [Parameter(Mandatory)]
        [string]$Repo_Path,

         [Parameter(Mandatory)]
        [string]$Prg_Path
    )

    Write-EnhancedLog -Message "Entering Compile-Win32_intunewin" -Level "WARNING"



    

    # Check for application image
    $Prg_img = if (Test-Path -Path (Join-Path -Path $Prg_Path -ChildPath "$($Prg.id).png")) {
        Join-Path -Path $Prg_Path -ChildPath "$($Prg.id).png"
    }
    else {
        "$Repo_Path\resources\template\winget\winget-managed.png"
    }

    # Download the latest IntuneWinAppUtil
    # Invoke-WebRequest -Uri $IntuneWinAppUtil_online -OutFile "$Repo_Path\resources\IntuneWinAppUtil.exe" -UseBasicParsing

    # Create the .intunewin file
    # Start-Process -FilePath "$Repo_Path\resources\IntuneWinAppUtil.exe" -ArgumentList "-c `"$Prg_Path`" -s install.ps1 -o `"$Prg_Path`" -q" -Wait -WindowStyle Hidden (when used in Linux do not use windowstyle hidden)
    # Start-Process -FilePath "$Repo_Path\resources\IntuneWinAppUtil.exe" -ArgumentList "-c `"$Prg_Path`" -s install.ps1 -o `"$Prg_Path`" -q" -Wait

    Upload-Win32App -Prg $Prg -Prg_Path $Prg_Path -Prg_img $Prg_img
    # Upload-Win32App -Prg $Prg -Prg_Path $Prg_Path

    Write-EnhancedLog -Message "Exiting Compile-Win32_intunewin" -Level "INFO"
}


function Create-AADGroup ($Prg) {


    # # Convert the Client Secret to a SecureString
    # $SecureClientSecret = ConvertTo-SecureString $connectionParams.ClientSecret -AsPlainText -Force

    # # Create a PSCredential object with the Client ID as the user and the Client Secret as the password
    # $ClientSecretCredential = New-Object System.Management.Automation.PSCredential ($connectionParams.ClientId, $SecureClientSecret)

    # # Connect to Microsoft Graph
    # Connect-MgGraph -TenantId $connectionParams.TenantId -ClientSecretCredential $ClientSecretCredential

    # Your code that interacts with Microsoft Graph goes here


    # Create Group
    # $grpname = "$($global:SettingsVAR.AADgrpPrefix )$($Prg.id)"
    Write-EnhancedLog -Message "setting Group Name" -Level "WARNING"
    $grpname = "SG007 - Intune - Apps - Microsoft Teams - WinGet - Windows Package Manager"
    if (!$(Get-MgGroup -Filter "DisplayName eq '$grpname'")) {
        # Write-Host "  Create AAD group for assigment:  $grpname"

        Write-EnhancedLog -Message " Did not find Group $grpname " -Level "WARNING"
        
        # $GrpObj = New-MgGroup -DisplayName "$grpname" -Description "App assigment: $($Prg.id) $($Prg.manager)" -MailEnabled:$False  -MailNickName $grpname -SecurityEnabled
    }
    else { $GrpObj = Get-MgGroup -Filter "DisplayName eq '$grpname'" }


    Write-EnhancedLog -Message " Assign Group > $grpname <  to  > $($Prg.Name)" -Level "WARNING"
  


    Write-EnhancedLog -Message " calling Get-IntuneWin32App " -Level "WARNING"
    $Win32App = Get-IntuneWin32App -DisplayName "$($Prg.Name)"


    Write-EnhancedLog -Message " calling Get-IntuneWin32App - done " -Level "INFO"


    Write-EnhancedLog -Message " calling Add-IntuneWin32AppAssignmentGroup " -Level "WARNING"
    Add-IntuneWin32AppAssignmentGroup -Include -ID $Win32App.id -GroupID $GrpObj.id -Intent "available" -Notification "showAll"


    Write-EnhancedLog -Message " calling Add-IntuneWin32AppAssignmentGroup - done " -Level "INFO"
}



function Get-CustomWin32AppName {
    [CmdletBinding()]
    param(
        [string]$PRGID
    )
    process {
        if (-not [string]::IsNullOrWhiteSpace($PRGID)) {
            return $PRGID  # Directly return PRGID if it's valid
        }
        else {
            return "DefaultAppName"  # Fallback if PRGID is not provided
        }
    }
}


# Get-CustomWin32AppName


function Invoke-PrinterInstallation {

    <#
.SYNOPSIS
Installs or uninstalls printer drivers based on JSON configuration files.

.DESCRIPTION
This PowerShell function reads printer installation settings from a specified printer configuration JSON file (printer.json) and application configuration JSON file (config.json). It constructs and optionally executes command lines for installing or uninstalling printer drivers. The function proceeds only if the 'PrinterInstall' attribute in the application configuration is set to true.

.PARAMETER PrinterConfigPath
The full file path to the printer configuration JSON file (printer.json). This file contains the printer settings such as PrinterName, PrinterIPAddress, PortName, DriverName, InfPathRelative, InfFileName, and DriverIdentifier.

.PARAMETER AppConfigPath
The full file path to the application configuration JSON file (config.json). This file contains application-wide settings including the 'PrinterInstall' flag that controls whether the installation or uninstallation should proceed.

.EXAMPLE
.\Invoke-PrinterInstallation -PrinterConfigPath "d:\path\to\printer.json" -AppConfigPath "d:\path\to\config.json"

Executes the Invoke-PrinterInstallation function using the specified printer and application configuration files. It constructs and displays the install and uninstall commands based on the configurations.

.INPUTS
None. You cannot pipe objects to Invoke-PrinterInstallation.

.OUTPUTS
String. Outputs the constructed install and uninstall commands to the console.

.NOTES
Version:        1.0
Author:         Your Name
Creation Date:  The Date
Purpose/Change: Initial function development

.LINK
URL to more information if available

#>

    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PrinterConfigPath, # Path to printer.json

        [Parameter(Mandatory = $true)]
        [string]$AppConfigPath  # Path to config.json
    )

    Begin {
        Write-EnhancedLog -Message "Starting Invoke-PrinterInstallation" -Level "INFO"
    }

    Process {
        try {
            if (-not (Test-Path -Path $PrinterConfigPath)) {
                Write-EnhancedLog -Message "Printer configuration file not found at path: $PrinterConfigPath" -Level "ERROR"
                throw "Printer configuration file not found."
            }

            if (-not (Test-Path -Path $AppConfigPath)) {
                Write-EnhancedLog -Message "Application configuration file not found at path: $AppConfigPath" -Level "ERROR"
                throw "Application configuration file not found."
            }

            $appConfig = Get-Content -Path $AppConfigPath -Raw | ConvertFrom-Json

            if ($appConfig.PrinterInstall -eq $true) {
                $printerConfig = Get-Content -Path $PrinterConfigPath -Raw | ConvertFrom-Json

                $InstallCommandLine = "%SystemRoot%\sysnative\WindowsPowerShell\v1.0\powershell.exe -windowstyle hidden -executionpolicy bypass -File ""install.ps1"""
                $UninstallCommandLine = "%SystemRoot%\sysnative\WindowsPowerShell\v1.0\powershell.exe -windowstyle hidden -executionpolicy bypass -File ""uninstall.ps1"""

                $printerConfig.psobject.Properties | ForEach-Object {
                    $InstallCommandLine += " -$($_.Name) `"$($_.Value)`""
                    $UninstallCommandLine += " -$($_.Name) `"$($_.Value)`""
                }

                Write-EnhancedLog -Message "Install and Uninstall command lines constructed successfully" -Level "VERBOSE"

                # Return a custom object containing both commands
                $commands = [PSCustomObject]@{
                    InstallCommand   = $InstallCommandLine
                    UninstallCommand = $UninstallCommandLine
                }

                return $commands

            }
            else {
                Write-EnhancedLog -Message "PrinterInstall is not set to true in the application configuration. No commands will be executed." -Level "WARNING"
            }

        }
        catch {
            Write-EnhancedLog -Message "An error occurred: $_" -Level "ERROR"
        }
    }

    End {
        Write-EnhancedLog -Message "Invoke-PrinterInstallation completed" -Level "INFO"
    }
}


# # Define paths to the configuration files
# $printerConfigPath = Join-Path -Path $PSScriptRoot -ChildPath "printer.json"
# $appConfigPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"

# Invoke-PrinterInstallation -PrinterConfigPath $printerConfigPath -AppConfigPath $appConfigPath


function Remove-IntuneWinFiles {


    <#
.SYNOPSIS
    Removes all *.intuneWin files from a specified directory.

.DESCRIPTION
    This function searches for all files with the .intuneWin extension
    in the specified directory and removes them. It logs actions taken
    and any errors encountered using the Write-EnhancedLog function.

.PARAMETER DirectoryPath
    The path to the directory from which *.intuneWin files will be removed.

.EXAMPLE
    Remove-IntuneWinFiles -DirectoryPath "d:\Users\aollivierre\AppData\Local\Intune-Win32-Deployer\apps-winget"
    Removes all *.intuneWin files from the specified directory and logs the actions.

.NOTES
    Ensure you have the necessary permissions to delete files in the specified directory.

#>

    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath
    )

    process {
        Write-EnhancedLog -Message "Starting to remove *.intuneWin files from $DirectoryPath recursively." -Level "INFO"

        try {
            # Include -Recurse to search within all subdirectories
            $files = Get-ChildItem -Path $DirectoryPath -Filter "*.intuneWin" -Recurse -ErrorAction Stop

            if ($files.Count -eq 0) {
                Write-EnhancedLog -Message "No *.intuneWin files found in $DirectoryPath." -Level "INFO"
            }
            else {
                foreach ($file in $files) {
                    Remove-Item $file.FullName -Force -ErrorAction Stop
                    Write-EnhancedLog -Message "Removed file: $($file.FullName)" -Level "INFO"
                }
            }
        }
        catch {
            Write-EnhancedLog -Message "Error removing *.intuneWin files: $_" -Level "ERROR"
            throw $_  # Optionally re-throw the error to handle it further up the call stack.
        }

        Write-EnhancedLog -Message "Completed removal of *.intuneWin files from $DirectoryPath recursively." -Level "INFO"
    }
}


function Upload-Win32App {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Prg,

        [Parameter(Mandatory = $true)]
        [string]$Prg_Path,

        [string]$Prg_img,

        [string]$Win32AppsRootPath,

        [string]$linetoadd,

        [Parameter(Mandatory = $true)]
        [pscustomobject]$config
    )

    Write-EnhancedLog -Message "Entering Upload-Win32App" -Level "WARNING"
    Write-EnhancedLog -Message "Uploading: $($Prg.name)" -Level "WARNING"

    $InstallCommandLines = Set-InstallCommandLine -config $config
    Log-Params -Params @{
        Prg      = $Prg
        Prg_Path = $Prg_Path
        Prg_img  = $Prg_img
    }

    $paths = Prepare-Paths -Prg $Prg -Prg_Path $Prg_Path -Win32AppsRootPath $Win32AppsRootPath
    $IntuneWinFile = Create-IntuneWinPackage -Prg $Prg -Prg_Path $Prg_Path -destinationPath $paths.destinationPath

    Upload-IntuneWinPackage -Prg $Prg -Prg_Path $Prg_Path -Prg_img $Prg_img -config $config -IntuneWinFile $IntuneWinFile -InstallCommandLine $InstallCommandLines.InstallCommandLine -UninstallCommandLine $InstallCommandLines.UninstallCommandLine
    # Start-Sleep -Seconds 10

    # Write-EnhancedLog -Message "Calling Create-AADGroup for $($Prg.name)" -Level "WARNING"
    # Create-AADGroup -Prg $Prg
    # Write-EnhancedLog -Message "Completed Create-AADGroup for $($Prg.name)" -Level "INFO"
}

function Set-InstallCommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$config
    )

    if ($config.serviceUIPSADT -eq $true) {
        $InstallCommandLine = "ServiceUI.exe -process:explorer.exe Deploy-Application.exe -DeploymentType install"
        $UninstallCommandLine = "ServiceUI.exe -process:explorer.exe Deploy-Application.exe -DeploymentType Uninstall"
    }
    elseif ($config.PSADT -eq $true) {
        $InstallCommandLine = "Deploy-Application.exe -DeploymentType install -DeployMode Silent"
        $UninstallCommandLine = "Deploy-Application.exe -DeploymentType Uninstall -DeployMode Silent"
    }
    else {
        $InstallCommandLine = "%SystemRoot%\sysnative\WindowsPowerShell\v1.0\powershell.exe -windowstyle hidden -executionpolicy bypass -command .\install.ps1"
        $UninstallCommandLine = "%SystemRoot%\sysnative\WindowsPowerShell\v1.0\powershell.exe -windowstyle hidden -executionpolicy bypass -command .\uninstall.ps1"
    }

    return @{
        InstallCommandLine   = $InstallCommandLine
        UninstallCommandLine = $UninstallCommandLine
    }
}

function Prepare-Paths {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Prg,

        [Parameter(Mandatory = $true)]
        [string]$Prg_Path,

        [Parameter(Mandatory = $true)]
        [string]$Win32AppsRootPath
    )

    if (-not (Test-Path -Path $Prg_Path)) {
        Write-EnhancedLog -Message "Source path $Prg_Path does not exist. Creating it." -Level "INFO"
        New-Item -Path $Prg_Path -ItemType Directory -Force
    }
    
    $destinationRootPath = Join-Path -Path $Win32AppsRootPath -ChildPath "Win32Apps"
    if (-not (Test-Path -Path $destinationRootPath)) {
        New-Item -Path $destinationRootPath -ItemType Directory -Force
    }

    $destinationPath = Join-Path -Path $destinationRootPath -ChildPath $Prg.name
    if (-not (Test-Path -Path $destinationPath)) {
        New-Item -Path $destinationPath -ItemType Directory -Force
    }

    Write-EnhancedLog -Message "Destination path created: $destinationPath" -Level "INFO"
    return @{
        destinationPath = $destinationPath
    }
}

function Create-IntuneWinPackage {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Prg,

        [Parameter(Mandatory = $true)]
        [string]$Prg_Path,

        [Parameter(Mandatory = $true)]
        [string]$destinationPath
    )
    try {
        Write-EnhancedLog -Message "Creating .intunewin package..." -Level "INFO"

        $setupFile = "install.ps1"
        # $Win32AppPackage = New-IntuneWin32AppPackage -SourceFolder $Prg_Path -SetupFile $setupFile -OutputFolder $destinationPath -Verbose -Force:$true

        # using New-IntuneWinPackage instead of New-IntuneWin32AppPackage because it creates a .intunewin file in a cross-platform way both on Windows and Linux
        New-IntuneWinPackage -SourcePath $Prg_Path -DestinationPath $destinationPath -SetupFile $setupFile -Verbose
        # Write-Host "Package creation completed successfully."
        Write-EnhancedLog -Message "Package creation completed successfully." -Level "INFO"

        $IntuneWinFile = Join-Path -Path $destinationPath -ChildPath "install.intunewin"
        
        # $IntuneWinFile = $Win32AppPackage.Path
        Write-EnhancedLog -Message "IntuneWinFile path set: $IntuneWinFile" -Level "INFO"
        return $IntuneWinFile
    }
    catch {
        Write-EnhancedLog -Message "Error creating .intunewin package: $_" -Level "ERROR"
        Write-Host "Error creating .intunewin package: $_"
        exit
    }
}

function Upload-IntuneWinPackage {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Prg,

        [Parameter(Mandatory = $true)]
        [string]$Prg_Path,

        [Parameter(Mandatory = $true)]
        [string]$Prg_img,

        [Parameter(Mandatory = $true)]
        [pscustomobject]$config,

        [Parameter(Mandatory = $true)]
        [string]$IntuneWinFile,

        [Parameter(Mandatory = $true)]
        [string]$InstallCommandLine,

        [Parameter(Mandatory = $true)]
        [string]$UninstallCommandLine
    )

    try {
        $DisplayName = "$($Prg.Name)"
        Write-EnhancedLog -Message "DisplayName set: $DisplayName" -Level "INFO"

        $DetectionRule = Create-DetectionRule -Prg_Path $Prg_Path
        $RequirementRule = Create-RequirementRule
        $Icon = Set-AppIcon -Prg_img $Prg_img

        $IntuneAppParams = @{
            FilePath                 = $IntuneWinFile
            Icon                     = $Icon
            DisplayName              = "$DisplayName ($($config.InstallExperience))"
            Description              = "$DisplayName ($($config.InstallExperience))"
            Publisher                = $config.Publisher
            AppVersion               = $config.AppVersion
            Developer                = $config.Developer
            Owner                    = $config.Owner
            CompanyPortalFeaturedApp = [System.Convert]::ToBoolean($config.CompanyPortalFeaturedApp)
            InstallCommandLine       = $InstallCommandLine
            UninstallCommandLine     = $UninstallCommandLine
            InstallExperience        = $config.InstallExperience
            RestartBehavior          = $config.RestartBehavior
            DetectionRule            = $DetectionRule
            RequirementRule          = $RequirementRule
            InformationURL           = $config.InformationURL
            PrivacyURL               = $config.PrivacyURL
        }

        # Log-Params -Params $IntuneAppParams

        # Create a copy of $IntuneAppParams excluding the $Icon
        $IntuneAppParamsForLogging = $IntuneAppParams.Clone()
        $IntuneAppParamsForLogging.Remove('Icon')

        Log-Params -Params $IntuneAppParamsForLogging

        Write-EnhancedLog -Message "Calling Add-IntuneWin32App with IntuneAppParams - in progress" -Level "WARNING"
        $Win32App = Add-IntuneWin32App @IntuneAppParams
        Write-EnhancedLog -Message "Win32 app added successfully. App ID: $($Win32App.id)" -Level "INFO"

        Write-EnhancedLog -Message "Assigning Win32 app to all users..." -Level "WARNING"
        Add-IntuneWin32AppAssignmentAllUsers -ID $Win32App.id -Intent "available" -Notification "showAll" -Verbose
        Write-EnhancedLog -Message "Assignment completed successfully." -Level "INFO"
    }
    catch {
        Write-EnhancedLog -Message "Error during IntuneWin32 app process: $_" -Level "ERROR"
        Write-Host "Error during IntuneWin32 app process: $_"
        exit
    }
}

function Create-DetectionRule {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prg_Path
    )

    Write-EnhancedLog -Message "Creating detection rule..." -Level "WARNING"
    $detectionScriptPath = Join-Path -Path $Prg_Path -ChildPath "check.ps1"
    if (-not (Test-Path -Path $detectionScriptPath)) {
        Write-Warning "Detection rule script file does not exist at path: $detectionScriptPath"
    }
    else {
        $DetectionRule = New-IntuneWin32AppDetectionRuleScript -ScriptFile $detectionScriptPath -EnforceSignatureCheck $false -RunAs32Bit $false
    }
    Write-EnhancedLog -Message "Detection rule set (calling New-IntuneWin32AppDetectionRuleScript) - done" -Level "INFO"

    return $DetectionRule
}

function Create-RequirementRule {
    Write-EnhancedLog -Message "Setting minimum requirements..." -Level "WARNING"
    $RequirementRule = New-IntuneWin32AppRequirementRule -Architecture "x64" -MinimumSupportedWindowsRelease "W10_1607"
    Write-EnhancedLog -Message "Minimum requirements set - done" -Level "INFO"

    return $RequirementRule
}

function Set-AppIcon {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prg_img
    )

    $Icon = New-IntuneWin32AppIcon -FilePath $Prg_img
    Write-EnhancedLog -Message "App icon set - done" -Level "INFO"

    return $Icon
}









# try {
#     Ensure-LoggingFunctionExists
#     # Continue with the rest of the script here
#     # exit
# }
# catch {
#     Write-Host "Critical error: $_"
#     exit
# }

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

# Setup logging
# Write-EnhancedLog -Message "Script Started" -Level "INFO"

################################################################################################################################
################################################################################################################################
################################################################################################################################

# Execute InstallAndImportModulesPSGallery function
# InstallAndImportModulesPSGallery -moduleJsonPath "$localScriptRoot/modules.json"

################################################################################################################################
################################################ END MODULE CHECKING ###########################################################
################################################################################################################################

    
################################################################################################################################
################################################ END LOGGING ###################################################################
################################################################################################################################

#  Define the variables to be used for the function
#  $PSADTdownloadParams = @{
#      GithubRepository     = "psappdeploytoolkit/psappdeploytoolkit"
#      FilenamePatternMatch = "PSAppDeployToolkit*.zip"
#      ZipExtractionPath    = Join-Path "$PSScriptRoot\private" "PSAppDeployToolkit"
#  }

#  Call the function with the variables
#  Download-PSAppDeployToolkit @PSADTdownloadParams

################################################################################################################################
################################################ END DOWNLOADING PSADT #########################################################
################################################################################################################################


##########################################################################################################################
############################################STARTING THE MAIN FUNCTION LOGIC HERE#########################################
##########################################################################################################################


################################################################################################################################
################################################ START GRAPH CONNECTING ########################################################
################################################################################################################################
# $accessToken = Connect-GraphWithCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword

# Log-Params -Params @{accessToken = $accessToken }

# Get-TenantDetails
#################################################################################################################################
################################################# END Connecting to Graph #######################################################
#################################################################################################################################





# Variables
$endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
$startDate = (Get-Date).AddDays(-30).ToString("yyyy-MM-ddTHH:mm:ssZ")
$baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
$headers = @{
    "Authorization" = "Bearer $accessToken"
    "Content-Type"  = "application/json"
}
# Initial URL with filters
$url = "$baseUrl`?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"
$intuneUrl = "https://graph.microsoft.com/v1.0/deviceManagement/managedDevices"
$tenantDetailsUrl = "https://graph.microsoft.com/v1.0/organization"

# Log initial parameters
$params = @{
    AccessToken = $accessToken
    EndDate     = $endDate
    StartDate   = $startDate
    BaseUrl     = $baseUrl
    Url         = $url
    IntuneUrl   = $intuneUrl
    TenantUrl   = $tenantDetailsUrl
}
# Log-Params -Params $params

# Function to make the API request and handle pagination


# Validate access to required URIs (uncomment when debugging)
# $uris = @($url, $intuneUrl, $tenantDetailsUrl)
# foreach ($uri in $uris) {
#     if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
#         Write-EnhancedLog "Validation failed. Halting script." -Color Red
#         exit 1
#     }
# }


# (we got them already now let's filter them)

# # Get all sign-in logs for the last 30 days
# $signInLogs = Get-SignInLogs -url $url -Headers $headers

# # Export to JSON for further processing
# $signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

# Write-EnhancedLog "Export complete. Check /usr/src/SignInLogs.json for results." -Color Green

# # Load the sign-in logs
# $json = Get-Content -Path '/usr/src/SignInLogs.json' | ConvertFrom-Json




# Define the root path where the scripts and exports are located
# $scriptRoot = "C:\MyScripts"

# Optionally, specify the names for the exports folder and subfolder
$exportsFolderName = "CustomExports"
$exportSubFolderName = "CustomSignInLogs"

# # Call the function to export sign-in logs to XML (and other formats)
# # Define the parameters to be splatted
$ExportAndProcessSignInLogsparams = @{
    # ScriptRoot          = $PSscriptRoot
    ScriptRoot          = $localScriptRoot
    ExportsFolderName   = $exportsFolderName
    ExportSubFolderName = $exportSubFolderName
    url                 = $url
    Headers             = $headers
}

# Call the function with splatted parameters
# Export-SignInLogs @params


# ExportAndProcessSignInLogs -ScriptRoot $PSscriptRoot -ExportsFolderName $exportsFolderName -ExportSubFolderName $exportSubFolderName -url $url -Headers $headers
ExportAndProcessSignInLogs @ExportAndProcessSignInLogsparams



# # Export to JSON for further processing
# $signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

# Write-EnhancedLog "Export complete. Check /usr/src/SignInLogs.json for results." -Color Green


# Write-EnhancedLog "Export complete. Check ""$PSscriptRoot/$exportsFolderName/$exportSubFolderName/SignInLogs_20240610143318.json"" for results." -Color Green


# # # Load the sign-in logs
# $json = Get-Content -Path "$PSscriptRoot/$exportsFolderName/$exportSubFolderName/SignInLogs_20240610143318.json" | ConvertFrom-Json





# $allSKUs = Get-MgSubscribedSku -Property SkuPartNumber, ServicePlans 
# $allSKUs | ForEach-Object {
#     Write-Host "Service Plan:" $_.SkuPartNumber
#     $_.ServicePlans | ForEach-Object {$_}
# }


















  

    # Initialize results in each runspace
    $localContext = Initialize-Results

    if ($_.userDisplayName -eq "On-Premises Directory Synchronization Service Account") {
        return
    }

    try {
        Process-DeviceItem -Item $_ -Context $localContext -Headers $using:Headers
    } catch {
        Handle-Error -ErrorRecord $_
    }

    return $localContext.Results
} -ThrottleLimit 10

return $results

}

# $JSON is being passed from ExportAndProcessSignInLogs.ps1
# $results = Process-AllDevices -Json $Json -Headers $Headers
$results = Process-AllDevicesParallel -Json $signInLogs -Headers $Headers -ScriptRoot $PSScriptRoot


# class SignInLog {
#     [string] $userDisplayName
#     [string] $deviceID
#     # [datetime] $signInDateTime
#     # Add other relevant properties here
# }



# Read JSON content from file
# $JsonFilePath = $json
# $JsonContent = Get-Content -Path $JsonFilePath | Out-String

# Example call to the function
# $Headers = @{"Authorization" = "Bearer <your_token>"}



# class ProcInfo {
#     [string] $Name
#     [int] $Id
# }

# ps | Select name, id | convertto-json | set-content test.json
# $reader = [System.IO.StreamReader]::new("$json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]').ToObject[SignInLog]()



# $results = Process-AllDevices -Json $json -Headers $Headers


# Output results
# $results | Format-Table -AutoSize




# Output results
# $results | Format-Table -AutoSize #(uncomment to view results)

# $results | Out-GridView -Title 'Corporate VS BYOD Devices - Compliant VS Non-Compliant'






# Export master report
$results | Export-Csv "$PSScriptRoot/$exportsFolderName/MasterReport.csv" -NoTypeInformation


# Exclude PII Removed entries
$filteredResults = $results | Where-Object { $_.DeviceStateInIntune -ne 'External' }

# Generate and export specific reports
$report1 = $filteredResults | Where-Object { ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Compliant' }
$report2 = $filteredResults | Where-Object { ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Non-Compliant' }
$report3 = $filteredResults | Where-Object { $_.TrustType -eq 'Azure AD Registered' -and $_.DeviceComplianceStatus -eq 'Compliant' }
$report4 = $filteredResults | Where-Object { $_.TrustType -eq 'Azure AD Registered' -and $_.DeviceComplianceStatus -eq 'Non-Compliant' }

$report1 | Export-Csv "$PSScriptRoot/$exportsFolderName/CorporateCompliant.csv" -NoTypeInformation
$report2 | Export-Csv "$PSScriptRoot/$exportsFolderName/CorporateIncompliant.csv" -NoTypeInformation
$report3 | Export-Csv "$PSScriptRoot/$exportsFolderName/BYODCompliant.csv" -NoTypeInformation
$report4 | Export-Csv "$PSScriptRoot/$exportsFolderName/BYOD_AND_CORP_ER_Incompliant.csv" -NoTypeInformation

# Output color-coded stats to console
$totalMaster = $results.Count
$totalReport1 = $report1.Count
$totalReport2 = $report2.Count
$totalReport3 = $report3.Count
$totalReport4 = $report4.Count

Write-EnhancedLog "Total entries in Master Report: $totalMaster" -Color Green
Write-EnhancedLog "Total entries in Corporate Compliant Report: $totalReport1" -Color Cyan
Write-EnhancedLog "Total entries in Corporate Incompliant Report: $totalReport2" -Color Yellow
Write-EnhancedLog "Total entries in BYOD Compliant Report: $totalReport3" -Color Blue
Write-EnhancedLog "Total entries in BYOD and CORP Entra Registered Incompliant Report: $totalReport4" -Color Red




# Group data by compliance status, trust type, device OS, and device state in Intune
$groupedData = $filteredResults | Group-Object -Property DeviceComplianceStatus, TrustType, DeviceOS, DeviceStateInIntune

# Initialize a new List to store the structured data
$structuredData = [System.Collections.Generic.List[PSCustomObject]]::new()

foreach ($group in $groupedData) {
    $complianceStatus = $group.Name.Split(',')[0].Trim()
    $trustType = $group.Name.Split(',')[1].Trim()
    $deviceOS = $group.Name.Split(',')[2].Trim()
    $deviceStateInIntune = $group.Name.Split(',')[3].Trim()
    $count = $group.Count

    $structuredData.Add([PSCustomObject]@{
            ComplianceStatus    = $complianceStatus
            TrustType           = $trustType
            DeviceOS            = $deviceOS
            DeviceStateInIntune = $deviceStateInIntune
            Count               = $count
        })
}

# Export the structured data to a CSV file
$structuredData | Export-Csv "$PSScriptRoot/$exportsFolderName/StructuredReport.csv" -NoTypeInformation

# Output structured data to console with color coding
foreach ($item in $structuredData) {
    $Level = switch ($item.ComplianceStatus) {
        "Compliant" { "Info" }
        "Non-Compliant" { "Warning" }
        default { "White" }
    }

    $Level = switch ($item.DeviceStateInIntune) {
        "Error" { "Error" }
        "Present" { "Notice" }
        default { "White" }
    }

  

    Write-EnhancedLog -Message "Compliance Status: $($item.ComplianceStatus), Trust Type: $($item.TrustType), Device OS: $($item.DeviceOS), Count: $($item.Count)" -Level $Level
}

# Additional grouping and exporting as needed
$reportCompliant = $structuredData | Where-Object { $_.ComplianceStatus -eq 'Compliant' }
$reportNonCompliant = $structuredData | Where-Object { $_.ComplianceStatus -eq 'Non-Compliant' }
$reportPresent = $structuredData | Where-Object { $_.DeviceStateInIntune -eq 'Present' }
$reportError = $structuredData | Where-Object { $_.DeviceStateInIntune -eq 'Error' }

$reportCompliant | Export-Csv "$PSScriptRoot/$exportsFolderName/Report_Compliant.csv" -NoTypeInformation
$reportNonCompliant | Export-Csv "$PSScriptRoot/$exportsFolderName/Report_NonCompliant.csv" -NoTypeInformation
$reportPresent | Export-Csv "$PSScriptRoot/$exportsFolderName/Report_Present.csv" -NoTypeInformation
$reportError | Export-Csv "$PSScriptRoot/$exportsFolderName/Report_Error.csv" -NoTypeInformation

# Export report for External devices separately
$reportExternal = $results | Where-Object { $_.DeviceStateInIntune -eq 'External' }
$reportExternal | Export-Csv "$PSScriptRoot/$exportsFolderName/Report_External.csv" -NoTypeInformation



Generate-LicenseReports -Results $results -PSScriptRoot $PSScriptRoot -ExportsFolderName $ExportsFolderName
Generate-PII-RemovedReport -Results $results -PSScriptRoot $PSScriptRoot -ExportsFolderName $ExportsFolderName