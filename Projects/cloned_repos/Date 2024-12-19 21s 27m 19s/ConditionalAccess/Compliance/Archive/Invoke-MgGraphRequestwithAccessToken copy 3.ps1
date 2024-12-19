# Read configuration from the JSON file
# Assign values from JSON to variables


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

# Read configuration from the JSON file
$configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
$env:MYMODULE_CONFIG_PATH = $configPath



# # Load client secrets from the JSON file
$secretsjsonPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"
$secrets = Get-Content -Path $secretsjsonPath | ConvertFrom-Json

# # Variables from JSON file
$tenantId = $secrets.tenantId
$clientId = $secrets.clientId
# $clientSecret = $secrets.clientSecret
$CertThumbprint = $secrets.CertThumbprint


# $config = Get-Content -Path $configPath -Raw | ConvertFrom-Json

# Assign values from JSON to variables
# $PackageName = $config.PackageName
# $PackageUniqueGUID = $config.PackageUniqueGUID
# $Version = $config.Version
# $PackageExecutionContext = $config.PackageExecutionContext
# $RepetitionInterval = $config.RepetitionInterval
# $ScriptMode = $config.ScriptMode




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

function Get-ModulesScriptPathsAndVariables {
    param (
        [string]$BaseDirectory
    )

    try {
        $ModulesFolderPath = Join-Path -Path $BaseDirectory -ChildPath "Modules"
        
        if (-not (Test-Path -Path $ModulesFolderPath)) {
            throw "Modules folder path does not exist: $ModulesFolderPath"
        }

        # Construct and return a PSCustomObject
        return [PSCustomObject]@{
            BaseDirectory     = $BaseDirectory
            ModulesFolderPath = $ModulesFolderPath
        }
    }
    catch {
        Write-Host "Error in finding Modules script files: $_" -ForegroundColor Red
        # Optionally, you could return a PSCustomObject indicating an error state
        # return [PSCustomObject]@{ Error = $_.Exception.Message }
    }
}



# Retrieve script paths and related variables
$DotSourcinginitializationInfo = Get-ModulesScriptPathsAndVariables -BaseDirectory $PSScriptRoot

# $DotSourcinginitializationInfo
$DotSourcinginitializationInfo | Format-List


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
                Import-Module $ModulePath -ErrorAction Stop
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




# Example of how to use the function
# # $PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
# $EnhancedLoggingAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedLoggingAO\3.0.0\EnhancedLoggingAO.psm1"


# # Call the function to import the module with retry logic
# Import-ModuleWithRetry -ModulePath $EnhancedLoggingAO

# Import-Module "E:\Code\CB\Entra\ARH\Private\EnhancedGraphAO\3.0.0\EnhancedGraphAO.psm1" -Verbose





# # Get the path to the EnhancedLoggingAO module directory
# $moduleDir = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedLoggingAO"

# # Get the latest version directory
# $latestVersion = Get-ChildItem -Path $moduleDir -Directory | Sort-Object Name -Descending | Select-Object -First 1

# if ($null -eq $latestVersion) {
#     Write-EnhancedLog -Message "No version directories found for EnhancedLoggingAO module." -Level "ERROR"
#     throw "No version directories found for EnhancedLoggingAO module."
# }

# # Construct the path to the module file
# $EnhancedLoggingAO = Join-Path -Path $latestVersion.FullName -ChildPath "EnhancedLoggingAO.psm1"

# # Call the function to import the module with retry logic
# Import-ModuleWithRetry -ModulePath $EnhancedLoggingAO





# # Get the path to the Modules directory
# $modulesDir = Join-Path -Path $PSScriptRoot -ChildPath "Modules"

# # Get all module directories
# $moduleDirectories = Get-ChildItem -Path $modulesDir -Directory

# foreach ($moduleDir in $moduleDirectories) {
#     # Get the latest version directory for the current module
#     $latestVersionDir = Get-ChildItem -Path $moduleDir.FullName -Directory | Sort-Object Name -Descending | Select-Object -First 1

#     if ($null -eq $latestVersionDir) {
#         Write-EnhancedLog -Message "No version directories found for module: $($moduleDir.Name)" -Level "ERROR"
#         continue
#     }

#     # Construct the path to the module file
#     $modulePath = Join-Path -Path $latestVersionDir.FullName -ChildPath "$($moduleDir.Name).psm1"

#     # Check if the module file exists
#     if (Test-Path -Path $modulePath) {
#         # Import the module with retry logic
#         try {
#             Import-ModuleWithRetry -ModulePath $modulePath
#             Write-EnhancedLog -Message "Successfully imported module: $($moduleDir.Name) from version: $($latestVersionDir.Name)" -Level "INFO"
#         }
#         catch {
#             Write-EnhancedLog -Message "Failed to import module: $($moduleDir.Name) from version: $($latestVersionDir.Name). Error: $_" -Level "ERROR"
#         }
#     }
#     else {
#         Write-EnhancedLog -Message "Module file not found: $modulePath" -Level "ERROR"
#     }
# }










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
    param ()

    Begin {
        # Get the path to the Modules directory
        $modulesDir = Join-Path -Path $PSScriptRoot -ChildPath "Modules"

        # Get all module directories
        $moduleDirectories = Get-ChildItem -Path $modulesDir -Directory

        # Log the number of discovered module directories
        write-host "Discovered module directories: $($moduleDirectories.Count)"  -ForegroundColor ([ConsoleColor]::Cyan)
    }

    Process {
        foreach ($moduleDir in $moduleDirectories) {
            # Get the latest version directory for the current module
            $latestVersionDir = Get-ChildItem -Path $moduleDir.FullName -Directory | Sort-Object Name -Descending | Select-Object -First 1

            if ($null -eq $latestVersionDir) {
                write-host "No version directories found for module: $($moduleDir.Name)" -ForegroundColor ([ConsoleColor]::Red)
                continue
            }

            # Construct the path to the module file
            $modulePath = Join-Path -Path $latestVersionDir.FullName -ChildPath "$($moduleDir.Name).psm1"

            # Check if the module file exists
            if (Test-Path -Path $modulePath) {
                # Import the module with retry logic
                try {
                    Import-ModuleWithRetry -ModulePath $modulePath
                    write-host "Successfully imported module: $($moduleDir.Name) from version: $($latestVersionDir.Name)"  -ForegroundColor ([ConsoleColor]::Green)
                }
                catch {
                    write-host "Failed to import module: $($moduleDir.Name) from version: $($latestVersionDir.Name). Error: $_"  -ForegroundColor ([ConsoleColor]::Red)
                }
            }
            else {
                write-host  "Module file not found: $modulePath" -ForegroundColor ([ConsoleColor]::Red)
            }
        }
    }

    End {
        write-host "Module import process completed using Import-LatestModulesLocalRepository from $modulesDir" -ForegroundColor ([ConsoleColor]::Cyan)
    }
}

Import-LatestModulesLocalRepository


# Auxiliary Function: Import-ModuleWithRetry
# function Import-ModuleWithRetry {
#     param (
#         [Parameter(Mandatory=$true)]
#         [string]$ModulePath
#     )

#     $retryCount = 3
#     $retryInterval = 5

#     for ($i = 1; $i -le $retryCount; $i++) {
#         try {
#             Import-Module -Name $ModulePath -Force -ErrorAction Stop
#             Write-EnhancedLog -Message "Imported module on attempt $i: $ModulePath" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)
#             return
#         }
#         catch {
#             Write-EnhancedLog -Message "Attempt $i failed to import module: $ModulePath. Error: $_" -Level "WARNING" -ForegroundColor ([ConsoleColor]::Yellow)
#             Start-Sleep -Seconds $retryInterval
#         }
#     }

#     throw "Failed to import module after $retryCount attempts: $ModulePath"
# }










# ################################################################################################################################
# ################################################ END MODULE LOADING ############################################################
# ################################################################################################################################



function Ensure-LoggingFunctionExists {
    if (Get-Command Write-EnhancedLog -ErrorAction SilentlyContinue) {
        Write-EnhancedLog -Message "Logging works" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)
    }
    else {
        throw "Write-EnhancedLog function not found. Terminating script."
    }
}

# Usage
try {
    Ensure-LoggingFunctionExists
    # Continue with the rest of the script here
    # exit
}
catch {
    Write-Host "Critical error: $_" -ForegroundColor Red
    exit
}

# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################

# Setup logging
Write-EnhancedLog -Message "Script Started" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)

# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


# $EnhancedGraphAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedGraphAO\3.0.0\EnhancedGraphAO.psm1"
# Import-ModuleWithRetry -ModulePath $EnhancedGraphAO



# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


# function Install-RequiredModules {

#     [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

#     # $requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Authentication")
#     $requiredModules = @("Microsoft.Graph.Authentication")

#     foreach ($module in $requiredModules) {
#         if (!(Get-Module -ListAvailable -Name $module)) {

#             Write-EnhancedLog -Message "Installing module: $module" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
#             Install-Module -Name $module -Force
#             Write-EnhancedLog -Message "Module: $module has been installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
#         }
#         else {
#             Write-EnhancedLog -Message "Module $module is already installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
#         }
#     }


#     $ImportedModules = @("Microsoft.Graph.Identity.DirectoryManagement", "Microsoft.Graph.Authentication")
    
#     foreach ($Importedmodule in $ImportedModules) {
#         if ((Get-Module -ListAvailable -Name $Importedmodule)) {
#             Write-EnhancedLog -Message "Importing module: $Importedmodule" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
#             Import-Module -Name $Importedmodule
#             Write-EnhancedLog -Message "Module: $Importedmodule has been Imported" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
#         }
#     }


# }
# # Call the function to install the required modules and dependencies
# Install-RequiredModules
# Write-EnhancedLog -Message "All modules installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)





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
    $moduleJsonPath = "$PSScriptRoot/modules.json"
    
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

            Write-EnhancedLog -Message "Modules installed and imported successfully." -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)
        }
        catch {
            Write-EnhancedLog -Message "Error processing modules.json: $_" -Level "ERROR" -ForegroundColor ([ConsoleColor]::Red)
        }
    }
    else {
        Write-EnhancedLog -Message "modules.json file not found." -Level "ERROR" -ForegroundColor ([ConsoleColor]::Red)
    }
}

# Auxiliary Functions
function Install-Modules {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )
    
    foreach ($module in $Modules) {
        if (-not (Get-Module -ListAvailable -Name $module)) {
            Install-Module -Name $module -Force
            Write-EnhancedLog -Message "Module '$module' installed." -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)
        }
        else {
            Write-EnhancedLog -Message "Module '$module' is already installed." -Level "INFO" -ForegroundColor ([ConsoleColor]::Yellow)
        }
    }
}

function Import-Modules {
    param (
        [Parameter(Mandatory = $true)]
        [string[]]$Modules
    )
    
    foreach ($module in $Modules) {
        if (Get-Module -ListAvailable -Name $module) {
            Import-Module -Name $module -Force
            Write-EnhancedLog -Message "Module '$module' imported." -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)
        }
        else {
            Write-EnhancedLog -Message "Module '$module' not found. Cannot import." -Level "ERROR" -ForegroundColor ([ConsoleColor]::Red)
        }
    }
}




# Execute main function
# Main
InstallAndImportModulesPSGallery





# ################################################################################################################################
# ################################################################################################################################

# Define variables
# $ClientId = '8230c33e-ff30-419c-a1fc-4caf98f069c9'
# $TenantId = 'b5dae566-ad8f-44e1-9929-5669f1dbb343'
# $CertThumbprint = '9B69D19C97BCE75B4208FDE6B2A4A53141628057'

# Example call to the function
# $accessToken = Get-MsGraphAccessTokenCert -tenantId $tenantId -clientId $clientId -certThumbprint $certThumbprint

# Log-Params -Params {
# $accessToken = $accessToken }


Write-EnhancedLog -message 'calling Connect-MgGraph with access token' -Level 'INFO' -ForegroundColor ([ConsoleColor]::Green)



# Define the splat
$GraphParams = @{
    ClientId              = $ClientId
    TenantId              = $TenantId
    CertificateThumbprint = $CertThumbprint
}

# Log the parameters
Log-Params -Params $GraphParams

# Connect to Microsoft Graph
Connect-MgGraph @GraphParams


# Write-Host 'calling Invoke-MgGraphRequest against the Graph API after calling connect-mggraph'

# # # Call the Microsoft Graph API to get organization details
# # $uri = "https://graph.microsoft.com/v1.0/organization"
# # $headers = @{ Authorization = "Bearer $accessToken" }

# $response = Invoke-WebRequest -Uri $uri -Headers $headers -Method Get
# # $responseContent = $response.Content | ConvertFrom-Json

# # # Output the response
# # $responseContent

# # # Parse the JSON response
# # $organizationDetails = $responseContent | ConvertFrom-Json

# # # Extract the required details
# # $tenantDetails = $organizationDetails.value[0]
# # $tenantName = $tenantDetails.DisplayName
# # $tenantId = $tenantDetails.Id
# # $tenantDomain = $tenantDetails.VerifiedDomains[0].Name


# Fetch organization details using the Graph cmdlet
$organization = Get-MgOrganization

# Output the organization details
# $organization | Format-List

# # # Output the extracted details
# # Write-Output "Tenant Name: $tenantName"
# # Write-Output "Tenant ID: $tenantId"
# # Write-Output "Tenant Domain: $tenantDomain"

# Disconnect-MgGraph



# Extract the required details
# $organization = $organization[0]   
$tenantName = $organization.DisplayName
$tenantId = $organization.Id
$tenantDomain = $organization.VerifiedDomains[0].Name


# # Output tenant summary
Write-EnhancedLog -Message "Tenant Name: $tenantName" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
Write-EnhancedLog -Message "Tenant ID: $tenantId" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
Write-EnhancedLog -Message "Tenant Domain: $tenantDomain" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)