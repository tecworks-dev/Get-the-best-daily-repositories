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
# $CertThumbprint = $secrets.CertThumbprint

$certPath = Join-Path -Path $PSScriptRoot -ChildPath 'graphcert.pfx'
$CertPassword = $secrets.CertPassword

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


# Auxiliary function to detect OS and set the Modules folder path
function Get-ModulesFolderPath {
    if ($PSVersionTable.Platform -eq 'Win32NT') {
        return "C:\code\modules"
    } elseif ($PSVersionTable.Platform -eq 'Unix') {
        return "/usr/src/modules"
    } else {
        throw "Unsupported operating system"
    }
}

# Store the outcome in $ModulesFolderPath
try {
    $ModulesFolderPath = Get-ModulesFolderPath
    Write-host "Modules folder path: $ModulesFolderPath"
} catch {
    Write-Error $_.Exception.Message
}



function Get-ModulesScriptPathsAndVariables {   
    param (
        [string]$BaseDirectory,
        $ModulesFolderPath
    )

    try {
        # $ModulesFolderPath = Join-Path -Path $BaseDirectory -ChildPath "Modules"
        
        
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
$DotSourcinginitializationInfo = Get-ModulesScriptPathsAndVariables -BaseDirectory $PSScriptRoot -ModulesFolderPath $ModulesFolderPath

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

Import-LatestModulesLocalRepository -ModulesFolderPath $ModulesFolderPath


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

# Setup logging
Write-EnhancedLog -Message "Script Started" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)

# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################

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

# Execute InstallAndImportModulesPSGallery function
InstallAndImportModulesPSGallery

# ################################################################################################################################
# ################################################################################################################################
Connect-GraphWithCert -tenantId $tenantId -clientId $clientId -certPath $certPath -certPassword $certPassword

Get-TenantDetails

# ################################################################################################################################
# ################################################################################################################################
