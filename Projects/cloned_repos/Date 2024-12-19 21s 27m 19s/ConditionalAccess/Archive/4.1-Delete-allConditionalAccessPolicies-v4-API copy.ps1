

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

# Find any PFX file in the root directory of the script
$pfxFiles = Get-ChildItem -Path $PSScriptRoot -Filter *.pfx

if ($pfxFiles.Count -eq 0) {
    Write-Error "No PFX file found in the root directory."
    throw "No PFX file found"
}
elseif ($pfxFiles.Count -gt 1) {
    Write-Error "Multiple PFX files found in the root directory. Please ensure there is only one PFX file."
    throw "Multiple PFX files found"
}

# Use the first (and presumably only) PFX file found
$certPath = $pfxFiles[0].FullName

Write-Output "PFX file found: $certPath"

$CertPassword = $secrets.CertPassword

# $DBG

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
            $global:scriptBasePath = $PSScriptRoot
            # $global:modulesBasePath = "$PSScriptRoot\modules"
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
Import-LatestModulesLocalRepository -ModulesFolderPath $ModulesFolderPath -ScriptPath $PSScriptRoot

###############################################################################################################################
############################################### END MODULE LOADING ############################################################
###############################################################################################################################
try {
    # Ensure-LoggingFunctionExists -LoggingFunctionName "# Write-EnhancedLog"
    # Continue with the rest of the script here
    # exit
}
catch {
    Write-Host "Critical error: $_" -ForegroundColor Red
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


$endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
# $startDate = (Get-Date).AddDays(-30).ToString("yyyy-MM-ddTHH:mm:ssZ")
$startDate = (Get-Date).AddDays(-90).ToString("yyyy-MM-ddTHH:mm:ssZ")
# $baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
$headers = @{
    "Authorization" = "Bearer $accessToken"
    "Content-Type"  = "application/json"
}
# Initial URL with filters
# $url = "$baseUrl`?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"
$enterpriseappsURL = "https://graph.microsoft.com/v1.0/applications"
$tenantDetailsUrl = "https://graph.microsoft.com/v1.0/organization"

# Log initial parameters
$params = @{
    AccessToken       = $accessToken
    EndDate           = $endDate
    StartDate         = $startDate
    # BaseUrl           = $baseUrl
    # Url               = $url
    enterpriseappsURL = $enterpriseappsURL
    TenantUrl         = $tenantDetailsUrl
}
Log-Params -Params $params

# Function to make the API request and handle pagination


# Validate access to required URIs
# $uris = @($url, $enterpriseappsURL, $tenantDetailsUrl)
$uris = @($enterpriseappsURL, $tenantDetailsUrl)
# $uris = @($url, $tenantDetailsUrl)
foreach ($uri in $uris) {
    if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
        Write-EnhancedLog "Validation failed. Halting script." -Color Red
        exit 1
    }
}




# Get the tenant details
$tenantDetails = $null
$tenantDetails = Get-TenantDetails
if ($null -eq $tenantDetails) {
    Write-EnhancedLog -Message "Unable to proceed without tenant details" -Level "ERROR"
    throw "Tenant Details name is empty. Cannot proceed without a valid tenant details"
    exit
}

#################################################################################################################################
################################################# END Connecting to Graph #######################################################
#################################################################################################################################

#Endregion #########################################DEALING WITH MODULES########################################################

# Validate certificate creation

# Main script execution
Write-EnhancedLog -Message "Script Started" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)


# Retrieve all Conditional Access policies
$policies = Get-MgBetaIdentityConditionalAccessPolicy

# Save the policies to a CSV file
$csvPath = "C:\Code\CB\Entra\Sandbox\Graph\export\ConditionalAccessPolicies.csv"
$policies | Export-Csv -Path $csvPath -NoTypeInformation

# Count of policies before deletion
$initialPolicyCount = $policies.Count
Write-Output "Number of Conditional Access policies before deletion: $initialPolicyCount"

# Confirm before proceeding with deletion
$confirmation = Read-Host "Are you sure you want to delete all Conditional Access policies? (yes/no)"
if ($confirmation -ne 'yes') {
    Write-Output "Operation aborted by the user."
    Disconnect-MgGraph
    exit
}

# Iterate through the policies and delete each one by ID
foreach ($policy in $policies) {
    $policyId = $policy.Id
    try {
        # Delete the Conditional Access policy by its ID
        Remove-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policyId
        Write-Output "Successfully requested deletion of Conditional Access policy with ID: $policyId"
    } catch {
        Write-Error "Failed to delete Conditional Access policy with ID: $policyId. Error: $_"
    }
}

# Retrieve the remaining policies after deletion
$remainingPolicies = Get-MgBetaIdentityConditionalAccessPolicy
$finalPolicyCount = $remainingPolicies.Count
Write-Output "Number of Conditional Access policies after deletion: $finalPolicyCount"

# Disconnect from Microsoft Graph
# Disconnect-MgGraph

