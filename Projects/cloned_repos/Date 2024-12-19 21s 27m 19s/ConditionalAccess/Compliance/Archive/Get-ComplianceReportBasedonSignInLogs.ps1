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

# $DBG


# Variables

#Todo add flow control to check if 30 days are available and if not then revert back to 7 days. The value must be between 1 and 30
$endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
# $startDate = (Get-Date).AddDays(-30).ToString("yyyy-MM-ddTHH:mm:ssZ")
$startDate = (Get-Date).AddDays(-90).ToString("yyyy-MM-ddTHH:mm:ssZ")
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


# Validate access to required URIs
# $uris = @($url, $intuneUrl, $tenantDetailsUrl)
$uris = @($url, $tenantDetailsUrl)
foreach ($uri in $uris) {
    if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
        Write-EnhancedLog "Validation failed. Halting script." -Color Red
        exit 1
    }
}



# Optionally, specify the names for the exports folder and subfolder
$exportsFolderName = "CustomExports"
$exportSubFolderName = "CustomSignInLogs"



# Define the parameters in a hashtable
$ExportAndProcessSignInLogsparams = @{
    ScriptRoot          = $PSScriptRoot
    ExportsFolderName   = $exportsFolderName
    ExportSubFolderName = $exportSubFolderName
    url                 = $url
    Headers             = $headers
}

# Call the function using splatting


$signInLogs = ExportAndProcessSignInLogs @ExportAndProcessSignInLogsparams

$signInLogs | Out-GridView

# $DBG


# Process the sign-in logs if any are returned
if ($signInLogs.Count -gt 0) {
    # Further processing of $signInLogs can go here...
    Write-Output "Sign-in logs found and processed."
}
else {
    Write-Output "No sign-in logs found."
}

$results = Process-SignInLogs -signInLogs $signInLogs -Headers $Headers


if ($results.Count -gt 0) {
    $results | Export-Csv "$PSScriptRoot/$exportsFolderName/MasterReport.csv" -NoTypeInformation
    $results | Out-GridView
}
else {
    Write-Host "No results to export."
}

# $DBG



# Exclude PII Removed entries
$filteredResults = $results | Where-Object { $_.DeviceStateInIntune -ne 'External' }

# Generate and export specific reports
$corporateCompliantDevices = $filteredResults | Where-Object { 
    ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Compliant' 
}

$corporateNonCompliantDevices = $filteredResults | Where-Object { 
    ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Non-Compliant' 
}

$byodCompliantDevices = $filteredResults | Where-Object { 
    ($_.TrustType -eq 'Azure AD Registered' -or [string]::IsNullOrEmpty($_.TrustType)) -and $_.DeviceComplianceStatus -eq 'Compliant' 
}

$byodAndCorpRegisteredNonCompliantDevices = $filteredResults | Where-Object { 
    ($_.TrustType -eq 'Azure AD Registered' -or [string]::IsNullOrEmpty($_.TrustType)) -and $_.DeviceComplianceStatus -eq 'Non-Compliant' 
}

# Export the reports to CSV
$corporateCompliantDevices | Export-Csv "$PSScriptRoot/$exportsFolderName/CorporateCompliant.csv" -NoTypeInformation
$corporateNonCompliantDevices | Export-Csv "$PSScriptRoot/$exportsFolderName/CorporateIncompliant.csv" -NoTypeInformation
$byodCompliantDevices | Export-Csv "$PSScriptRoot/$exportsFolderName/BYODCompliant.csv" -NoTypeInformation
$byodAndCorpRegisteredNonCompliantDevices | Export-Csv "$PSScriptRoot/$exportsFolderName/BYOD_AND_CORP_ER_Incompliant.csv" -NoTypeInformation

# Output color-coded stats to console
$totalMaster = $results.Count
$totalCorporateCompliant = $corporateCompliantDevices.Count
$totalCorporateNonCompliant = $corporateNonCompliantDevices.Count
$totalBYODCompliant = $byodCompliantDevices.Count
$totalBYODAndCorpRegisteredNonCompliant = $byodAndCorpRegisteredNonCompliantDevices.Count

Write-EnhancedLog "Total entries in Master Report: $totalMaster" -Level 'INFO'
Write-EnhancedLog "Total entries in Corporate Compliant Report: $totalCorporateCompliant" -Level 'INFO'
Write-EnhancedLog "Total entries in Corporate Non-Compliant Report: $totalCorporateNonCompliant" -Level 'DEBUG'
Write-EnhancedLog "Total entries in BYOD Compliant Report: $totalBYODCompliant" -Level 'DEBUG' 
Write-EnhancedLog "Total entries in BYOD and CORP Entra Registered Non-Compliant Report: $totalBYODAndCorpRegisteredNonCompliant" -Level 'DEBUG'





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