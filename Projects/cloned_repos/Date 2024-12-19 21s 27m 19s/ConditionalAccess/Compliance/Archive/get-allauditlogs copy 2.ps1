#!ToDO work on creatng a function for importing all modules in the modules folders without specifying the path of each module.
#fix permissions of the client app to add Intune permissions


# Read configuration from the JSON file
# Assign values from JSON to variables

# Read configuration from the JSON file
$configPath = Join-Path -Path $PSScriptRoot -ChildPath "config.json"
$env:MYMODULE_CONFIG_PATH = $configPath



# Load client secrets from the JSON file
$secretsjsonPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"
$secrets = Get-Content -Path $secretsjsonPath | ConvertFrom-Json

# Variables from JSON file
$tenantId = $secrets.tenantId
$clientId = $secrets.clientId
$clientSecret = $secrets.clientSecret


function Get-ModulesScriptPathsAndVariables {
    

    <#
    .SYNOPSIS
    Dot-sources all PowerShell scripts in the 'Modules' folder relative to the script root.
    
    .DESCRIPTION
    This function finds all PowerShell (.ps1) scripts in a 'Modules' folder located in the script root directory and dot-sources them. It logs the process, including any errors encountered, with optional color coding.
    
    .EXAMPLE
    Dot-SourceModulesScripts
    
    Dot-sources all scripts in the 'Modules' folder and logs the process.
    
    .NOTES
    Ensure the Write-EnhancedLog function is defined before using this function for logging purposes.
    #>
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


# Example of how to use the function
# $PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$EnhancedLoggingAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedLoggingAO\2.5.0\EnhancedLoggingAO.psm1"
$EnhancedGraphAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedGraphAO\2.5.0\EnhancedGraphAO.psm1"




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
                Write-EnhancedLog -Message "Module: $ModulePath imported successfully." -Level "INFO"
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
            Write-EnhancedLog -Message "Module: $ModulePath loaded successfully." -Level "INFO"
        }
        else {
            Write-Host -Message "Failed to load module $ModulePath within the maximum retry limit."
        }
    }
}


# Call the function to import the module with retry logic
Import-ModuleWithRetry -ModulePath $EnhancedLoggingAO
Import-ModuleWithRetry -ModulePath $EnhancedGraphAO

# Import-Module "E:\Code\CB\Entra\ARH\Private\EnhancedGraphAO\2.0.0\EnhancedGraphAO.psm1" -Verbose


# ################################################################################################################################
# ################################################ END MODULE LOADING ############################################################
# ################################################################################################################################




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


# $EnhancedGraphAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedGraphAO\2.5.0\EnhancedGraphAO.psm1"
# Import-ModuleWithRetry -ModulePath $EnhancedGraphAO





# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################



# Call the function to install the required modules and dependencies
# Install-RequiredModules
# Write-EnhancedLog -Message "All modules installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


# Example call to the function
$accessToken = Get-MsGraphAccessToken -tenantId $tenantId -clientId $clientId -clientSecret $clientSecret


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################




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


# Validate access to required URIs
$uris = @($url, $intuneUrl, $tenantDetailsUrl)
foreach ($uri in $uris) {
    if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
        Write-EnhancedLog "Validation failed. Halting script." -Color Red
        exit 1
    }
}


# (we got them already now let's filter them)

# Get all sign-in logs for the last 30 days
$signInLogs = Get-SignInLogs -url $url -Headers $headers

# Export to JSON for further processing
$signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

Write-EnhancedLog "Export complete. Check /usr/src/SignInLogs.json for results." -Color Green

# Load the sign-in logs
$json = Get-Content -Path '/usr/src/SignInLogs.json' | ConvertFrom-Json


# Process the logs and filter details
$results = foreach ($item in $json) {
    $deviceState = Get-DeviceStateInIntune -deviceId $item.deviceDetail.deviceId

    [PSCustomObject]@{
        'DeviceName'             = $item.deviceDetail.displayName
        'UserName'               = $item.userDisplayName
        'DeviceEntraID'          = $item.deviceDetail.deviceId
        'UserEntraID'            = $item.userId
        'DeviceOS'               = $item.deviceDetail.operatingSystem
        'DeviceComplianceStatus' = if ($item.deviceDetail.isCompliant) { "Compliant" } else { "Non-Compliant" }
        'DeviceStateInIntune'    = $deviceState
        'TrustType'              = $item.deviceDetail.trustType
    }
}

# Export master report
$results | Export-Csv '/usr/src/MasterReport.csv' -NoTypeInformation

# Generate and export specific reports
$report1 = $results | Where-Object { ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Compliant' }
$report2 = $results | Where-Object { ($_.TrustType -eq 'Hybrid Azure AD joined' -or $_.TrustType -eq 'Azure AD Joined') -and $_.DeviceComplianceStatus -eq 'Non-Compliant' }
$report3 = $results | Where-Object { $_.TrustType -eq 'Azure AD Registered' -and $_.DeviceComplianceStatus -eq 'Compliant' }
$report4 = $results | Where-Object { $_.TrustType -eq 'Azure AD Registered' -and $_.DeviceComplianceStatus -eq 'Non-Compliant' }

$report1 | Export-Csv '/usr/src/CorporateCompliant.csv' -NoTypeInformation
$report2 | Export-Csv '/usr/src/CorporateIncompliant.csv' -NoTypeInformation
$report3 | Export-Csv '/usr/src/BYODCompliant.csv' -NoTypeInformation
$report4 | Export-Csv '/usr/src/BYODIncompliant.csv' -NoTypeInformation

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
Write-EnhancedLog "Total entries in BYOD Incompliant Report: $totalReport4" -Color Red

# # Fetch tenant details
# $tenantDetailsUrl = "https://graph.microsoft.com/v1.0/organization"
# $tenantResponse = Invoke-WebRequest -Uri $tenantDetailsUrl -Headers $headers -Method Get
# $tenantDetails = ($tenantResponse.Content | ConvertFrom-Json).value[0]

# $tenantName = $tenantDetails.displayName
# $tenantId = $tenantDetails.id
# $tenantDomain = $tenantDetails.verifiedDomains[0].name
# $appId = $signInLogs[0].appId
# $appName = $signInLogs[0].appDisplayName

# # Output tenant summary
# Write-EnhancedLog "Tenant Name: $tenantName" -Color White
# Write-EnhancedLog "Tenant ID: $tenantId" -Color White
# Write-EnhancedLog "Tenant Domain: $tenantDomain" -Color White
# Write-EnhancedLog "App ID: $appId" -Color White
# Write-EnhancedLog "App Name: $appName" -Color White





# # Call the Microsoft Graph API to get organization details
# $uri = "https://graph.microsoft.com/v1.0/organization"
# $headers = @{ Authorization = "Bearer $accessToken" }

# $response = Invoke-WebRequest -Uri $uri -Headers $headers -Method Get

# # Output the raw response content
# $response.Content








# # Fetch organization details using the Graph cmdlet
# $organization = Get-MgOrganization

# # Output the organization details
# $organization | Format-List






# Call the Microsoft Graph API to get organization details
$uri = "https://graph.microsoft.com/v1.0/organization"
# $headers = @{ Authorization = "Bearer $accessToken" }

$response = Invoke-WebRequest -Uri $uri -Headers $headers -Method Get

# Output the raw response content
$jsonContent  = $response.Content





# Parse the JSON response
$organizationDetails = $jsonContent | ConvertFrom-Json

# Extract the required details
$tenantDetails = $organizationDetails.value[0]
$tenantName = $tenantDetails.DisplayName
$tenantId = $tenantDetails.Id
$tenantDomain = $tenantDetails.VerifiedDomains[0].Name

# Output the extracted details
Write-Output "Tenant Name: $tenantName"
Write-Output "Tenant ID: $tenantId"
Write-Output "Tenant Domain: $tenantDomain"





# # Extract the required details
# # $organization = $organization[0]   
# $tenantName = $organization.DisplayName
# $tenantId = $organization.Id
# $tenantDomain = $organization.VerifiedDomains[0].Name

# # Assume $signInLogs is already defined somewhere in your script
# # $signInLogs = <Your logic to get sign-in logs>

# if ($signInLogs -and $signInLogs.Count -gt 0) {
#     $appId = $signInLogs[0].appId
#     $appName = $signInLogs[0].appDisplayName
# } else {
#     $appId = "N/A"
#     $appName = "N/A"
# }

# Output tenant summary
Write-EnhancedLog -Message "Tenant Name: $tenantName" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
Write-EnhancedLog -Message "Tenant ID: $tenantId" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
Write-EnhancedLog -Message "Tenant Domain: $tenantDomain" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
# Write-EnhancedLog -Message "App ID: $appId" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
# Write-EnhancedLog -Message "App Name: $appName" -Level "INFO" -ForegroundColor ([ConsoleColor]::White)
