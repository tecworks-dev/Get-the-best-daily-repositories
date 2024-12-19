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



# Load client secrets from the JSON file
$secretsjsonPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"
$secrets = Get-Content -Path $secretsjsonPath | ConvertFrom-Json

# Variables from JSON file
$tenantId = $secrets.tenantId
$clientId = $secrets.clientId
$clientSecret = $secrets.clientSecret


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




# Example of how to use the function
# $PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$EnhancedLoggingAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedLoggingAO\2.5.0\EnhancedLoggingAO.psm1"


# Call the function to import the module with retry logic
Import-ModuleWithRetry -ModulePath $EnhancedLoggingAO

# Import-Module "E:\Code\CB\Entra\ARH\Private\EnhancedGraphAO\2.5.0\EnhancedGraphAO.psm1" -Verbose


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


$EnhancedGraphAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedGraphAO\2.5.0\EnhancedGraphAO.psm1"
Import-ModuleWithRetry -ModulePath $EnhancedGraphAO





# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


function Install-RequiredModules {

    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # $requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Authentication")
    $requiredModules = @("Microsoft.Graph.Authentication")

    foreach ($module in $requiredModules) {
        if (!(Get-Module -ListAvailable -Name $module)) {

            Write-EnhancedLog -Message "Installing module: $module" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
            Install-Module -Name $module -Force
            Write-EnhancedLog -Message "Module: $module has been installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
        }
        else {
            Write-EnhancedLog -Message "Module $module is already installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
        }
    }


    $ImportedModules = @("Microsoft.Graph.Identity.DirectoryManagement", "Microsoft.Graph.Authentication")
    
    foreach ($Importedmodule in $ImportedModules) {
        if ((Get-Module -ListAvailable -Name $Importedmodule)) {
            Write-EnhancedLog -Message "Importing module: $Importedmodule" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
            Import-Module -Name $Importedmodule
            Write-EnhancedLog -Message "Module: $Importedmodule has been Imported" -Level "INFO" -ForegroundColor ([ConsoleColor]::Cyan)
        }
    }


}
# Call the function to install the required modules and dependencies
Install-RequiredModules
Write-EnhancedLog -Message "All modules installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


# Example call to the function
$accessToken = Get-MsGraphAccessToken -tenantId $tenantId -clientId $clientId -clientSecret $clientSecret

# Define the headers
# $headers = @{
#     "Authorization" = "Bearer $accessToken"
# }

# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


function Log-Params {
    param (
        [hashtable]$Params
    )

    foreach ($key in $Params.Keys) {
        Write-Host "$key $($Params[$key])"
    }
}


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


Log-Params -Params @{accessToken = $accessToken}


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


# # Define the access token
# $accessToken = "your_access_token_here"

# # Define the headers
# $headers = @{
#     "Authorization" = "Bearer $accessToken"
# }



# Define the access token
# $accessToken = "your_access_token_here"




# Convert the plain text access token to a SecureString
$plainAccessToken = $accessToken
$secureAccessToken = ConvertTo-SecureString $plainAccessToken -AsPlainText -Force


Write-Host 'calling Connect-MgGraph with access token'

# Connect to Microsoft Graph using the access token
Connect-MgGraph -AccessToken $secureAccessToken


Write-Host 'calling Invoke-MgGraphRequest against the Graph API after calling connect-mggraph'

# Now you can make requests using Invoke-MgGraphRequest
# $response = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/me"

# Output the response
# $response




# # Create a custom GraphRequestSession
# $graphSession = New-Object Microsoft.Graph.PowerShell.Authentication.Helpers.GraphRequestSession
# $graphSession.BearerToken = $accessToken

# # Make the request using the custom session
# $response = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/me" -GraphRequestSession $graphSession

# # Output the response
# $response









# # Make the request using Invoke-MgGraphRequest
# $response = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/me" -Headers $headers
# $response = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/organization"

# # Output the response
# $response



$headers = @{
    "Authorization" = "Bearer $accessToken"
    "Content-Type"  = "application/json"
}

# Fetch tenant details
# $tenantDetailsUrl = "https://graph.microsoft.com/v1.0/organization"
# $tenantResponse = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/organization" -Headers $headers
# $tenantResponse = Invoke-MgGraphRequest -Method GET -Uri "https://graph.microsoft.com/v1.0/organization"
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












# Fetch organization details using the Graph cmdlet
$organization = Get-MgOrganization

# Output the organization details
$organization | Format-List






# Extract the required details
# $organization = $organization[0]   
$tenantName = $organization.DisplayName
$tenantId = $organization.Id
$tenantDomain = $organization.VerifiedDomains[0].Name

# Assume $signInLogs is already defined somewhere in your script
# $signInLogs = <Your logic to get sign-in logs>

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