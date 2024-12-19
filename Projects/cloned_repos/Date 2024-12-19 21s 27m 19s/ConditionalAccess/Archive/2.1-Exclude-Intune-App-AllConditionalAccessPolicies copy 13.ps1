# $ErrorActionPreference = 'SilentlyContinue'

# Start-Transcript -Path "C:\Code\CB\Entra\Sandbox\Graph\Exclude_AppFromAllCAPoliciesUsingBeta-v1.log"
# Install the Microsoft Graph Beta module if not already installed
# Install-Module Microsoft.Graph.Beta -Scope Allusers -AllowClobber -Force

# Import the Microsoft Graph Beta module
# Import-Module Microsoft.Graph.Beta


# Import-Module Microsoft.Graph.Identity.SignIns



# Start-Transcript -Path "C:\Code\CB\Entra\Sandbox\Graph\Exclude_AppFromAllCAPoliciesUsingBeta-v1.log"
# Install the Microsoft Graph Beta module if not already installed
# Install-Module Microsoft.Graph.Beta -Scope Allusers -AllowClobber -Force

# Install-Module Microsoft.Graph.Authentication -Scope Allusers -AllowClobber -Force
# Install-Module Microsoft.Graph.dentity -Scope Allusers -AllowClobber -Force

# Install-Module Microsoft.Graph.Beta.Authentication -Scope Allusers -AllowClobber -Force
# Install-Module Microsoft.Graph.Beta.Identity.Signins -Scope Allusers -AllowClobber -Force


# Import the Microsoft Graph Beta module
# Import-Module Microsoft.Graph.Beta



# Import-Module Microsoft.Graph.Identity.SignIns

# Import-Module Microsoft.Graph.Beta.Authentication
# Import-Module Microsoft.Graph.Beta.Identity



# Import-Module Microsoft.Graph.Authentication
# Import-Module Microsoft.Beta.Authentication
# Import-Module Microsoft.Graph.Beta.Identity.signins




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

# Import-Module "E:\Code\CB\Entra\ARH\Private\EnhancedGraphAO\2.0.0\EnhancedGraphAO.psm1" -Verbose


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


$EnhancedGraphAO = Join-Path -Path $PSScriptRoot -ChildPath "Modules\EnhancedGraphAO\2.0.0\EnhancedGraphAO.psm1"
Import-ModuleWithRetry -ModulePath $EnhancedGraphAO





# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################


function Install-RequiredModules {

    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # $requiredModules = @("Microsoft.Graph", "Microsoft.Graph.Authentication")
    $requiredModules = @("Microsoft.Graph.Authentication", "Microsoft.Graph.Beta.Identity.Signins")

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


    $ImportedModules = @("Microsoft.Graph.Identity.DirectoryManagement", "Microsoft.Graph.Authentication", "Microsoft.Graph.Beta.Identity.Signins")
    
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


Write-EnhancedLog -Message "calling Connect-MgGraph with access token" -Level "INFO" -ForegroundColor ([ConsoleColor]::Yellow) 

# Connect to Microsoft Graph using the access token
Connect-MgGraph -AccessToken $secureAccessToken



# Connect to Microsoft Graph
# Connect-MgGraph -Scopes "Policy.ReadWrite.ConditionalAccess", "Group.ReadWrite.All", "Policy.Read.All, Application.Read.All"

function Get-ConditionalAccessPoliciesViaMgGraph {
    $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
    $allPolicies = @()


    Write-EnhancedLog -Message "Entering Get-ConditionalAccessPoliciesViaMgGraph " -Level "INFO" -ForegroundColor ([ConsoleColor]::Yellow) 

    do {
        $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer"="odata.maxpagesize=999"}
        $policies = $response.Value
        $allPolicies += $policies

        $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
    } while ($uri)

    return $allPolicies

    Write-EnhancedLog -Message "Exiting Get-ConditionalAccessPoliciesViaMgGraph " -Level "INFO" -ForegroundColor ([ConsoleColor]::Green) 
}

function ExcludeAppFromAllCAPoliciesUsingBeta {
    param (
        [System.Collections.Hashtable]$ExcludeApps  # Hashtable with AppId as key and AppName as value
    )

    # Retrieve all Conditional Access Policies
    $allPolicies = Get-ConditionalAccessPoliciesViaMgGraph

    foreach ($policy in $allPolicies) {
        foreach ($appId in $ExcludeApps.Keys) {
            $appName = $ExcludeApps[$appId]

            # Check if the policy supports excluding apps
            if (-not $policy.conditions.applications) {
                Write-Host "Policy: $($policy.displayName) does not support excluding apps. Skipping..."
                continue
            }

            # Check if the app is already excluded
            if ($policy.conditions.applications.excludeApplications -contains $appId) {
                Write-Host "App '$appName' ($appId) is already excluded in Policy: $($policy.displayName)"
                continue
            }

            # Prepare the updated list of excluded apps
            $updatedExcludeApps = $policy.conditions.applications.excludeApplications + $appId

            # Construct the updated conditions object
            $updatedConditions = @{
                applications = @{
                    excludeApplications = $updatedExcludeApps
                }
            }

            # Update the Conditional Access Policy to exclude the app
            try {
                Update-MgBetaIdentityConditionalAccessPolicy -ConditionalAccessPolicyId $policy.id -Conditions $updatedConditions
                Write-Host "Updated Policy: $($policy.displayName) to exclude App '$appName' ($appId)"
            } catch {
                # Write-Host "Failed to update Policy: $($policy.displayName) for App '$appName' ($appId). Error: $($_.Exception.Message)"
                #I supress the error here because it's too verbose but usually it fails for policies that don't support excluding apps
                Write-Host "Failed to update Policy: $($policy.displayName) for App '$appName' ($appId)."
            }
        }
    }
}

# Define the apps to exclude with their friendly names
$excludeApps = @{
    'd4ebce55-015a-49b5-a083-c84d1797ae8c' = 'Microsoft Intune Enrollment'
    '0000000a-0000-0000-c000-000000000000' = 'Microsoft Intune'
    '00000003-0000-0ff1-ce00-000000000000' = 'Office 365 SharePoint Online'
    '766d89a4-d6a6-444d-8a5e-e1a18622288a' = 'OneDrive'
}

# Assuming ExcludeAppFromAllCAPoliciesUsingBeta is a function that accepts a hashtable parameter named -ExcludeApps

# Iterate over each app in the $excludeApps hashtable
foreach ($app in $excludeApps.GetEnumerator()) {
    # Create a new hashtable with only the current app
    $singleAppExclude = @{ $app.Name = $app.Value }
    
    # Call the function for the current app
    ExcludeAppFromAllCAPoliciesUsingBeta -ExcludeApps $singleAppExclude
}





# Stop-Transcript



#note that you will get the following error and that's ok because some tenants may not include that App ID at all or the CA policy does not support excluding apps in general


# Update-MgBetaIdentityConditionalAccessPolicy : The server could not process the request because it is malformed or incorrect.
# Status: 400 (BadRequest)
# ErrorCode: BadRequest
# Date: 2024-02-26T16:02:32
# Headers:
# Transfer-Encoding             : chunked
# Vary                          : Accept-Encoding
# Strict-Transport-Security     : max-age=31536000
# request-id                    : 3003446d-b020-4746-a0ff-15e5eb84256a
# client-request-id             : b8e95072-2b5f-4842-bc37-e3ddd6140a57
# x-ms-ags-diagnostic           : {"ServerInfo":{"DataCenter":"Canada Central","Slice":"E","Ring":"5","ScaleUnit":"001","RoleInstance":"YT1PEPF00001F24"}}
# Link                          : <https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://
# developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/c
# hanges?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:netw
# orkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;
# rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https
# ://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:authenticationFlows&from=2023-06-01&to=2023-07-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-u
# s/graph/changes?$filterby=beta,PrivatePreview:authenticationFlows&from=2023-06-01&to=2023-07-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,Priv
# atePreview:authenticationFlows&from=2023-06-01&to=2023-07-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:insiderRiskLevels&from=2
# 023-07-01&to=2023-08-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:insiderRiskLevels&from=2023-07-01&to=2023-08-01>;rel="depreca
# tion";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer
# .microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$f
# ilterby=beta,PrivatePreview:networkAccess&from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:networkAccess
# &from=2022-02-01&to=2022-03-01>;rel="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:secureAppSessionMode&from=2023-07-01&to=2023-08-01>;re
# l="deprecation";type="text/html",<https://developer.microsoft-tst.com/en-us/graph/changes?$filterby=beta,PrivatePreview:secureAppSessionMode&from=2023-07-01&to=2023-08-01>;rel="deprecation";type="text/html"
# Deprecation                   : Thu, 17 Feb 2022 23:59:59 GMT
# Sunset                        : Sat, 17 Feb 2024 23:59:59 GMT
# Cache-Control                 : no-cache
# Date                          : Mon, 26 Feb 2024 16:02:31 GMT
# At C:\Code\CB\Entra\CCI\Graph\2.1-Exclude-Intune-App-AllConditionalAccessPolicies copy 13.ps1:68 char:17
# + ...             Update-MgBetaIdentityConditionalAccessPolicy -Conditional ...
# +                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     + CategoryInfo          : InvalidOperation: ({ ConditionalAc...lAccessPolicy }:<>f__AnonymousType293`3) [Update-MgBetaId..._UpdateExpanded], Exception
#     + FullyQualifiedErrorId : BadRequest,Microsoft.Graph.Beta.PowerShell.Cmdlets.UpdateMgBetaIdentityConditionalAccessPolicy_UpdateExpanded










#successful Example

# App 'Microsoft Intune Enrollment' (d4ebce55-015a-49b5-a083-c84d1797ae8c) is already excluded in Policy: CA008 - BLOCK - aollivierre@bcclsp.org - All locations except trusted location
# App 'Microsoft Intune' (0000000a-0000-0000-c000-000000000000) is already excluded in Policy: CA008 - BLOCK - aollivierre@bcclsp.org - All locations except trusted location
# App 'OneDrive' (766d89a4-d6a6-444d-8a5e-e1a18622288a) is already excluded in Policy: CA008 - BLOCK - aollivierre@bcclsp.org - All locations except trusted location
# App 'Office 365 SharePoint Online' (00000003-0000-0ff1-ce00-000000000000) is already excluded in Policy: CA008 - BLOCK - aollivierre@bcclsp.org - All locations except trusted location



#Run script at least 6 times to ensure all app are excluded and then export the configs using https://idpowertoys.com/ca everytime (log out of https://idpowertoys.com/ca and login back again between each export to ensure fresh data)