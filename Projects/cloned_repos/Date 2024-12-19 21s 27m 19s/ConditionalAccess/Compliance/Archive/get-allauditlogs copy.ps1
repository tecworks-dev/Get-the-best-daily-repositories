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
# Install-RequiredModules
# Write-EnhancedLog -Message "All modules installed" -Level "INFO" -ForegroundColor ([ConsoleColor]::Green)


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


# # Convert the plain text access token to a SecureString
# $plainAccessToken = $accessToken
# $secureAccessToken = ConvertTo-SecureString $plainAccessToken -AsPlainText -Force


# Write-Host 'calling Connect-MgGraph with access token'

# # Connect to Microsoft Graph using the access token
# Connect-MgGraph -AccessToken $secureAccessToken


# Write-Host 'calling Invoke-MgGraphRequest against the Graph API after calling connect-mggraph'


# ################################################################################################################################
# ################################################################################################################################
# ################################################################################################################################

# #region device audit

# $exportpath = "c:\temp\auditlog.csv"


# function getallpagination () {
#     [cmdletbinding()]
        
#     param
#     (
#         $url
#     )
#         $response = (Invoke-MgGraphRequest -uri $url -Method Get -Headers $headers -OutputType PSObject)
#         $alloutput = $response.value
        
#         $alloutputNextLink = $response."@odata.nextLink"
        
#         while ($null -ne $alloutputNextLink) {
#             $alloutputResponse = (Invoke-MGGraphRequest -Uri $alloutputNextLink -Method Get -Headers $headers -outputType PSObject)
#             $alloutputNextLink = $alloutputResponse."@odata.nextLink"
#             $alloutput += $alloutputResponse.value
#         }
        
#         return $alloutput
#         }
    

# $uri = "https://graph.microsoft.com/beta/deviceManagement/auditEvents"
# $eventsvalues = getallpagination -url $uri
# $eventsvalues =  $eventsvalues | select-object * -ExpandProperty Actor
# $eventsvalues =  $eventsvalues | select-object resources, userPrincipalName, displayName, category, activityType, activityDateTime, activityOperationType, id 

# $listofevents = @()
# $counter = 0
# foreach ($event in $eventsvalues)
# {
#     $counter++
#     $id = $event.id
#     Write-Progress -Activity 'Processing Entries' -CurrentOperation $id -PercentComplete (($counter / $eventsvalues.count) * 100)
#     $eventobject = [pscustomobject]@{
#         changedItem = $event.Resources.displayName
#         changedBy = $event.userPrincipalName
#         change = $event.displayName
#         changeCategory = $event.category
#         activityType = $event.activityType
#         activityDateTime = $event.activityDateTime
#         id = $event.id
#     }
#     $listofevents += $eventobject
# }



# Add-Type -AssemblyName System.Windows.Forms

# $form = New-Object System.Windows.Forms.Form
# $form.Text = "Export or View"
# $form.Width = 300
# $form.Height = 150
# $form.StartPosition = "CenterScreen"

# $label = New-Object System.Windows.Forms.Label
# $label.Text = "Select an option:"
# $label.Location = New-Object System.Drawing.Point(10, 20)
# $label.AutoSize = $true
# $form.Controls.Add($label)

# $exportButton = New-Object System.Windows.Forms.Button
# $exportButton.Text = "Export"
# $exportButton.Location = New-Object System.Drawing.Point(100, 60)
# $exportButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
# $form.AcceptButton = $exportButton
# $form.Controls.Add($exportButton)

# $viewButton = New-Object System.Windows.Forms.Button
# $viewButton.Text = "View"
# $viewButton.Location = New-Object System.Drawing.Point(180, 60)
# $viewButton.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
# $form.CancelButton = $viewButton
# $form.Controls.Add($viewButton)

# $result = $form.ShowDialog()

# if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
#     # Export code here    
#     if ($null -eq $listofevents) {
#        write-host "Nothing to display"
#     }
#     else {
#         $listofevents | export-csv $exportpath
#     }

# } elseif ($result -eq [System.Windows.Forms.DialogResult]::Cancel) {
#     # View code here
#     if ($null -eq $listofevents) {
#        write-host "Nothing to display"
#     }
#     else {
#         $selected = $listofevents | Out-GridView -PassThru

# $selectedevents = @()

# foreach ($item in $selected) {
#     $selectedid = $item.id
# $uri = "https://graph.microsoft.com/beta/deviceManagement/auditEvents/$selectedid"
# $changedcontent = (Invoke-MgGraphRequest -Uri $uri -Method GET -Headers $headers -ContentType "application/json" -OutputType PSObject)

# $eventobject = [pscustomobject]@{
#     change = $changedcontent.displayName
#     changeCategory = $changedcontent.category
#     activityType = $changedcontent.activityType
#     activityDateTime = $changedcontent.activityDateTime
#     id = $changedcontent.id
#     activity = $changedcontent.activity
#     activityResult = $changedcontent.activityResult
#     activityOperationType = $changedcontent.activityOperationType
#     componentName = $changedcontent.componentName
#     type = $changedcontent.actor.type
#     auditActorType = $changedcontent.actor.auditActorType
#     userPermissions = $changedcontent.actor.userPermissions
#     applicationId = $changedcontent.actor.applicationId
#     applicationDisplayName = $changedcontent.actor.applicationDisplayName
#     userPrincipalName = $changedcontent.actor.userPrincipalName
#     servicePrincipalName = $changedcontent.actor.servicePrincipalName
#     ipAddress = $changedcontent.actor.ipAddress
#     userId = $changedcontent.actor.userId
#     remoteTenantId = $changedcontent.actor.remoteTenantId
#     remoteUserId = $changedcontent.actor.remoteUserId
#     resourcedisplayname = $changedcontent.resource.displayName
#     resourcetype = $changedcontent.resource.type
#     auditResourceType = $changedcontent.resource.auditResourceType
#     resourceId = $changedcontent.resource.resourceId
# }

# ##Resources is an open-ended array depending on the size of the policy
# ##We can't have multiple items in the object with the same name so we'll use an incrementing number

# ##Set to 0
# $i = 0
# ##Loop through the array
# foreach ($resource in $changedcontent.resources.modifiedproperties) {
#     ##Create a new property with the name and value
#     $name = "Name" + $i
#     $oldvalue = "OldValue" + $i
#     $newvalue = "NewValue" + $i
#     $eventobject | Add-Member -MemberType NoteProperty -Name $name -Value $resource.displayName
#     $eventobject | Add-Member -MemberType NoteProperty -Name $oldvalue -Value $resource.oldValue
#     $eventobject | Add-Member -MemberType NoteProperty -Name $newvalue -Value $resource.newValue
#     ##Increment
#     $i++
# }
# $selectedevents += $eventobject
# }
# $selectedevents | Out-GridView
#     }
# }
# #endregion device audit



# # Variables
# # $accessToken = "YOUR_ACCESS_TOKEN"
# $startDate = "2024-05-05T00:00:00Z"
# $endDate = "2024-05-31T23:59:59Z"
# $baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
# $headers = @{
#     "Authorization" = "Bearer $accessToken"
#     "Content-Type"  = "application/json"
# }

# # Initial URL with filters
# $url = "$baseUrl?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"

# # Function to make the API request and handle pagination
# function Get-SignInLogs {
#     param (
#         [string]$url
#     )
    
#     $allLogs = @()

#     while ($url) {
#         # Make the API request
#         $response = Invoke-WebRequest -Uri $url -Headers $headers -Method Get
#         $data = ($response.Content | ConvertFrom-Json)
        
#         # Collect the logs
#         $allLogs += $data.value
        
#         # Check for pagination
#         $url = $data.'@odata.nextLink'
        
#         # Implementing basic retry logic for rate limiting
#         if ($response.StatusCode -eq 429) {
#             Start-Sleep -Seconds 10
#             continue
#         }
#     }

#     return $allLogs
# }

# # Get all sign-in logs
# $signInLogs = Get-SignInLogs -url $url

# # Export to CSV
# $signInLogs | Select-Object userPrincipalName, createdDateTime, status, clientAppUsed, ipAddress | Export-Csv -Path "SignInLogs.csv" -NoTypeInformation




# # Variables
# # $accessToken = "YOUR_ACCESS_TOKEN"
# $endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
# $startDate = (Get-Date).AddHours(-24).ToString("yyyy-MM-ddTHH:mm:ssZ")
# $baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
# $headers = @{
#     "Authorization" = "Bearer $accessToken"
#     "Content-Type"  = "application/json"
# }

# # Initial URL with filters
# $url = "$baseUrl`?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"

# # Function to make the API request and handle pagination
# function Get-SignInLogs {
#     param (
#         [string]$url
#     )
    
#     $allLogs = @()

#     while ($url) {
#         try {
#             # Make the API request
#             $response = Invoke-WebRequest -Uri $url -Headers $headers -Method Get
#             $data = ($response.Content | ConvertFrom-Json)
            
#             # Collect the logs
#             $allLogs += $data.value
            
#             # Check for pagination
#             $url = $data.'@odata.nextLink'
#         } catch {
#             Write-Host "Error: $($_.Exception.Message)"
#             break
#         }
#     }

#     return $allLogs
# }

# # Get all sign-in logs for the last 24 hours
# $signInLogs = Get-SignInLogs -url $url

# # Export to CSV
# $signInLogs | Select-Object userPrincipalName, createdDateTime, status, clientAppUsed, ipAddress | Export-Csv -Path "/usr/src/SignInLogs.csv" -NoTypeInformation

# Write-Host "Export complete. Check /usr/src/SignInLogs.csv for results."








# Variables
# $accessToken = "YOUR_ACCESS_TOKEN"
$endDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
$startDate = (Get-Date).AddHours(-24).ToString("yyyy-MM-ddTHH:mm:ssZ")
$baseUrl = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
$headers = @{
    "Authorization" = "Bearer $accessToken"
    "Content-Type"  = "application/json"
}

# # Initial URL with filters
$url = "$baseUrl`?`$filter=createdDateTime ge $startDate and createdDateTime le $endDate"

# Function to make the API request and handle pagination
function Get-SignInLogs {
    param (
        [string]$url
    )
    
    $allLogs = @()

    while ($url) {
        try {
            # Make the API request
            $response = Invoke-WebRequest -Uri $url -Headers $headers -Method Get
            $data = ($response.Content | ConvertFrom-Json)
            
            # Collect the logs
            $allLogs += $data.value
            
            # Check for pagination
            $url = $data.'@odata.nextLink'
        } catch {
            Write-Host "Error: $($_.Exception.Message)"
            break
        }
    }

    return $allLogs
}

# Get all sign-in logs for the last 24 hours
$signInLogs = Get-SignInLogs -url $url

# Export to JSON
$signInLogs | ConvertTo-Json -Depth 10 | Out-File -FilePath "/usr/src/SignInLogs.json" -Encoding utf8

Write-Host "Export complete. Check /usr/src/SignInLogs.json for results."
