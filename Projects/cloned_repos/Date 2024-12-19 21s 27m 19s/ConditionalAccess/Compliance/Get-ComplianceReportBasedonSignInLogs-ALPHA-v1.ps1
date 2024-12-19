# Check if running in PowerShell 7
if ($PSVersionTable.PSVersion.Major -lt 7) {
    # Log a warning and exit
    Write-Warning "This script is optimized for PowerShell 7. Please run this script in PowerShell 7 unless you're installing modules in production mode."
    exit
}

# Continue with the rest of your script if in PowerShell 7
Write-Host "Running in PowerShell 7. Proceeding with the script..."



$global:mode = 'dev'
$global:SimulatingIntune = $false
# $ExitOnCondition = $false

[System.Environment]::SetEnvironmentVariable('EnvironmentMode', $global:mode, 'Machine')
[System.Environment]::SetEnvironmentVariable('EnvironmentMode', $global:mode, 'process')

# Alternatively, use this PowerShell method (same effect)
# Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -Name 'EnvironmentMode' -Value 'dev'

$global:mode = $env:EnvironmentMode
$global:LOG_ASYNC = $true #Enable Async mode (all levels except Warnings, Errors and Criticals are treated as Debug which means they are written to the log file without showing on the console)
$global:LOG_SILENT = $true  # Enable silent mode (all levels are treated as Debug)


# Toggle based on the environment mode
switch ($mode) {
    'dev' {
        Write-Host "Running in development mode" -ForegroundColor Yellow
        # Your development logic here
    }
    'prod' {
        Write-Host "Running in production mode" -ForegroundColor Green
        # Your production logic here
    }
    default {
        Write-Host "Unknown mode. Defaulting to production." -ForegroundColor Red
        # Default to production
    }
}





# The script checks for an environment variable named LOG_ASYNC. If this environment variable is set to "true", it sets the $Async switch to $true.
# [System.Environment]::SetEnvironmentVariable('LOG_ASYNC', 'false', 'Process')


# $envVarName = 'LOG_ASYNC'
# $envVarValue = 'false'
# $envVarScope = 'Process'

# # Log before setting the environment variable
# Write-Host "Setting environment variable $envVarName to $envVarValue in scope $envVarScope."

# # Set the environment variable
# [System.Environment]::SetEnvironmentVariable($envVarName, $envVarValue, $envVarScope)

# # Log after setting the environment variable
# if ([System.Environment]::GetEnvironmentVariable($envVarName, $envVarScope) -eq $envVarValue) {
#     Write-Host "Environment variable $envVarName successfully set to $envVarValue."
# } else {
#     Write-Host "Failed to set environment variable $envVarName."
# }



# When to Use Each Mode:
# Async Logging is ideal for production environments where performance is a priority, and logging is a lower priority in terms of timing.
# Sync Logging is more suited for development, debugging, or critical logging scenarios where you need to ensure that log entries are written immediately and in a specific order.

# Check if async logging is enabled
# if ($global:LOG_ASYNC) {
#     # Initialize the global log queue and start the async logging job
#     if (-not $global:LogQueue) {
#         $global:LogQueue = [System.Collections.Concurrent.ConcurrentQueue[PSCustomObject]]::new()

#         $global:LogJob = Start-Job -ScriptBlock {
#             param ($logQueue)

#             while ($true) {
#                 if ($logQueue.TryDequeue([ref]$logItem)) {
#                     Write-PSFMessage -Level $logItem.Level -Message $logItem.Message -FunctionName $logItem.FunctionName
#                 }
#                 else {
#                     Start-Sleep -Milliseconds 100
#                 }
#             }
#         } -ArgumentList $global:LogQueue
#     }
# }


function Write-ComplianceReportBasedonSigninLogs {
    param (
        [string]$Message,
        [string]$Level = "INFO"
        # [switch]$Async = $false  # Control whether logging should be async or not
    )

    # Check if the Async switch is not set, then use the global variable if defined
    # if (-not $Async) {
    #     $Async = $global:LOG_ASYNC
    # }

    # Get the PowerShell call stack to determine the actual calling function
    $callStack = Get-PSCallStack
    $callerFunction = if ($callStack.Count -ge 2) { $callStack[1].Command } else { '<Unknown>' }

    # Prepare the formatted message with the actual calling function information
    $formattedMessage = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] [$callerFunction] $Message"

    # if ($Async) {
    #     # Enqueue the log message for async processing
    #     $logItem = [PSCustomObject]@{
    #         Level        = $Level
    #         Message      = $formattedMessage
    #         FunctionName = $callerFunction
    #     }
    #     $global:LogQueue.Enqueue($logItem)
    
    
    # Display the log message based on the log level using Write-Host
    switch ($Level.ToUpper()) {
        "DEBUG" { Write-Host $formattedMessage -ForegroundColor DarkGray }
        "INFO" { Write-Host $formattedMessage -ForegroundColor Green }
        "NOTICE" { Write-Host $formattedMessage -ForegroundColor Cyan }
        "WARNING" { Write-Host $formattedMessage -ForegroundColor Yellow }
        "ERROR" { Write-Host $formattedMessage -ForegroundColor Red }
        "CRITICAL" { Write-Host $formattedMessage -ForegroundColor Magenta }
        default { Write-Host $formattedMessage -ForegroundColor White }
    }

    # Append to log file synchronously
    $logFilePath = [System.IO.Path]::Combine($env:TEMP, 'ComplianceSigninLogs.log')
    $formattedMessage | Out-File -FilePath $logFilePath -Append -Encoding utf8
}



#region FIRING UP MODULE STARTER
#################################################################################################
#                                                                                               #
#                                 FIRING UP MODULE STARTER                                      #
#                                                                                               #
#################################################################################################


# Define the mutex name (should be the same across all scripts needing synchronization)
$mutexName = "Global\MyCustomMutexForModuleInstallation"

# Create or open the mutex
$mutex = [System.Threading.Mutex]::new($false, $mutexName)

# Set initial back-off parameters
$initialWaitTime = 5       # Initial wait time in seconds
$maxAttempts = 10           # Maximum number of attempts
$backOffFactor = 2         # Factor to increase the wait time for each attempt

$attempt = 0
$acquiredLock = $false

# Try acquiring the mutex with dynamic back-off
while (-not $acquiredLock -and $attempt -lt $maxAttempts) {
    $attempt++
    Write-ComplianceReportBasedonSigninLogs -Message "Attempt $attempt to acquire the lock..."

    # Try to acquire the mutex with a timeout
    $acquiredLock = $mutex.WaitOne([TimeSpan]::FromSeconds($initialWaitTime))

    if (-not $acquiredLock) {
        # If lock wasn't acquired, wait for the back-off period before retrying
        Write-ComplianceReportBasedonSigninLogs "Failed to acquire the lock. Retrying in $initialWaitTime seconds..." -Level 'WARNING'
        Start-Sleep -Seconds $initialWaitTime

        # Increase the wait time using the back-off factor
        $initialWaitTime *= $backOffFactor
    }
}

try {
    if ($acquiredLock) {
        Write-ComplianceReportBasedonSigninLogs -Message "Acquired the lock. Proceeding with module installation and import."

        # Start timing the critical section
        $executionTime = [System.Diagnostics.Stopwatch]::StartNew()

        # Critical section starts here

        # Conditional check for dev and prod mode
        if ($global:mode -eq "dev") {
            # In dev mode, import the module from the local path
            Write-ComplianceReportBasedonSigninLogs -Message "Running in dev mode. Importing module from local path."
            Import-Module 'C:\code\ModulesV2\EnhancedModuleStarterAO\EnhancedModuleStarterAO.psm1'
        }
        elseif ($global:mode -eq "prod") {
            # In prod mode, execute the script from the URL
            Write-ComplianceReportBasedonSigninLogs -Message "Running in prod mode. Executing the script from the remote URL."
            # Invoke-Expression (Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1")


            # Check if running in PowerShell 5 (only in prod mode)
            if ($PSVersionTable.PSVersion.Major -ne 5) {
                Write-ComplianceReportBasedonSigninLogs -Message "Not running in PowerShell 5. Relaunching the command with PowerShell 5."

                # Reset Module Paths when switching from PS7 to PS5 process
                Reset-ModulePaths

                # Get the path to PowerShell 5 executable
                $ps5Path = "$Env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

                # Relaunch the Invoke-Expression command with PowerShell 5
                & $ps5Path -Command "Invoke-Expression (Invoke-RestMethod 'https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1')"
            }
            else {
                # If running in PowerShell 5, execute the command directly
                Write-ComplianceReportBasedonSigninLogs -Message "Running in PowerShell 5. Executing the command."
                Invoke-Expression (Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1")
            }

        }
        else {
            Write-ComplianceReportBasedonSigninLogs -Message "Invalid mode specified. Please set the mode to either 'dev' or 'prod'." -Level 'WARNING'
            exit 1
        }

        # Optional: Wait for debugger if needed
        # Wait-Debugger


        # Define a hashtable for splatting
        $moduleStarterParams = @{
            Mode                   = $global:mode
            SkipPSGalleryModules   = $false
            SkipCheckandElevate    = $false
            SkipPowerShell7Install = $false
            SkipEnhancedModules    = $false
            SkipGitRepos           = $true
        }

        # Check for PowerShell 5 only in prod mode
        if ($global:mode -eq "prod" -and $PSVersionTable.PSVersion.Major -ne 5) {
            Write-ComplianceReportBasedonSigninLogs -Message  "Not running in PowerShell 5. Relaunching the function call with PowerShell 5."

            # Get the path to PowerShell 5 executable
            $ps5Path = "$Env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

            Reset-ModulePaths

            # Relaunch the Invoke-ModuleStarter function call with PowerShell 5
            & $ps5Path -Command {
                # Recreate the hashtable within the script block for PowerShell 5
                $moduleStarterParams = @{
                    Mode                   = $global:mode
                    SkipPSGalleryModules   = $false
                    SkipCheckandElevate    = $false
                    SkipPowerShell7Install = $false
                    SkipEnhancedModules    = $false
                    SkipGitRepos           = $true
                }
                Invoke-ModuleStarter @moduleStarterParams
            }
        }
        else {
            # If running in PowerShell 5 or dev mode, execute the function directly
            Write-ComplianceReportBasedonSigninLogs -Message "Running in PowerShell 5 or dev mode. Executing Invoke-ModuleStarter."
            Invoke-ModuleStarter @moduleStarterParams
        }

        # Critical section ends here
        $executionTime.Stop()

        # Measure the time taken and log it
        $timeTaken = $executionTime.Elapsed.TotalSeconds
        Write-ComplianceReportBasedonSigninLogs -Message "Critical section execution time: $timeTaken seconds"

        Write-ComplianceReportBasedonSigninLogs -Message "Module installation and import completed."
    }
    else {
        Write-Warning "Failed to acquire the lock after $maxAttempts attempts. Exiting the script."
        exit 1
    }
}
catch {
    Write-Error "An error occurred: $_"
}
finally {
    # Release the mutex if it was acquired
    if ($acquiredLock) {
        $mutex.ReleaseMutex()
        Write-ComplianceReportBasedonSigninLogs -Message "Released the lock."
    }

    # Dispose of the mutex object
    $mutex.Dispose()
}

#endregion FIRING UP MODULE STARTER





#region HANDLE PSF MODERN LOGGING
#################################################################################################
#                                                                                               #
#                            HANDLE PSF MODERN LOGGING                                          #
#                                                                                               #
#################################################################################################
# Check if the current user is an administrator
$isAdmin = CheckAndElevate -ElevateIfNotAdmin $false

# Set the configuration and register it with the appropriate scope based on admin privileges
if ($isAdmin) {
    # If the user is admin, register in the SystemDefault scope
    Set-PSFConfig -Fullname 'PSFramework.Logging.FileSystem.ModernLog' -Value $true -PassThru | Register-PSFConfig -Scope SystemDefault
}
else {
    # If the user is not admin, register in the User scope
    Set-PSFConfig -Fullname 'PSFramework.Logging.FileSystem.ModernLog' -Value $true -PassThru | Register-PSFConfig -Scope UserDefault
}




# Check if the current user is an administrator
$isAdmin = CheckAndElevate -ElevateIfNotAdmin $false

# Set the configuration and register it with the appropriate scope based on admin privileges
if ($isAdmin) {
    # If the user is admin, 

    
    # Enable asynchronous logging in PSFramework
    Set-PSFConfig -FullName 'PSFramework.Logging.FileSystem.Asynchronous' -Value $true -PassThru | Register-PSFConfig -Scope SystemDefault
}
else {
    
    # Enable asynchronous logging in PSFramework
    Set-PSFConfig -FullName 'PSFramework.Logging.FileSystem.Asynchronous' -Value $true -PassThru | Register-PSFConfig -Scope UserDefault
}



# Define the base logs path and job name
$JobName = "ComplianceReportBasedonSignInLogs"
$parentScriptName = Get-ParentScriptName
Write-EnhancedLog -Message "Parent Script Name: $parentScriptName"

# Call the Get-PSFCSVLogFilePath function to generate the dynamic log file path
$paramGetPSFCSVLogFilePath = @{
    LogsPath         = 'C:\Logs\PSF'
    JobName          = $jobName
    parentScriptName = $parentScriptName
}

$csvLogFilePath = Get-PSFCSVLogFilePath @paramGetPSFCSVLogFilePath

$instanceName = "$parentScriptName-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

# Configure the PSFramework logging provider to use CSV format
$paramSetPSFLoggingProvider = @{
    Name            = 'logfile'
    InstanceName    = $instanceName  # Use a unique instance name
    FilePath        = $csvLogFilePath  # Use the dynamically generated file path
    Enabled         = $true
    FileType        = 'CSV'
    EnableException = $true
}
Set-PSFLoggingProvider @paramSetPSFLoggingProvider
#endregion HANDLE PSF MODERN LOGGING


#region HANDLE Transript LOGGING
#################################################################################################
#                                                                                               #
#                            HANDLE Transript LOGGING                                           #
#                                                                                               #
#################################################################################################
# Start the script with error handling
try {
    # Generate the transcript file path
    $GetTranscriptFilePathParams = @{
        TranscriptsPath  = "C:\Logs\Transcript"
        JobName          = $jobName
        parentScriptName = $parentScriptName
    }
    $transcriptPath = Get-TranscriptFilePath @GetTranscriptFilePathParams
    
    # Start the transcript
    Write-EnhancedLog -Message "Starting transcript at: $transcriptPath"
    Start-Transcript -Path $transcriptPath
}
catch {
    Write-EnhancedLog -Message "An error occurred during script execution: $_" -Level 'ERROR'
    if ($transcriptPath) {
        Stop-Transcript
        Write-EnhancedLog -Message "Transcript stopped." -Level 'WARNING'
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -Level 'ERROR'
    }

    # Stop PSF Logging

    # Ensure the log is written before proceeding
    Wait-PSFMessage

    # Stop logging in the finally block by disabling the provider
    Set-PSFLoggingProvider -Name 'logfile' -InstanceName $instanceName -Enabled $false

    Handle-Error -ErrorRecord $_
    throw $_  # Re-throw the error after logging it
}
#endregion HANDLE Transript LOGGING


try {


    #region Script Logic
    #################################################################################################
    #                                                                                               #
    #                                    Script Logic                                               #
    #                                                                                               #
    #################################################################################################


    #################################################################################################################################
    ################################################# START VARIABLES ###############################################################
    #################################################################################################################################

    # #First, load secrets and create a credential object:
    # # Assuming secrets.json is in the same directory as your script
    # $secretsPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"

    # # Load the secrets from the JSON file
    # $secrets = Get-Content -Path $secretsPath -Raw | ConvertFrom-Json

    # #  Variables from JSON file
    # $tenantId = $secrets.tenantId
    # $clientId = $secrets.clientId

    # # Find any PFX file in the root directory of the script
    # $pfxFiles = Get-ChildItem -Path $PSScriptRoot -Filter *.pfx

    # if ($pfxFiles.Count -eq 0) {
    #     Write-Error "No PFX file found in the root directory."
    #     throw "No PFX file found"
    # }
    # elseif ($pfxFiles.Count -gt 1) {
    #     Write-Error "Multiple PFX files found in the root directory. Please ensure there is only one PFX file."
    #     throw "Multiple PFX files found"
    # }

    # # Use the first (and presumably only) PFX file found
    # $certPath = $pfxFiles[0].FullName

    # Write-Output "PFX file found: $certPath"

    # $CertPassword = $secrets.CertPassword





    #region LOADING SECRETS FOR GRAPH
    #################################################################################################
    #                                                                                               #
    #                                 LOADING SECRETS FOR GRAPH                                     #
    #                                                                                               #
    #################################################################################################


    #     Start
    #   |
    #   v
    # Check if secrets directory exists
    #   |
    #   +-- [Yes] --> Check if tenant folders exist
    #   |                |
    #   |                +-- [Yes] --> List tenant folders
    #   |                |                |
    #   |                |                v
    #   |                |       Display list and prompt user for tenant selection
    #   |                |                |
    #   |                |                v
    #   |                |       Validate user's selected tenant folder
    #   |                |                |
    #   |                |                +-- [Valid] --> Check if secrets.json exists
    #   |                |                |                 |
    #   |                |                |                 +-- [Yes] --> Load secrets from JSON file
    #   |                |                |                 |                |
    #   |                |                |                 |                v
    #   |                |                |                 |        Check for PFX file
    #   |                |                |                 |                |
    #   |                |                |                 |                +-- [Yes] --> Validate single PFX file
    #   |                |                |                 |                |                 |
    #   |                |                |                 |                |                 v
    #   |                |                |                 |                |        Assign values from secrets to variables
    #   |                |                |                 |                |                 |
    #   |                |                |                 |                |                 v
    #   |                |                |                 |                +--> Write log "PFX file found"
    #   |                |                |                 |
    #   |                |                |                 +-- [No] --> Error: secrets.json not found
    #   |                |                |                
    #   |                |                +-- [Invalid] --> Error: Invalid tenant folder
    #   |                |                
    #   |                +-- [No] --> Error: No tenant folders found
    #   |
    #   +-- [No] --> Error: Secrets directory not found
    #   |
    #   v
    # End


    # Define the path to the secrets directory
    $secretsDirPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets"

    # Check if the secrets directory exists
    if (-Not (Test-Path -Path $secretsDirPath)) {
        Write-Error "Secrets directory not found at '$secretsDirPath'."
        throw "Secrets directory not found"
    }

    # List all folders (tenants) in the secrets directory
    $tenantFolders = Get-ChildItem -Path $secretsDirPath -Directory

    if ($tenantFolders.Count -eq 0) {
        Write-Error "No tenant folders found in the secrets directory."
        throw "No tenant folders found"
    }

    # Display the list of tenant folders and ask the user to confirm
    Write-Host "Available tenant folders:"
    $tenantFolders | ForEach-Object { Write-Host "- $($_.Name)" }

    $selectedTenant = Read-Host "Enter the name of the tenant folder you want to use"

    # Validate the user's selection
    $selectedTenantPath = Join-Path -Path $secretsDirPath -ChildPath $selectedTenant

    if (-Not (Test-Path -Path $selectedTenantPath)) {
        Write-Error "The specified tenant folder '$selectedTenant' does not exist."
        throw "Invalid tenant folder"
    }

    # Define paths for the secrets.json and PFX files
    $secretsJsonPath = Join-Path -Path $selectedTenantPath -ChildPath "secrets.json"
    $pfxFiles = Get-ChildItem -Path $selectedTenantPath -Filter *.pfx

    # Check if secrets.json exists
    if (-Not (Test-Path -Path $secretsJsonPath)) {
        Write-Error "secrets.json file not found in '$selectedTenantPath'."
        throw "secrets.json file not found"
    }

    # Load the secrets from the JSON file
    $secrets = Get-Content -Path $secretsJsonPath -Raw | ConvertFrom-Json

    # Check if a PFX file exists
    if ($pfxFiles.Count -eq 0) {
        Write-Error "No PFX file found in the '$selectedTenantPath' directory."
        throw "No PFX file found"
    }
    elseif ($pfxFiles.Count -gt 1) {
        Write-Error "Multiple PFX files found in the '$selectedTenantPath' directory. Please ensure there is only one PFX file."
        throw "Multiple PFX files found"
    }

    # Use the first (and presumably only) PFX file found
    $certPath = $pfxFiles[0].FullName

    Write-EnhancedLog -Message "PFX file found: $certPath" -Level 'INFO'

    # Assign values from JSON to variables
    $tenantId = $secrets.TenantId
    $clientId = $secrets.ClientId
    $CertPassword = $secrets.CertPassword


    #endregion LOADING SECRETS FOR GRAPH


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

    #Uncooment the following when debugging
    # $uris = @($url, $tenantDetailsUrl)
    # foreach ($uri in $uris) {
    #     if (-not (Validate-UriAccess -uri $uri -Headers $headers)) {
    #         Write-EnhancedLog "Validation failed. Halting script." -Color Red
    #         exit 1
    #     }
    # }

    # Wait-Debugger



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


    # $signInLogs = ExportAndProcessSignInLogs @ExportAndProcessSignInLogsparams

    # $signInLogs | Out-GridView -Title 'Original Sign Logs Pre-Processing'

    # $signInLogs | Out-HtmlView -Title 'Original Sign Logs Pre-Processing'




    # Step 1: Load the original sign-in logs
    $signInLogs = ExportAndProcessSignInLogs @ExportAndProcessSignInLogsparams

    # Optional: Display the original sign-in logs
    # $signInLogs | Out-GridView -Title 'Original Sign-In Logs Pre-Processing'
    $signInLogs | Out-HtmlView -Title 'Original Sign-In Logs Pre-Processing'


    #Commenting out the filtering of successfulLogins and exporting successfulLogins to Out-HtmlView

    # Step 2: Filter only successful logins (ErrorCode == 0)
    # $successfulLogins = $signInLogs | Where-Object { $_.Status.ErrorCode -eq 0 }

    # Optional: Display the filtered successful sign-in logs
    # $successfulLogins | Out-GridView -Title 'Successful Sign-In Logs'
    # $successfulLogins | Out-HtmlView -Title 'Successful Sign-In Logs'

    # Step 3: Export the filtered successful logins to a new JSON file
    $outputJsonPath = "$PSScriptroot\CustomExports\CustomSignInLogs\successful_signin_logs.json"
    $successfulLogins | ConvertTo-Json -Depth 5 | Set-Content -Path $outputJsonPath

    Write-Host "Filtered JSON file with successful logins saved to $outputJsonPath"


    # Wait-Debugger

    # $DBG


    # Ensure the signInLogs variable is not null before using it
    if ($null -eq $signInLogs -or @($signInLogs).Count -eq 0) {
        Write-Warning "No sign-in logs were loaded."
        exit 1
    }
    else {
        Write-Host "Loaded $(@($signInLogs).Count) sign-in logs."
    }

    # Ensure there are successful logins to process
    if ($null -eq $successfulLogins -or @($successfulLogins).Count -eq 0) {
        Write-Warning "No successful sign-in logs found."
        exit 1
    }
    else {
        Write-Host "Loaded $(@($successfulLogins).Count) successful sign-in logs."
    }


    # Wait-Debugger

    # Step 3: Process only successful sign-in logs
    $results = Process-SignInLogs -signInLogs $successfulLogins -Headers $Headers


    # $results = Process-SignInLogs -signInLogs $signInLogs -Headers $Headers

    # Ensure $results is treated as an array
    $results = @($results)

    if ($results.Count -gt 0) {
        $results | Export-Csv "$PSScriptRoot/$exportsFolderName/MasterReport.csv" -NoTypeInformation
        $results | Out-GridView -Title 'resulting Sign Logs Post-Processing'
        $results | Out-HtmlView -Title 'resulting Sign Logs Post-Processing'
    }
    else {
        Write-EnhancedLog -Message "No results to export." -Level 'WARNING'
    }


    # Wait-Debugger

    # Wait-Debugger
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
    Write-EnhancedLog "Total entries in Corporate Non-Compliant Report: $totalCorporateNonCompliant" -Level 'INFO'
    Write-EnhancedLog "Total entries in BYOD Compliant Report: $totalBYODCompliant" -Level 'INFO' 
    Write-EnhancedLog "Total entries in BYOD and CORP Entra Registered Non-Compliant Report: $totalBYODAndCorpRegisteredNonCompliant" -Level 'INFO'





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


    
    # Stop the logging job when done
    # Stop-Job -Job $global:LogJob
    # Remove-Job -Job $global:LogJob


}

catch {
    Write-EnhancedLog -Message "An error occurred during script execution: $_" -Level 'ERROR'
    if ($transcriptPath) {
        Stop-Transcript
        Write-EnhancedLog -Message "Transcript stopped." -Level 'WARNING'
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -Level 'ERROR'
    }

    # Stop PSF Logging

    # Ensure the log is written before proceeding
    Wait-PSFMessage

    # Stop logging in the finally block by disabling the provider
    Set-PSFLoggingProvider -Name 'logfile' -InstanceName $instanceName -Enabled $false

    Handle-Error -ErrorRecord $_
    throw $_  # Re-throw the error after logging it
} 
finally {
    # Ensure that the transcript is stopped even if an error occurs
    if ($transcriptPath) {
        Stop-Transcript
        Write-EnhancedLog -Message "Transcript stopped." -Level 'WARNING'
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -Level 'ERROR'
    }
    # 

    
    # Ensure the log is written before proceeding
    # Wait-PSFMessage




    # Your script code and logging...
    # Write-PSFMessage -Level Info -Message "Starting some operation."

    # ...your operations and logging continue

    # Write-PSFMessage -Level Info -Message "Some operation completed."

    # At the end of the script, make sure all logs are written to file
    Write-PSFMessage -Level Host -Message "Finishing script. Flushing logs."

    # Ensure all log messages are flushed from the buffer
    # Stop-PSFFunction
    Wait-PSFMessage  # This ensures all pending logs are written before script exits

    Write-PSFMessage -Level Host -Message "Script execution fully completed."




    # Stop logging in the finally block by disabling the provider
    Set-PSFLoggingProvider -Name 'logfile' -InstanceName $instanceName -Enabled $false


    # # Stop the logging job when done
    # Stop-Job -Job $global:LogJob
    # Remove-Job -Job $global:LogJob

}

