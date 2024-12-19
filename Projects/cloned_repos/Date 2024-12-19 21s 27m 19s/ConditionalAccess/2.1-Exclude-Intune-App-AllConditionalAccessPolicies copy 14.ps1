# Set environment variable globally for all users
[System.Environment]::SetEnvironmentVariable('EnvironmentMode', 'dev', 'Machine')

# Retrieve the environment mode (default to 'prod' if not set)
$mode = $env:EnvironmentMode

#region FIRING UP MODULE STARTER
#################################################################################################
#                                                                                               #
#                                 FIRING UP MODULE STARTER                                      #
#                                                                                               #
#################################################################################################

Invoke-Expression (Invoke-RestMethod "https://raw.githubusercontent.com/aollivierre/module-starter/main/Install-EnhancedModuleStarterAO.ps1")

# Wait-Debugger

# Define a hashtable for splatting
$moduleStarterParams = @{
    Mode                   = 'dev'
    SkipPSGalleryModules   = $true
    SkipCheckandElevate    = $true
    SkipPowerShell7Install = $true
    SkipEnhancedModules    = $true
    SkipGitRepos           = $true
}

# Call the function using the splat
Invoke-ModuleStarter @moduleStarterParams


# Wait-Debugger

#endregion FIRING UP MODULE STARTER

# Toggle based on the environment mode
switch ($mode) {
    'dev' {
        Write-EnhancedLog -Message "Running in development mode" -Level 'WARNING'
        # Your development logic here
    }
    'prod' {
        Write-EnhancedLog -Message "Running in production mode" -ForegroundColor Green
        # Your production logic here
    }
    default {
        Write-EnhancedLog -Message "Unknown mode. Defaulting to production." -ForegroundColor Red
        # Default to production
    }
}



#region HANDLE PSF MODERN LOGGING
#################################################################################################
#                                                                                               #
#                            HANDLE PSF MODERN LOGGING                                          #
#                                                                                               #
#################################################################################################
Set-PSFConfig -Fullname 'PSFramework.Logging.FileSystem.ModernLog' -Value $true -PassThru | Register-PSFConfig -Scope SystemDefault

# Define the base logs path and job name
$JobName = "WindowsUpdates"
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
        Write-EnhancedLog -Message "Transcript stopped." -ForegroundColor Cyan
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -ForegroundColor Red
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

    ################################################################################################################################
    ################################################ START GRAPH CONNECTING ########################################################
    ################################################################################################################################


        #region LOADING SECRETS FOR GRAPH
    #################################################################################################
    #                                                                                               #
    #                                 LOADING SECRETS FOR GRAPH                                     #
    #                                                                                               #
    #################################################################################################


    #First, load secrets and create a credential object:
    # Assuming secrets.json is in the same directory as your script
    $secretsPath = Join-Path -Path $PSScriptRoot -ChildPath "secrets.json"

    # Load the secrets from the JSON file
    $secrets = Get-Content -Path $secretsPath -Raw | ConvertFrom-Json

    # Read configuration from the JSON file
    # Assign values from JSON to variables


    #  Variables from JSON file
    $tenantId = $secrets.TenantId
    $clientId = $secrets.ClientId

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

    Write-EnhancedLog -Message "PFX file found: $certPath" -Level 'INFO'

    $CertPassword = $secrets.CertPassword

    #endregion LOADING SECRETS FOR GRAPH

    # Define the splat for Connect-GraphWithCert
    $graphParams = @{
        tenantId        = $tenantId
        clientId        = $clientId
        certPath        = $certPath
        certPassword    = $certPassword
        ConnectToIntune = $true
        ConnectToTeams  = $false
    }

    # Connect to Microsoft Graph, Intune, and Teams
    #Commeting the next line out as I'm connecting interactively using Connect-mgGraph first
    # $accessToken = Connect-GraphWithCert @graphParams

    Log-Params -Params @{accessToken = $accessToken }

    Get-TenantDetails
    #################################################################################################################################
    ################################################# END Connecting to Graph #######################################################
    #################################################################################################################################

    function Get-ConditionalAccessPoliciesViaMgGraph {
        $uri = "https://graph.microsoft.com/beta/identity/conditionalAccess/policies"
        
        # Initialize a list for better performance
        $allPolicies = [System.Collections.Generic.List[PSObject]]::new()
    
        do {
            # Fetch the policies via Graph API
            $response = Invoke-MgGraphRequest -Uri $uri -Method GET -Headers @{"Prefer" = "odata.maxpagesize=999" }
            $policies = $response.Value
    
            # Add the policies to the list
            $allPolicies.Add($policies)
    
            # Check for next link for pagination
            $uri = if ($response.'@odata.nextLink') { $response.'@odata.nextLink' } else { $null }
        } while ($uri)
    
        # Convert the list back to an array (optional)
        return $allPolicies.ToArray()
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
                    Write-EnhancedLog -Message "Policy: $($policy.displayName) does not support excluding apps. Skipping..."
                    continue
                }

                # Check if the app is already excluded
                if ($policy.conditions.applications.excludeApplications -contains $appId) {
                    Write-EnhancedLog -Message "App '$appName' ($appId) is already excluded in Policy: $($policy.displayName)"
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
                    Write-EnhancedLog -Message "Updated Policy: $($policy.displayName) to exclude App '$appName' ($appId)"
                }
                catch {
                    # Write-EnahhWrite-EnhancedLog -Message "Failed to update Policy: $($policy.displayName) for App '$appName' ($appId). Error: $($_.Exception.Message)"
                    #I supress the error here because it's too verbose but usually it fails for policies that don't support excluding apps
                    Write-EnhancedLog -Message "Failed to update Policy: $($policy.displayName) for App '$appName' ($appId)."
                }
            }
        }
    }

    # Define the apps to exclude with their friendly names
    $excludeApps = @{
        'd4ebce55-015a-49b5-a083-c84d1797ae8c' = 'Microsoft Intune Enrollment'
        '0000000a-0000-0000-c000-000000000000' = 'Microsoft Intune'
        # '00000003-0000-0ff1-ce00-000000000000' = 'Office 365 SharePoint Online'
        # '766d89a4-d6a6-444d-8a5e-e1a18622288a' = 'OneDrive'
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


    # Wait-Debugger
    

 
    #endregion Script Logic
}
catch {
    Write-EnhancedLog -Message "An error occurred during script execution: $_" -Level 'ERROR'
    if ($transcriptPath) {
        Stop-Transcript
        Write-EnhancedLog -Message "Transcript stopped." -ForegroundColor Cyan
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -ForegroundColor Red
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
        Write-EnhancedLog -Message "Transcript stopped." -ForegroundColor Cyan
        # Stop logging in the finally block

    }
    else {
        Write-EnhancedLog -Message "Transcript was not started due to an earlier error." -ForegroundColor Red
    }
    # 

    
    # Ensure the log is written before proceeding
    Wait-PSFMessage

    # Stop logging in the finally block by disabling the provider
    Set-PSFLoggingProvider -Name 'logfile' -InstanceName $instanceName -Enabled $false

}



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