#the following does not work keeps returning 


# WARNING: Command's Invoke-MgGraphRequest definition is missing. Skip getting its dependencies
# WARNING: Command's Invoke-MgGraphRequest definition is missing. Skip getting its dependencies
# WARNING: Command's Invoke-MgGraphRequest definition is missing. Skip getting its dependencies
# WARNING: Command's Invoke-MgGraphRequest definition is missing. Skip getting its dependencies
# WARNING: Command's Invoke-MgGraphRequest definition is missing. Skip getting its dependencies
# WARNING: Command's Get-MgApplication definition is missing. Skip getting its dependencies
# WARNING: Command's Update-MgApplication definition is missing. Skip getting its dependencies
# WARNING: Unable to find command 'Invoke-MSGraphRequest' (source: ) details. Skip getting its dependencies
# InvalidOperation: Cannot index into a null array.
# InvalidOperation: Cannot index into a null array.
# InvalidOperation: Cannot index into a null array.
# WARNING: Unable to find command 'Remove-O365OrphanedMailbox' (source: ) details. Skip getting its dependencies
# WARNING: 'Find-MgGraphCommand' was unable to find command ''?!
# WARNING: Be noted that it is impossible to tell whether found permissions for some command are all required, or just some subset of them (for least-privileged access). Consult the Microsoft Graph Permissions Reference documentation to identify the least-privileged permission for your use case :(

# # Ensure required modules are installed
# Install-Module -Name MSGraphStuff -Scope Allusers -Force -AllowClobber
# Install-Module -Name DependencySearch -Scope Allusers -Force -AllowClobber
# Install-Module -Name Microsoft.Graph.Authentication -Scope Allusers -Force -AllowClobber
# Install-Module -Name Microsoft.Graph.Reports -Scope Allusers -Force -AllowClobber
# Install-Module -Name Microsoft.Graph.Users -Scope Allusers -Force -AllowClobber

# Load the necessary modules
Import-Module MSGraphStuff
Import-Module DependencySearch
Import-Module Microsoft.Graph.Authentication
Import-Module Microsoft.Graph.Reports
Import-Module Microsoft.Graph.Users

# Example code that includes the URI we want to check using a recognized command
# Example code that includes a recognized command
# $codeToCheck = @"
# Get-MgUser -UserId 'user@example.com'
# "@

# Write the example code to a temporary file
# $tempScriptPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "tempGraphScript.ps1")
# Set-Content -Path $tempScriptPath -Value $codeToCheck

# Cache available modules to speed up repeated 'Get-CodeGraphPermissionRequirement' function invocations
$availableModules = @(Get-Module -ListAvailable)

# Get the required permissions for the example code
# $permissions = Get-CodeGraphPermissionRequirement -scriptPath $tempScriptPath -goDeep -availableModules $availableModules -permType "application", "delegated"
$permissions = Get-CodeGraphPermissionRequirement -scriptPath "C:\Code\CB\Entra\ICTC\Entra\Devices\Beta\GraphCalls.ps1" -goDeep -availableModules $availableModules -permType "application", "delegated"

# Output the permissions
$permissions | Out-GridView

# Clean up the temporary file
# Remove-Item -Path $tempScriptPath