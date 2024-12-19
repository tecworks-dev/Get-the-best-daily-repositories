# # Ensure required modules are installed
Install-Module -Name MSGraphStuff -Scope Allusers -Force -AllowClobber
Install-Module -Name DependencySearch -Scope Allusers -Force -AllowClobber
Install-Module -Name Microsoft.Graph.Authentication -Scope Allusers -Force -AllowClobber

# Load the necessary modules
Import-Module MSGraphStuff
Import-Module DependencySearch
Import-Module Microsoft.Graph.Authentication

# Example code that includes the URI we want to check
$codeToCheck = @"
Invoke-MgGraphRequest -Uri 'https://graph.microsoft.com/v1.0/auditLogs/signIns' -Method Get
"@

# Write the example code to a temporary file
$tempScriptPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "tempGraphScript.ps1")
Set-Content -Path $tempScriptPath -Value $codeToCheck

# Cache available modules to speed up repeated 'Get-CodeGraphPermissionRequirement' function invocations
$availableModules = @(Get-Module -ListAvailable)

# Get the required permissions for the example code
$permissions = Get-CodeGraphPermissionRequirement -scriptPath $tempScriptPath -goDeep -availableModules $availableModules -permType "application", "delegated"

# Output the permissions
$permissions | Out-GridView

# Clean up the temporary file
Remove-Item -Path $tempScriptPath
