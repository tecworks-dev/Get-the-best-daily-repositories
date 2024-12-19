# Install MSGraphStuff module from the PowerShell Gallery
Install-Module -Name MSGraphStuff -Scope AllUsers -Force -AllowClobber

# Example code that includes the URI we want to check
$codeToCheck = @"
Invoke-RestMethod -Uri 'https://graph.microsoft.com/v1.0/auditLogs/signIns' -Method Get
"@

# Write the example code to a temporary file
$tempScriptPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "tempGraphScript.ps1")
Set-Content -Path $tempScriptPath -Value $codeToCheck

# Load the MSGraphStuff module
Import-Module MSGraphStuff

# Get the required permissions for the example code
$permissions = Get-CodeGraphPermissionRequirement -scriptPath $tempScriptPath -permType "application", "delegated"

# Output the permissions
$permissions | Out-GridView

# Clean up the temporary file
Remove-Item -Path $tempScriptPath
