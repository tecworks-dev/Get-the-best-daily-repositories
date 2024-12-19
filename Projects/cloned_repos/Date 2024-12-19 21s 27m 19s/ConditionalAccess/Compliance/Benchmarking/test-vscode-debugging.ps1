function Get-Greeting {
    param (
        [string]$Name
    )
    return "Hello, $Name!"
}

# Import the module
Import-Module "$PSScriptRoot\helloworld.psm1" -Force

# Test the function
$greeting = Get-Greeting -Name "World"
Write-Host $greeting

$DBG

# Breakpoint to inspect the variable
$greeting
