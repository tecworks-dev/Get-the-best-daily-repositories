# Enable asynchronous logging in PSFramework
Set-PSFConfig -FullName PSFramework.Logging.FileSystem.Asynchronous -Value $true -PassThru | Register-PSFConfig


# Example of logging a message
Write-PSFMessage -Level 'Debug' -Message "This is a Debug message."
