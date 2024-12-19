function Log-FunctionCall {
    param (
        [string]$Message,
        [string]$Level = 'INFO'
    )

    $callerFunction = (Get-PSCallStack)[1].Command
    $formattedMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level] [Function: $callerFunction] $Message"
    Write-Host $formattedMessage
}

function Test-Function {
    Log-FunctionCall -Message "This is a test message from the function."
}

Test-Function




class Logger {
    [void] LogClassCall([string]$Message, [string]$Level = 'INFO') {
        $callerFunction = (Get-PSCallStack)[1].Command
        $callerClass = $this.GetType().Name
        $formattedMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level] [Class: $callerClass] [Function: $callerFunction] $Message"
        Write-Host $formattedMessage
    }
}

class TestClass {
    [Logger]$logger

    TestClass() {
        $this.logger = [Logger]::new()
    }

    [void] TestMethod() {
        $this.logger.LogClassCall("This is a test message from the class method.", "INFO")
    }
}

$testInstance = [TestClass]::new()
$testInstance.TestMethod()
