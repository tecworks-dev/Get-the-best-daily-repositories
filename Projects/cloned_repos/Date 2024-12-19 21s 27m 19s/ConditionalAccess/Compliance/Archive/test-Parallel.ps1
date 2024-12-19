# Define a function that uses ForEach-Object -Parallel
function Demo-ParallelScriptRoot {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ScriptRoot
    )

    # Example array to iterate over
    $array = 1..5

    # Use ForEach-Object -Parallel
    $array | ForEach-Object -Parallel {
        # Use $using: scope to pass $ScriptRoot
        $localScriptRoot = $using:ScriptRoot
        Write-Output "Script Root in parallel block: $localScriptRoot"
    } -ThrottleLimit 2
}

# Call the function
Demo-ParallelScriptRoot -ScriptRoot $PSScriptRoot
