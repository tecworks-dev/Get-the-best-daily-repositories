# Define the class
class SignInLog {
    [string] $userDisplayName
    [string] $deviceId
}

# Path to the JSON file
$JsonFilePath = "C:\log.json"

# Initialize a list to store the filtered log objects
$logList = [System.Collections.Generic.List[SignInLog]]::new()

# Open the JSON file and create a JsonDocument
$fileStream = [System.IO.File]::OpenRead($JsonFilePath)

try {
    # Read and process the JSON file incrementally
    $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

    foreach ($element in $jsonDoc.RootElement.EnumerateArray()) {
        $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        $deviceId = $element.GetProperty("deviceDetail").GetProperty("deviceId").GetString()

        if ($userDisplayName -ne "On-Premises Directory Synchronization Service Account") {
            $signInLog = [SignInLog]::new()
            $signInLog.userDisplayName = $userDisplayName
            $signInLog.deviceId = $deviceId
            $logList.Add($signInLog)
        }
    }

    # Output the filtered logs
    # $logList | ForEach-Object {
    #     Write-Host "User: $($_.userDisplayName), Device ID: $($_.deviceId)"
    # }
}
finally {
    # Clean up
    $fileStream.Dispose()
}
