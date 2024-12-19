# Define the class
class SignInLog {
    [string] $userDisplayName
    [string] $deviceId
}

# Path to the JSON file
$JsonFilePath = "C:\log.json"

# Load the JSON file
$reader = [System.IO.StreamReader]::new($JsonFilePath)
$jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)

# Filter out specific users and create instances of SignInLog
$filteredLogs = $jarray.SelectTokens('$..[?(@.userDisplayName != ''On-Premises Directory Synchronization Service Account'')]')

# Initialize a list to store the filtered log objects
$logList = [System.Collections.Generic.List[SignInLog]]::new()

# Convert filtered JSON tokens to SignInLog objects
foreach ($log in $filteredLogs) {
    $userDisplayName = [string]$log.userDisplayName
    $deviceId = [string]$log.deviceDetail.deviceId
    $signInLog = [SignInLog]::new()
    $signInLog.userDisplayName = $userDisplayName
    $signInLog.deviceId = $deviceId
    $logList.Add($signInLog)
}

# Output the filtered logs
# $logList | ForEach-Object {
#     Write-Host "User: $($_.userDisplayName), Device ID: $($_.deviceId)"
# }
