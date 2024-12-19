# Define the class
class SignInLog {
    [string] $userDisplayName
    [string] $deviceId
}

# Path to the JSON file
$JsonFilePath = "C:\log.json"

# Initialize a list to store the filtered log objects
$logList = [System.Collections.Generic.List[SignInLog]]::new()

# Open the JSON file and create a JsonTextReader
$reader = [System.IO.StreamReader]::new($JsonFilePath)
$jsonReader = [Newtonsoft.Json.JsonTextReader]::new($reader)

# Read and process the JSON file incrementally
while ($jsonReader.Read()) {
    if ($jsonReader.TokenType -eq [Newtonsoft.Json.JsonToken]::StartObject) {
        $jObject = [Newtonsoft.Json.Linq.JObject]::Load($jsonReader)
        $userDisplayName = $jObject["userDisplayName"]
        $deviceId = $jObject["deviceDetail"]["deviceId"]

        if ($userDisplayName -ne "On-Premises Directory Synchronization Service Account") {
            $signInLog = [SignInLog]::new()
            $signInLog.userDisplayName = [string]$userDisplayName
            $signInLog.deviceId = [string]$deviceId
            $logList.Add($signInLog)
        }
    }
}

# # Output the filtered logs
# $logList | ForEach-Object {
#     Write-Host "User: $($_.userDisplayName), Device ID: $($_.deviceId)"
# }
