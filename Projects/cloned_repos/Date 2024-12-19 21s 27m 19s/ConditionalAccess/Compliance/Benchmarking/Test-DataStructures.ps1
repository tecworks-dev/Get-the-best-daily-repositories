# Helper function to measure execution time
function Measure-ExecutionTime {
    param (
        [scriptblock]$Code,
        [int]$Iterations = 100
    )
    $totalTime = [timespan]::Zero
    for ($i = 0; $i -lt $Iterations; $i++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        & $Code
        $stopwatch.Stop()
        $totalTime += $stopwatch.Elapsed
    }
    return $totalTime.TotalMilliseconds / $Iterations
}

# Helper function to parse JSON data
function Parse-JsonData {
    param (
        [string]$JsonData
    )
    $jsonData = $JsonData.Trim()
    if ($jsonData.StartsWith('[') -and $jsonData.EndsWith(']')) {
        return $jsonData | ConvertFrom-Json
    }
    elseif ($jsonData.StartsWith('{') -and $jsonData.EndsWith('}')) {
        return , $jsonData | ConvertFrom-Json
    }
    else {
        Write-Warning "Invalid JSON data encountered: $jsonData"
        return $null
    }
}

# Method 1: Using Array
function Process-SignInLogs-Array {
    param (
        [string]$JsonFilePath
    )
    $uniqueIds = @()
    $lines = Get-Content -Path $JsonFilePath -ReadCount 0
    foreach ($line in $lines) {
        $logs = Parse-JsonData -JsonData $line
        if ($logs) {
            foreach ($log in $logs) {
                $deviceId = $log.deviceDetail.deviceId
                if (-not $uniqueIds.Contains($deviceId)) {
                    $uniqueIds += $deviceId
                }
            }
        }
    }
}

# Method 2: Using Hashtable
function Process-SignInLogs-Hashtable {
    param (
        [string]$JsonFilePath
    )
    $uniqueIds = @{}
    $lines = Get-Content -Path $JsonFilePath -ReadCount 0
    foreach ($line in $lines) {
        $logs = Parse-JsonData -JsonData $line
        if ($logs) {
            foreach ($log in $logs) {
                $deviceId = $log.deviceDetail.deviceId
                if (-not $uniqueIds.ContainsKey($deviceId)) {
                    $uniqueIds[$deviceId] = $true
                }
            }
        }
    }
}

# Method 3: Using HashSet
function Process-SignInLogs-HashSet {
    param (
        [string]$JsonFilePath
    )
    $uniqueIds = [System.Collections.Generic.HashSet[string]]::new()
    $lines = Get-Content -Path $JsonFilePath -ReadCount 0
    foreach ($line in $lines) {
        $logs = Parse-JsonData -JsonData $line
        if ($logs) {
            foreach ($log in $logs) {
                $deviceId = $log.deviceDetail.deviceId
                $uniqueIds.Add($deviceId) | Out-Null
            }
        }
    }
}

# JSON file path
$jsonFilePath = "C:\log.json"

# Measure execution times
$arrayTime = Measure-ExecutionTime -Code { Process-SignInLogs-Array -JsonFilePath $jsonFilePath }
$hashtableTime = Measure-ExecutionTime -Code { Process-SignInLogs-Hashtable -JsonFilePath $jsonFilePath }
$hashSetTime = Measure-ExecutionTime -Code { Process-SignInLogs-HashSet -JsonFilePath $jsonFilePath }

# Output results
Write-Host "Average execution time using Array: $arrayTime ms"
Write-Host "Average execution time using Hashtable: $hashtableTime ms"
Write-Host "Average execution time using HashSet: $hashSetTime ms"