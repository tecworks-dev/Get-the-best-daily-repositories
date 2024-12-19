# Load System.Text.Json assembly
Add-Type -AssemblyName System.Text.Json

# Method 3: Using System.Text.Json.JsonDocument
function Process-Json-JsonDocument-Method3 {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $fileStream = [System.IO.File]::OpenRead($JsonFilePath)
    $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

    # Simulate processing
    $elementCount = $jsonDoc.RootElement.GetArrayLength()

    $fileStream.Close()

    $stopwatch.Stop()
    return $stopwatch.Elapsed.TotalMilliseconds
}

# Method 4: Using System.Text.Json.JsonDocument with an Efficient File Open
function Process-Json-JsonDocument-Method4 {
    param (
        [string]$JsonFilePath
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    # Open the file using FileStream with buffering and SequentialScan
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $jsonDoc = [System.Text.Json.JsonDocument]::Parse($fileStream)

    # Simulate processing
    $elementCount = $jsonDoc.RootElement.GetArrayLength()

    $fileStream.Close()

    $stopwatch.Stop()
    return $stopwatch.Elapsed.TotalMilliseconds
}

# Run each method 100 times and record the elapsed times
$resultsMethod3 = @()
$resultsMethod4 = @()
$JsonFilePath = "C:\log.json"

for ($i = 0; $i -lt 100; $i++) {
    $resultsMethod3 += Process-Json-JsonDocument-Method3 -JsonFilePath $JsonFilePath
    $resultsMethod4 += Process-Json-JsonDocument-Method4 -JsonFilePath $JsonFilePath
}

# Calculate average times
$averageTimeMethod3 = ($resultsMethod3 | Measure-Object -Average).Average
$averageTimeMethod4 = ($resultsMethod4 | Measure-Object -Average).Average

# Display results
Write-Host "Method 3: Average time using System.Text.Json.JsonDocument took $averageTimeMethod3 ms"
Write-Host "Method 4: Average time using System.Text.Json.JsonDocument with FileStream took $averageTimeMethod4 ms"






# $code = @"
# using System;
# using System.IO;
# using System.Text.Json;

# public class JsonProcessor
# {
#     public static long ProcessJson(string jsonFilePath)
#     {
#         var stopwatch = System.Diagnostics.Stopwatch.StartNew();

#         using (FileStream fileStream = new FileStream(jsonFilePath, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan))
#         {
#             byte[] buffer = new byte[4096];
#             int bytesRead;
#             long elementCount = 0;

#             while ((bytesRead = fileStream.Read(buffer, 0, buffer.Length)) > 0)
#             {
#                 var jsonReader = new Utf8JsonReader(buffer.AsSpan(0, bytesRead), isFinalBlock: bytesRead < buffer.Length, state: default);

#                 while (jsonReader.Read())
#                 {
#                     if (jsonReader.TokenType == JsonTokenType.StartObject)
#                     {
#                         elementCount++;
#                     }
#                 }
#             }

#             stopwatch.Stop();
#             Console.WriteLine($"Utf8JsonReader took {stopwatch.Elapsed.TotalMilliseconds} ms, processed {elementCount} elements");
#             return stopwatch.ElapsedMilliseconds;
#         }
#     }
# }
# "@




# # Compile the C# code
# Add-Type -TypeDefinition $code -Language CSharp

# # Define the PowerShell function to call the C# method
# function Process-Json-Utf8JsonReader {
#     param (
#         [string]$JsonFilePath
#     )

#     return [JsonProcessor]::ProcessJson($JsonFilePath)
# }

# # Run the method 100 times and record the elapsed times
# $resultsMethod5 = @()
# $JsonFilePath = "C:\log.json"

# for ($i = 0; $i -lt 100; $i++) {
#     $resultsMethod5 += Process-Json-Utf8JsonReader -JsonFilePath $JsonFilePath
# }

# # Calculate average time
# $averageTimeMethod5 = ($resultsMethod5 | Measure-Object -Average).Average

# # Display results
# Write-Host "Method 5: Average time using System.Text.Json.Utf8JsonReader took $averageTimeMethod5 ms"






