

# ps | Select name, id | convertto-json | set-content test.json
# $reader = [System.IO.StreamReader]::new("$pwd\test.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# # $jarray.SelectToken('$').SelectTokens('$..[?(@.Name == ''pwsh'')]')
# $jarray.Count





# # Load the JSON data from the file
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)

# # # Search for the user with the specified userPrincipalName and display the userDisplayName
# $user = $jarray.SelectToken('$').SelectTokens('$..[?(@.userPrincipalName == ''r.fakhouri@ictc-ctic.ca'')]')

# if ($user) {
#     $user.userDisplayName
# } else {
#     Write-Output "User not found"
# }

# # # Close the reader
# $reader.Close()







# # Load the JSON data from the file
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)

# # Search for the user with the specified userPrincipalName and display the userDisplayName
# $users = $jarray.SelectToken('$').SelectTokens('$..[?(@.userPrincipalName == ''r.fakhouri@ictc-ctic.ca'')]')

# if ($users.Count -gt 0) {
#     foreach ($user in $users) {
#         $user.userDisplayName
#     }
# } else {
#     Write-Output "User not found"
# }

# # Close the reader
# $reader.Close()







# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# $reader.Close()
# $jarray.Count




# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# $users = $jarray.SelectToken('$').SelectTokens('$..[?(@.userPrincipalName == ''r.fakhouri@ictc-ctic.ca'')]')
# # foreach ($user in $users) {
# #     $user | Format-Table -Property userPrincipalName, userDisplayName
# # }
# $users.Count
# # $users | gm
# $reader.Close()
# # $users.Count




# Load the JSON data from the file
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# $reader.Close()

# # Search for the user with the specified userPrincipalName and count the matches
# $users = $jarray.SelectToken('$').SelectTokens('$..[?(@.userPrincipalName == ''r.fakhouri@ictc-ctic.ca'')]')
# $count = $users.Count
# $count







# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()
# Write-Output "JSON Content:"
# Write-Output $content

# # Step 2: Parse the JSON content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $jarray = [Newtonsoft.Json.Linq.JArray]::Load([NewtonSoft.Json.JsonTextReader]$reader)
# $reader.Close()
# Write-Output "Parsed JSON Array:"
# $jarray

# # Step 3: Search for the specific userPrincipalName
# $users = $jarray.SelectToken('$').SelectTokens('$..[?(@.userPrincipalName == ''r.fakhouri@ictc-ctic.ca'')]')
# Write-Output "Matching Users:"
# $users

# # Step 4: Count the matches
# $count = $users.Count
# Write-Output "Count of Matching Users:"
# $count











# # Load the JSON data from the file
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Display the raw JSON content
# Write-Output "JSON Content:"
# Write-Output $content

# # Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)
# Write-Output "Parsed JSON Array:"
# $jarray

# # Check the structure of the JSON array
# Write-Output "JSON Array Type:"
# $jarray.GetType()
# Write-Output "Number of Items in JSON Array:"
# $jarray.Count
# Write-Output "Types of Items in JSON Array:"
# $jarray | ForEach-Object { $_.GetType().FullName }
# Write-Output "User Principal Names in JSON Array:"
# $jarray | ForEach-Object { $_.userPrincipalName }

# # Search for the user with the specified userPrincipalName
# $users = $jarray | Where-Object { $_.userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' }
# Write-Output "Matching Users:"
# $users

# # Count the matches
# $count = $users.Count
# Write-Output "Count of Matching Users:"
# $count












# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# $jarray | ForEach-Object {
#     Write-Output "UserPrincipalName: $($_.userPrincipalName)"
#     Write-Output "UserDisplayName: $($_.userDisplayName)"
# }

# # Step 4: Search for the specific userPrincipalName
# $users = $jarray | Where-Object { $_.userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' }

# # Step 5: Count the matches
# $count = $users.Count

# # Output the count
# $count











# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = $userObject["userPrincipalName"]
#     $userDisplayName = $userObject["userDisplayName"]
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Step 4: Search for the specific userPrincipalName
# $users = $jarray | Where-Object {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$_
#     $userObject["userPrincipalName"] -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Step 5: Count the matches
# $count = $users.Count

# # Output the count
# $count







# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userDisplayName = [string]$userObject["userDisplayName"]
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Step 4: Search for the specific userPrincipalName
# $users = $jarray | Where-Object {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$_
#     [string]$userObject["userPrincipalName"] -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Step 5: Count the matches
# $count = $users.Count

# # Output the count
# $count











# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userDisplayName = [string]$userObject["userDisplayName"]
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Step 4: Search for the specific userPrincipalName
# $users = $jarray | Where-Object {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$_
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Step 5: Count the matches
# $count = $users.Count

# # Output the count
# $count







# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userDisplayName = [string]$userObject["userDisplayName"]
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Step 4: Search for the specific userPrincipalName
# $users = $jarray | Where-Object {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$_
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Step 5: Count the matches
# $count = ($users | Measure-Object).Count

# # Output the count
# $count










# # Step 1: Load the JSON file content
# $reader = [System.IO.StreamReader]::new("C:\log.json")
# $content = $reader.ReadToEnd()
# $reader.Close()

# # Step 2: Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($content)

# # Step 3: Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = [string]$userObject["userPrincipalName"]
#     $userDisplayName = [string]$userObject["userDisplayName"]
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Step 4: Search for the specific userPrincipalName
# $filteredUsers = @()
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     if ([string]$userObject["userPrincipalName"] -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers += $userObject
#     }
# }

# # Step 5: Count the matches
# $count = $filteredUsers.Count

# # Output the count
# $count








# # Load the JSON file content
# $jsonContent = Get-Content -Path "C:\log.json" -Raw

# # Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($jsonContent)

# # Filter the specific userPrincipalName
# $filteredUsers = $jarray | Where-Object {
#     $_["userPrincipalName"] -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user["userPrincipalName"])"
#     Write-Output "UserDisplayName: $($user["userDisplayName"])"
# }

# # Count the matches
# $count = $filteredUsers.Count
# Write-Output "Count of Matching Users: $count"











# # Load the JSON file content
# $jsonContent = Get-Content -Path "C:\log.json" -Raw

# # Parse the JSON content
# $jarray = [Newtonsoft.Json.Linq.JArray]::Parse($jsonContent)

# # Output each item's properties to verify structure
# foreach ($item in $jarray) {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$item
#     $userPrincipalName = $userObject["userPrincipalName"].Value
#     $userDisplayName = $userObject["userDisplayName"].Value
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
# }

# # Filter the specific userPrincipalName
# $filteredUsers = $jarray | Where-Object {
#     $userObject = [Newtonsoft.Json.Linq.JObject]$_
#     $userPrincipalName = $userObject["userPrincipalName"].Value
#     $userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca'
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"





# # Step 1: Load the JSON file content
# $jsonContent = Get-Content -Path "C:\log.json" -Raw

# # Step 2: Parse the JSON content
# $jarray = $jsonContent | ConvertFrom-Json

# # Step 3: Filter the specific userPrincipalName
# $filteredUsers = $jarray | Where-Object { $_.userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Step 4: Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"










# # Load the JSON file content
# $jsonContent = [System.IO.File]::ReadAllText("C:\log.json")

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users
# $filteredUsers = @()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers += [PSCustomObject]@{
#             userPrincipalName = $userPrincipalName
#             userDisplayName = $userDisplayName
#         }
#     }
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"


















# # Load the JSON file content
# $jsonContent = [System.IO.File]::ReadAllText("C:\log.json")

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[PSCustomObject]]::new()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers.Add([PSCustomObject]@{
#             userPrincipalName = $userPrincipalName
#             userDisplayName = $userDisplayName
#         })
#     }
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"







# # Load the JSON file content using .NET methods directly for efficiency
# $jsonContent = [System.IO.File]::ReadAllText("C:\log.json")

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[PSCustomObject]]::new()

# # Iterate over each element in the JSON array using Span<T> for in-place processing
# $enumerator = $rootElement.EnumerateArray()

# while ($enumerator.MoveNext()) {
#     $element = $enumerator.Current
#     if ($element.TryGetProperty("userPrincipalName", [ref]$null) -and
#         $element.GetProperty("userPrincipalName").GetString() -eq 'r.fakhouri@ictc-ctic.ca') {
        
#         $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#         $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
#         $filteredUsers.Add([PSCustomObject]@{
#             userPrincipalName = $userPrincipalName
#             userDisplayName = $userDisplayName
#         })
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"






# # Load the JSON file content using FileStream and StreamReader for efficiency
# $path = "C:\log.json"
# $fileStream = [System.IO.FileStream]::new($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
# $streamReader = [System.IO.StreamReader]::new($fileStream)
# $jsonContent = $streamReader.ReadToEnd()
# $streamReader.Close()
# $fileStream.Close()

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[PSCustomObject]]::new()

# # Iterate over each element in the JSON array using Span<T> for in-place processing
# $enumerator = $rootElement.EnumerateArray()

# while ($enumerator.MoveNext()) {
#     $element = $enumerator.Current
#     if ($element.TryGetProperty("userPrincipalName", [ref]$null) -and
#         $element.GetProperty("userPrincipalName").GetString() -eq 'r.fakhouri@ictc-ctic.ca') {
        
#         $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#         $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
#         $filteredUsers.Add([PSCustomObject]@{
#             userPrincipalName = $userPrincipalName
#             userDisplayName = $userDisplayName
#         })
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"





# # Load the JSON file content using FileStream and StreamReader for efficiency
# $path = "C:\log.json"
# $fileStream = [System.IO.FileStream]::new($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
# $streamReader = [System.IO.StreamReader]::new($fileStream)
# $jsonContent = $streamReader.ReadToEnd()
# $streamReader.Close()
# $fileStream.Close()

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[PSCustomObject]]::new()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers.Add([PSCustomObject]@{
#             userPrincipalName = $userPrincipalName
#             userDisplayName = $userDisplayName
#         })
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"












# # Define the custom class for user data
# class User {
#     [string]$userPrincipalName
#     [string]$userDisplayName

#     User([string]$userPrincipalName, [string]$userDisplayName) {
#         $this.userPrincipalName = $userPrincipalName
#         $this.userDisplayName = $userDisplayName
#     }
# }

# # Load the JSON file content using FileStream and StreamReader for efficiency
# $path = "C:\log.json"
# $fileStream = [System.IO.FileStream]::new($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
# $streamReader = [System.IO.StreamReader]::new($fileStream)
# $jsonContent = $streamReader.ReadToEnd()
# $streamReader.Close()
# $fileStream.Close()

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[User]]::new()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"














# # Define the custom class for user data
# class User {
#     [string]$userPrincipalName
#     [string]$userDisplayName

#     User([string]$userPrincipalName, [string]$userDisplayName) {
#         $this.userPrincipalName = $userPrincipalName
#         $this.userDisplayName = $userDisplayName
#     }
# }

# # Define the JSON file path
# $JsonFilePath = "C:\log.json"

# # Load the JSON file content using FileStream and StreamReader with SequentialScan for efficiency
# $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
# $streamReader = [System.IO.StreamReader]::new($fileStream)
# $jsonContent = $streamReader.ReadToEnd()
# $streamReader.Close()
# $fileStream.Close()

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[User]]::new()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
#         $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"












# # Define the custom class for user data
# class User {
#     [string]$userPrincipalName
#     [string]$userDisplayName

#     User([string]$userPrincipalName, [string]$userDisplayName) {
#         $this.userPrincipalName = $userPrincipalName
#         $this.userDisplayName = $userDisplayName
#     }
# }

# # Define the JSON file path
# $JsonFilePath = "C:\log.json"

# # Load the JSON file content using FileStream and StreamReader with SequentialScan for efficiency
# $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
# $streamReader = [System.IO.StreamReader]::new($fileStream)
# $jsonContent = $streamReader.ReadToEnd()
# $streamReader.Close()
# $fileStream.Close()

# # Parse the JSON content using System.Text.Json
# $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
# $rootElement = $jsonDocument.RootElement

# # Define a list to hold filtered users using a more efficient collection
# $filteredUsers = [System.Collections.Generic.List[User]]::new()

# # Define a HashSet to keep track of unique userPrincipalNames
# $uniqueUserPrincipalNames = [System.Collections.Generic.HashSet[string]]::new()

# # Iterate over each element in the JSON array
# foreach ($element in $rootElement.EnumerateArray()) {
#     $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
#     $userDisplayName = $element.GetProperty("userDisplayName").GetString()
    
#     # Output each item's properties to verify structure
#     Write-Output "UserPrincipalName: $userPrincipalName"
#     Write-Output "UserDisplayName: $userDisplayName"
    
#     # Filter the specific userPrincipalName
#     if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' -and $uniqueUserPrincipalNames.Add($userPrincipalName)) {
#         $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
#     }
# }

# # Output each matched user's properties
# foreach ($user in $filteredUsers) {
#     Write-Output "UserPrincipalName: $($user.userPrincipalName)"
#     Write-Output "UserDisplayName: $($user.userDisplayName)"
# }

# # Count the matches
# $count = $filteredUsers.Count

# # Output the count
# Write-Output "Count of Matching Users: $count"








# Define the custom class for user data
class User {
    [string]$userPrincipalName
    [string]$userDisplayName

    User([string]$userPrincipalName, [string]$userDisplayName) {
        $this.userPrincipalName = $userPrincipalName
        $this.userDisplayName = $userDisplayName
    }
}

# Define the JSON file path
$JsonFilePath = "C:\Code\CB\Entra\ICTC\Entra\Devices\Beta\CustomExports\CustomSignInlogs\log.json"

# Method 1: Using PSCustomObject with List
function Method1 {
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $streamReader = [System.IO.StreamReader]::new($fileStream)
    $jsonContent = $streamReader.ReadToEnd()
    $streamReader.Close()
    $fileStream.Close()

    $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
    $rootElement = $jsonDocument.RootElement

    $filteredUsers = [System.Collections.Generic.List[PSCustomObject]]::new()

    foreach ($element in $rootElement.EnumerateArray()) {
        $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
        $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
        if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
            $filteredUsers.Add([PSCustomObject]@{
                userPrincipalName = $userPrincipalName
                userDisplayName = $userDisplayName
            })
        }
    }
}

# Method 2: Using Custom Class with List
function Method2 {
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $streamReader = [System.IO.StreamReader]::new($fileStream)
    $jsonContent = $streamReader.ReadToEnd()
    $streamReader.Close()
    $fileStream.Close()

    $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
    $rootElement = $jsonDocument.RootElement

    $filteredUsers = [System.Collections.Generic.List[User]]::new()

    foreach ($element in $rootElement.EnumerateArray()) {
        $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
        $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
        if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca') {
            $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
        }
    }
}

# Method 3: Using Custom Class with List and HashSet for uniqueness
function Method3 {
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $streamReader = [System.IO.StreamReader]::new($fileStream)
    $jsonContent = $streamReader.ReadToEnd()
    $streamReader.Close()
    $fileStream.Close()

    $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
    $rootElement = $jsonDocument.RootElement

    $filteredUsers = [System.Collections.Generic.List[User]]::new()
    $uniqueUserPrincipalNames = [System.Collections.Generic.HashSet[string]]::new()

    foreach ($element in $rootElement.EnumerateArray()) {
        $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
        $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
        if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' -and $uniqueUserPrincipalNames.Add($userPrincipalName)) {
            $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
        }
    }
}






# Method 4: Using Custom Class with List and Dictionary for uniqueness
function Method4 {
    $fileStream = [System.IO.FileStream]::new($JsonFilePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read, 4096, [System.IO.FileOptions]::SequentialScan)
    $streamReader = [System.IO.StreamReader]::new($fileStream)
    $jsonContent = $streamReader.ReadToEnd()
    $streamReader.Close()
    $fileStream.Close()

    $jsonDocument = [System.Text.Json.JsonDocument]::Parse($jsonContent)
    $rootElement = $jsonDocument.RootElement

    $filteredUsers = [System.Collections.Generic.List[User]]::new()
    $uniqueUserPrincipalNames = [System.Collections.Generic.Dictionary[string, bool]]::new()

    foreach ($element in $rootElement.EnumerateArray()) {
        $userPrincipalName = $element.GetProperty("userPrincipalName").GetString()
        $userDisplayName = $element.GetProperty("userDisplayName").GetString()
        
        if ($userPrincipalName -eq 'r.fakhouri@ictc-ctic.ca' -and -not $uniqueUserPrincipalNames.ContainsKey($userPrincipalName)) {
            $uniqueUserPrincipalNames[$userPrincipalName] = $true
            $filteredUsers.Add([User]::new($userPrincipalName, $userDisplayName))
        }
    }
}

# Benchmark each method
function Benchmark-Method {
    param (
        [scriptblock]$Method
    )

    $iterations = 10
    $stopwatch = [System.Diagnostics.Stopwatch]::new()
    $totalElapsed = [System.TimeSpan]::Zero

    for ($i = 0; $i -lt $iterations; $i++) {
        $stopwatch.Restart()
        & $Method
        $stopwatch.Stop()
        $totalElapsed += $stopwatch.Elapsed
    }

    $averageElapsed = $totalElapsed.TotalMilliseconds / $iterations
    return $averageElapsed
}

# # Run benchmarks
# $avgMethod1 = Benchmark-Method { Method1 }
# $avgMethod2 = Benchmark-Method { Method2 }
# $avgMethod3 = Benchmark-Method { Method3 }

# # Output results
# "Average execution time for Method1 (PSCustomObject with List): $avgMethod1 ms"
# "Average execution time for Method2 (Custom Class with List): $avgMethod2 ms"
# "Average execution time for Method3 (Custom Class with List and HashSet): $avgMethod3 ms"








# # Define the JSON file path
# $JsonFilePath = "C:\log.json"



# Benchmark each method
# function Benchmark-Method {
#     param (
#         [scriptblock]$Method
#     )

#     $iterations = 1000
#     $stopwatch = [System.Diagnostics.Stopwatch]::new()
#     $totalElapsed = [System.TimeSpan]::Zero

#     for ($i = 0; $i -lt $iterations; $i++) {
#         $stopwatch.Restart()
#         & $Method
#         $stopwatch.Stop()
#         $totalElapsed += $stopwatch.Elapsed
#     }

#     $averageElapsed = $totalElapsed.TotalMilliseconds / $iterations
#     return $averageElapsed
# }

# Run benchmarks
$avgMethod1 = Benchmark-Method { Method1 }
$avgMethod2 = Benchmark-Method { Method2 }
$avgMethod3 = Benchmark-Method { Method3 }
$avgMethod4 = Benchmark-Method { Method4 }

# Output results
"Average execution time for Method1 (PSCustomObject with List): $avgMethod1 ms"
"Average execution time for Method2 (Custom Class with List): $avgMethod2 ms"
"Average execution time for Method3 (Custom Class with List and HashSet): $avgMethod3 ms"
"Average execution time for Method4 (Custom Class with List and Dictionary): $avgMethod4 ms"