# Example JSON data
$Json = @(
    @{userDisplayName = 'User1'; userPrincipalName = 'user1@example.com'},
    @{userDisplayName = 'On-Premises Directory Synchronization Service Account'; userPrincipalName = 'sync@example.com'},
    @{userDisplayName = 'User2'; userPrincipalName = 'user2@example.com'}
)

# Convert the array to JSON and then back to an array of PSCustomObjects to simulate receiving JSON data
$JsonData = $Json | ConvertTo-Json | ConvertFrom-Json

# Process the JSON data in parallel
$JsonData | ForEach-Object -Parallel {
    # Simulate some processing for each user
    $user = $_
    $result = [PSCustomObject]@{
        UserDisplayName = $user.userDisplayName
        UserPrincipalName = $user.userPrincipalName
        ProcessedTime = (Get-Date).ToString()
    }
    # Output the result
    $result
} -ThrottleLimit 4
