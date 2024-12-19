# relative v1.0 URI
Invoke-MgGraphRequest -Uri "v1.0/auditLogs/directoryAudits?filter=activityDateTime ge $startDate" -OutputType PSObject

# absolute beta URI, special (PATCH) request method
Invoke-MgGraphRequest -Method PATCH -Uri "https://graph.microsoft.com/beta/identity/conditionalAccess/policies" -Body $body

# relative URI with variable instead of ID
Invoke-MgGraphRequest -Uri "v1.0/groups/$($group.Id)/settings" -Method POST -Body $json -ContentType "application/json"

# URI defined via positional argument
Invoke-MgGraphRequest -OutputType PSObject "https://graph.microsoft.com/v1.0/devices"

# non resolvable URI
Invoke-MgGraphRequest -OutputType PSObject -Uri $msGraphPermissionsRequestUri

# official Mg command (read operation)
Get-MgApplication -ApplicationId "123456"

# official Mg command (write operation)
Update-MgApplication

# Invoke-MSGraphRequest
Invoke-MSGraphRequest -Url 'https://graph.microsoft.com/beta/servicePrincipals?$select=id'

# Invoke-WebRequest
Invoke-WebRequest "https://graph.microsoft.com/v1.0/devices"

# Invoke-RestMethod
Invoke-RestMethod -Uri "https://graph.microsoft.com/v1.0/devices" -header $header

# using Invoke-RestMethod alias
irm -Uri "https://graph.microsoft.com/v1.0/users" -header $header

# dependant function that has its own Graph API calls
Remove-O365OrphanedMailbox