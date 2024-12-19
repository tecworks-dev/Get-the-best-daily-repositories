# Ensure you are authenticated
Connect-MgGraph -Scopes "Directory.Read.All"

# Fetch organization details using the Graph cmdlet
$organization = Get-MgOrganization

# Output the organization details
$organization | Format-List
