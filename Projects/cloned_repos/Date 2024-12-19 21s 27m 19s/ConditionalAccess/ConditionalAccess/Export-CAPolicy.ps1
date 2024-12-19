<#
    .SYNOPSIS
	Conditional Access Export Utility with Microsoft Graph PowerShell (MgGraph).

    .DESCRIPTION
	Exports CA Policy to HTML Format for auditing/historical purposes.
#>

# ExportLocation
$ExportLocation = "C:\temp\CAPolicy.html"

# Test Graph Module
$GraphModule = Get-Module "Microsoft.Graph" -ListAvailable
If ($null -eq $GraphModule) {
    Write-Host "Microsoft.Graph Module not installed" -ForegroundColor Yellow
    Write-Host "Use: Install-Module -Name Microsoft.Graph" -ForegroundColor Yellow
    break
}

# Connect-MgGraph
$MgContext = Get-MgContext
If ($null -eq $MgContext) {
    Write-host "Connect-MgGraph"
    Connect-MgGraph -Scopes 'Policy.Read.All', 'Directory.Read.All', 'Application.Read.All'
}
else {
    Write-host "Disconnect-MgGraph"
    Disconnect-MgGraph -ErrorAction SilentlyContinue | Out-Null
    Write-host "Connect-MgGraph"
    Connect-MgGraph -Scopes 'Policy.Read.All', 'Directory.Read.All', 'Application.Read.All'
}

# Collect CA Policy
# $CAPolicy = Get-MgIdentityConditionalAccessPolicy -All
$CAPolicy = Get-MgBetaIdentityConditionalAccessPolicy -All

if (-not $CAPolicy) {
    Write-Host "No CA policies found. Stopping script." -ForegroundColor Red
    return
}

# Tenant Informations
$TenantData = Get-MgOrganization
$TenantName = $TenantData.DisplayName
$date = Get-Date

Write-Host "Exporting: CA Policy" -ForegroundColor Cyan
$CAExport = [PSCustomObject]@()

$AdUsers = @()
$Apps = @()
# Extract Values
Write-host "Extracting: CA Policy Data" -ForegroundColor Cyan
foreach ( $Policy in $CAPolicy) {
    ### Conditions ###
    $IncludeUG = $null
    $IncludeUG = $Policy.Conditions.Users.IncludeUsers
    $IncludeUG += $Policy.Conditions.Users.IncludeGroups
    $IncludeUG += $Policy.Conditions.Users.IncludeRoles

    $ExcludeUG = $null
    $ExcludeUG = $Policy.Conditions.Users.ExcludeUsers
    $ExcludeUG += $Policy.Conditions.Users.ExcludeGroups
    $ExcludeUG += $Policy.Conditions.Users.ExcludeRoles

    $Apps += $Policy.Conditions.Applications.IncludeApplications
    $Apps += $Policy.Conditions.Applications.ExcludeApplications

    $AdUsers += $ExcludeUG
    $AdUsers += $IncludeUG

    $InclLocation = $null
    $ExclLocation = $null
    $InclLocation = $Policy.Conditions.Locations.includelocations
    $ExclLocation = $Policy.Conditions.Locations.Excludelocations

    $InclPlat = $null
    $ExclPlat = $null
    $InclPlat = $Policy.Conditions.Platforms.IncludePlatforms
    $ExclPlat = $Policy.Conditions.Platforms.ExcludePlatforms
    $InclDev = $null
    $ExclDev = $null
    $InclDev = $Policy.Conditions.Devices.IncludeDevices
    $ExclDev = $Policy.Conditions.Devices.ExcludeDevices
    $devFilters = $null
    $devFilters = $Policy.Conditions.Devices.DeviceFilter.Rule

    $CAExport += New-Object PSObject -Property @{
        ### Users ###
        Users                                               = ""
        Name                                                = $Policy.DisplayName;
        PolicyID                                            = $Policy.ID
        Status                                              = $Policy.State;
        UsersInclude                                        = ($IncludeUG -join ", `r`n");
        UsersExclude                                        = ($ExcludeUG -join ", `r`n");
        ### Cloud apps or actions ###
        'Cloud apps or actions'                             = "";
        ApplicationsIncluded                                = ($Policy.Conditions.Applications.IncludeApplications -join ", `r`n");
        ApplicationsExcluded                                = ($Policy.Conditions.Applications.ExcludeApplications -join ", `r`n");
        userActions                                         = ($Policy.Conditions.Applications.IncludeUserActions -join ", `r`n");
        AuthContext                                         = ($Policy.Conditions.Applications.IncludeAuthenticationContextClassReferences -join ", `r`n");
        ### Conditions ###
        Conditions                                          = "";
        UserRisk                                            = ($Policy.Conditions.UserRiskLevels -join ", `r`n");
        SignInRisk                                          = ($Policy.Conditions.SignInRiskLevels -join ", `r`n");
        PlatformsInclude                                    = ($InclPlat -join ", `r`n");
        PlatformsExclude                                    = ($ExclPlat -join ", `r`n");
        LocationsIncluded                                   = ($InclLocation -join ", `r`n");
        LocationsExcluded                                   = ($ExclLocation -join ", `r`n");
        ClientApps                                          = ($Policy.Conditions.ClientAppTypes -join ", `r`n");
        DevicesIncluded                                     = ($InclDev -join ", `r`n");
        DevicesExcluded                                     = ($ExclDev -join ", `r`n");
        DeviceFilters                                       = ($devFilters -join ", `r`n");

        ### Grant Controls ###
        GrantControls                                       = "";
        BuiltInControls                                     = $($Policy.GrantControls.BuiltInControls)
        TermsOfUse                                          = $($Policy.GrantControls.TermsOfUse)
        CustomControls                                      = $($Policy.GrantControls.CustomAuthenticationFactors)
        GrantOperator                                       = $Policy.GrantControls.Operator

        ### Session Controls ###
        SessionControls                                     = ""
        SessionControlsAdditionalProperties                 = $Policy.SessionControls.AdditionalProperties
        ApplicationEnforcedRestrictionsIsEnabled            = $Policy.SessionControls.ApplicationEnforcedRestrictions.IsEnabled
        ApplicationEnforcedRestrictionsAdditionalProperties = $Policy.SessionControls.ApplicationEnforcedRestrictions.AdditionalProperties
        CloudAppSecurityType                                = $Policy.SessionControls.CloudAppSecurity.CloudAppSecurityType
        CloudAppSecurityIsEnabled                           = $Policy.SessionControls.CloudAppSecurity.IsEnabled
        CloudAppSecurityAdditionalProperties                = $Policy.SessionControls.CloudAppSecurity.AdditionalProperties
        DisableResilienceDefaults                           = $Policy.SessionControls.DisableResilienceDefaults
        PersistentBrowserIsEnabled                          = $Policy.SessionControls.PersistentBrowser.IsEnabled
        PersistentBrowserMode                               = $Policy.SessionControls.PersistentBrowser.Mode
        PersistentBrowserAdditionalProperties               = $Policy.SessionControls.PersistentBrowser.AdditionalProperties
        SignInFrequencyAuthenticationType                   = $Policy.SessionControls.SignInFrequency.AuthenticationType
        SignInFrequencyInterval                             = $Policy.SessionControls.SignInFrequency.FrequencyInterval
        SignInFrequencyIsEnabled                            = $Policy.SessionControls.SignInFrequency.IsEnabled
        SignInFrequencyType                                 = $Policy.SessionControls.SignInFrequency.Type
        SignInFrequencyValue                                = $Policy.SessionControls.SignInFrequency.Value
        SignInFrequencyAdditionalProperties                 = $Policy.SessionControls.SignInFrequency.AdditionalProperties


    }
}

# Swith user/group Guid to display names
Write-host "Converting: Entra ID Guids" -ForegroundColor Cyan
# Filter out Objects
$cajson = $CAExport | ConvertTo-Json -Depth 4
$ADsearch = $AdUsers | Where-Object { $_ -ne 'All' -and $_ -ne 'GuestsOrExternalUsers' -and $_ -ne 'None' }
$AdNames = @{}
Get-MgDirectoryObjectById -ids $ADsearch | ForEach-Object {
    $obj = $_.Id
    #$disp = $_.displayName
    $disp = $_.AdditionalProperties.displayName
    $AdNames.$obj = $disp
    $cajson = $cajson -replace "$obj", "$disp"
}
$CAExport = $cajson | ConvertFrom-Json
# Switch Apps Guid with Display names
$allApps = Get-MgServicePrincipal -All
$allApps | Where-Object { $_.AppId -in $Apps } | ForEach-Object {
    $obj = $_.AppId
    $disp = $_.DisplayName
    $cajson = $cajson -replace "$obj", "$disp"
}
# Switch named location Guid for Display Names
Get-MgIdentityConditionalAccessNamedLocation | ForEach-Object {
    $obj = $_.Id
    $disp = $_.DisplayName
    $cajson = $cajson -replace "$obj", "$disp"
}
# Switch Roles Guid to Names
#Get-MgDirectoryRole | ForEach-Object{
Get-MgDirectoryRoleTemplate | ForEach-Object {
    $obj = $_.Id
    $disp = $_.DisplayName
    $cajson = $cajson -replace "$obj", "$disp"
}
$CAExport = $cajson | ConvertFrom-Json

# Export Setup
Write-host "Pivoting: CA to Export Format" -ForegroundColor Cyan
$pivot = @()
$rowItem = New-Object PSObject
$rowitem | Add-Member -type NoteProperty -Name 'CA Item' -Value "row1"
$Pcount = 1
foreach ($CA in $CAExport) {
    $rowitem | Add-Member -type NoteProperty -Name "Policy $pcount" -Value "row1"
    #$ca.Name
    $pcount += 1
}
$pivot += $rowItem

# Add Data to Report
$Rows = $CAExport | Get-Member | Where-Object { $_.MemberType -eq "NoteProperty" }
$Rows | ForEach-Object {
    $rowItem = New-Object PSObject
    $rowname = $_.Name
    $rowitem | Add-Member -type NoteProperty -Name 'CA Item' -Value $_.Name
    $Pcount = 1
    foreach ($CA in $CAExport) {
        $ca | Get-Member | Where-Object { $_.MemberType -eq "NoteProperty" } | ForEach-Object {
            $a = $_.name
            $b = $ca.$a
            if ($a -eq $rowname) {
                $rowitem | Add-Member -type NoteProperty -Name "Policy $pcount" -Value $b
            }
        }
        $pcount += 1
    }
    $pivot += $rowItem
}

# Column Sorting Order
$sort = "Name", "PolicyID", "Status", "Users", "UsersInclude", "UsersExclude", "Cloud apps or actions", "ApplicationsIncluded", "ApplicationsExcluded", `
    "userActions", "AuthContext", "Conditions", "UserRisk", "SignInRisk", "PlatformsInclude", "PlatformsExclude", "LocationsIncluded", `
    "LocationsExcluded", "ClientApps", "Devices", "DevicesIncluded", "DevicesExcluded", "DeviceFilters", `
    "GrantControls", "BuiltInControls", "TermsOfUse", "CustomControls", "GrantOperator", `
    "SessionControls", "SessionControlsAdditionalProperties", "ApplicationEnforcedRestrictionsIsEnabled", "ApplicationEnforcedRestrictionsAdditionalProperties", `
    "CloudAppSecurityType", "CloudAppSecurityIsEnabled", "CloudAppSecurityAdditionalProperties", "DisableResilienceDefaults", "PersistentBrowserIsEnabled", `
    "PersistentBrowserMode", "PersistentBrowserAdditionalProperties", "SignInFrequencyAuthenticationType", "SignInFrequencyInterval", "SignInFrequencyIsEnabled", `
    "SignInFrequencyType", "SignInFrequencyValue", "SignInFrequencyAdditionalProperties"

# Debug
#$pivot | Sort-Object $sort | Out-GridView

# HTML Export
Write-host "Saving to File: HTML in $ExportLocation" -ForegroundColor Cyan
$jquery = '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
	<script>
	$(document).ready(function(){
		$("tr").click(function(){
		if(!$(this).hasClass("selected")){
			$(this).addClass("selected");
		} else {
			$(this).removeClass("selected");
		}

		});
		$("th").click(function(){
		if(!$(this).hasClass("colselected")){
			$(this).addClass("colselected");
		} else {
			$(this).removeClass("colselected");
		}

		});
	});
	</script>'
$html = "<html><head><base href='https://docs.microsoft.com/' target='_blank'>
	$jquery<style>
	.title{
		display: block;
		font-size: 2em;
		margin-block-start: 0.67em;
		margin-block-end: 0.67em;
		margin-inline-start: 0px;
		margin-inline-end: 0px;
		font-weight: bold;
		font-family: Segoe UI;
	}
	table{
		border-collapse: collapse;
		margin: 25px 0;
		font-size: 0.9em;
		font-family: Segoe UI;
		min-width: 400px;
		box-shadow: 0 0 20px rgba(0, 0, 0, 0.15) ;
		text-align: center;
	}
	thead tr {
		background-color: #009879;
		color: #ffffff;
		text-align: left;
	}
	th, td {
		min-width: 250px;
		padding: 12px 15px;
		border: 1px solid lightgray;
		vertical-align: top;
	}
	td {
		vertical-align: top;
	}
	tbody tr {
		border-bottom: 1px solid #dddddd;
	}
	tbody tr:nth-of-type(even) {
		background-color: #f3f3f3;
	}
	tbody tr:nth-of-type(5), tbody tr:nth-of-type(8), body tr:nth-of-type(13), tbody tr:nth-of-type(24), tbody tr:nth-of-type(29){
		background-color: #36c;
		text-aling:left !important
	}
	tbody tr:last-of-type {
		border-bottom: 2px solid #009879;
	}
	tr:hover{
	background-color: #ffea76!important;
	}
	.selected:not(th){
		background-color:#ffea76!important;
	}
	th{
		background-color:white !important;
	}
	.colselected {
	background-color: rgb(93, 236, 213)!important;
	}
	table tr th:first-child,table tr td:first-child {
		position: sticky;
		inset-inline-start: 0;
		background-color: #36c!important;
		Color: #fff;
		font-weight: bolder;
		text-align: center;
	}
	</style></head><body> <div class='Title'>CA Export: $Tenantname - $Date </div>"

Write-host "Launching: Web Browser" -ForegroundColor Cyan
$Launch = $ExportLocation
$HTML += $pivot  | Where-Object { $_."CA Item" -ne 'row1' } | Sort-object { $sort.IndexOf($_."CA Item") } | convertto-html -Fragment
$HTML | Out-File $Launch
start-process $Launch