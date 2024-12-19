# Functions for policy generation
function New-CAPolicyName {
    param (
        [string]$CategoryCode,
        [string]$SequenceNumber,
        [string]$Scope,
        [string]$Action,
        [string]$Condition,
        [string]$Clients,
        [string]$Version
    )
    
    "$CategoryCode$SequenceNumber - $($Scope.ToUpper()) - $($Action.ToUpper()) - $Condition - when - $Clients - v$Version.json"
}

function Export-CAPolicy {
    param (
        [string]$OutputPath = ".\CAPolicies.csv"
    )

    $policies = @(
        # Identity & Authentication Policies
        @{
            CategoryCode = "CAP"; SequenceNumber = "001"; Scope = "GLOBAL"; Action = "BLOCK"
            Condition = "LegacyAuth"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAP"; SequenceNumber = "002"; Scope = "GLOBAL"; Action = "BLOCK"
            Condition = "DeviceCodeAuth"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAP"; SequenceNumber = "003"; Scope = "ADMINS"; Action = "GRANT"
            Condition = "PhishingResistantMFADaily"; Clients = "BrowserModernAuthClients"; Version = "1.0"
        },

        # Risk-Based Policies - Sign-In Risk
        # Admins
        @{
            CategoryCode = "CAU"; SequenceNumber = "001"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "SignInRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "002"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "SignInRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "003"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "SignInRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },
        # Users
        @{
            CategoryCode = "CAU"; SequenceNumber = "004"; Scope = "USERS"; Action = "GRANT"
            Condition = "SignInRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "005"; Scope = "USERS"; Action = "BLOCK"
            Condition = "SignInRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "006"; Scope = "USERS"; Action = "BLOCK"
            Condition = "SignInRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },
        # Guests
        @{
            CategoryCode = "CAU"; SequenceNumber = "007"; Scope = "GUESTS"; Action = "GRANT"
            Condition = "SignInRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "008"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "SignInRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "009"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "SignInRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },

        # Risk-Based Policies - User Risk
        # Admins
        @{
            CategoryCode = "CAU"; SequenceNumber = "010"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "UserRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "011"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "UserRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "012"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "UserRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },
        # Users
        @{
            CategoryCode = "CAU"; SequenceNumber = "013"; Scope = "USERS"; Action = "GRANT"
            Condition = "UserRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "014"; Scope = "USERS"; Action = "BLOCK"
            Condition = "UserRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "015"; Scope = "USERS"; Action = "BLOCK"
            Condition = "UserRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },
        # Guests
        @{
            CategoryCode = "CAU"; SequenceNumber = "016"; Scope = "GUESTS"; Action = "GRANT"
            Condition = "UserRiskLow"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "017"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "UserRiskMedium"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAU"; SequenceNumber = "018"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "UserRiskHigh"; Clients = "AllClients"; Version = "1.0"
        },

        # Location-Based Access
        # Location-Based Access - Allowed Countries
        @{
            CategoryCode = "CAL"; SequenceNumber = "001"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "NonAllowedCountriesAndNonTrustedLocation"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "002"; Scope = "USERS"; Action = "BLOCK"
            Condition = "NonAllowedCountries"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "003"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "NonAllowedCountries"; Clients = "AllClients"; Version = "1.0"
        },
        # Location-Based Access - High Risk Countries
        @{
            CategoryCode = "CAL"; SequenceNumber = "004"; Scope = "ADMINS"; Action = "BLOCK"
            Condition = "HighRiskCountries"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "005"; Scope = "USERS"; Action = "BLOCK"
            Condition = "HighRiskCountries"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "006"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "HighRiskCountries"; Clients = "AllClients"; Version = "1.0"
        },
        # Location-Based Access - Trusted Locations
        @{
            CategoryCode = "CAL"; SequenceNumber = "007"; Scope = "ADMINS"; Action = "GRANT"
            Condition = "TrustedLocationsOnly"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "008"; Scope = "USERS"; Action = "GRANT"
            Condition = "TrustedLocationsOnly"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAL"; SequenceNumber = "009"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "NonTrustedLocations"; Clients = "AllClients"; Version = "1.0"
        },

        # Device-Based Policies
        @{
            CategoryCode = "CAD"; SequenceNumber = "001"; Scope = "GLOBAL"; Action = "BLOCK"
            Condition = "UnsupportedPlatforms"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAD"; SequenceNumber = "002"; Scope = "ADMINS"; Action = "GRANT"
            Condition = "RequireMDMManagedDevice"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAD"; SequenceNumber = "003"; Scope = "USERS"; Action = "GRANT"
            Condition = "RequireMDMOrMAM"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAD"; SequenceNumber = "004"; Scope = "GLOBAL"; Action = "SESSION"
            Condition = "BlockFileDownloadsUnmanaged"; Clients = "AllClients"; Version = "1.0"
        },

        # Guest-Specific Policies
        @{
            CategoryCode = "CAG"; SequenceNumber = "001"; Scope = "GUESTS"; Action = "BLOCK"
            Condition = "SensitiveApps"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAG"; SequenceNumber = "002"; Scope = "GUESTS"; Action = "GRANT"
            Condition = "RequireMFAEvery7Days"; Clients = "AllClients"; Version = "1.0"
        },

        # Registration & Enrollment
        @{
            CategoryCode = "CAT"; SequenceNumber = "001"; Scope = "GLOBAL"; Action = "SESSION"
            Condition = "RequireReAuthForMFARegistration"; Clients = "AllClients"; Version = "1.0"
        },

        # Break-Glass
        @{
            CategoryCode = "CAB"; SequenceNumber = "001"; Scope = "SELECTED"; Action = "GRANT"
            Condition = "EmergencyAccess"; Clients = "BrowserModernAuthClients"; Version = "1.0"
        },

        # Session Controls
        @{
            CategoryCode = "CAS"; SequenceNumber = "001"; Scope = "ADMINS"; Action = "SESSION"
            Condition = "9HourFrequencyNoPersistence"; Clients = "AllClients"; Version = "1.0"
        },
        @{
            CategoryCode = "CAS"; SequenceNumber = "002"; Scope = "USERS"; Action = "SESSION"
            Condition = "9HourFrequencyBYOD"; Clients = "UnmanagedDevices"; Version = "1.0"
        },
        @{
            CategoryCode = "CAS"; SequenceNumber = "003"; Scope = "GUESTS"; Action = "SESSION"
            Condition = "DefaultSessionControls"; Clients = "AllClients"; Version = "1.0"
        }
    )

    $output = foreach ($policy in $policies) {
        [PSCustomObject]@{
            PolicyName   = New-CAPolicyName @policy
            CategoryCode = $policy.CategoryCode
            Scope        = $policy.Scope
            Action       = $policy.Action
            Condition    = $policy.Condition
            Clients      = $policy.Clients
            Version      = $policy.Version
        }
    }

    # Export to CSV
    $output | Export-Csv -Path $OutputPath -NoTypeInformation

    # Generate HTML report
    $HTMLPath = $OutputPath.Replace('.csv', '.html')
    
    $metadata = @{
        GeneratedBy   = $env:USERNAME
        GeneratedOn   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        TotalPolicies = $output.Count
        Categories    = ($output | Group-Object CategoryCode | ForEach-Object { "$($_.Name): $($_.Count)" }) -join ", "
    }
    
    New-HTML -Title "Conditional Access Policies" -FilePath $HTMLPath -ShowHTML {
        New-HTMLSection -HeaderText "Generation Summary" {
            New-HTMLPanel {
                New-HTMLText -Text @"
                <h3>Report Details</h3>
                <ul>
                    <li>Generated By: $($metadata.GeneratedBy)</li>
                    <li>Generated On: $($metadata.GeneratedOn)</li>
                    <li>Total Policies: $($metadata.TotalPolicies)</li>
                    <li>Categories: $($metadata.Categories)</li>
                </ul>
"@
            }
        }
        
        New-HTMLSection -HeaderText "Conditional Access Policies" {
            New-HTMLTable -DataTable $output -ScrollX -Buttons @('copyHtml5', 'excelHtml5', 'csvHtml5') -SearchBuilder {
                New-TableCondition -Name 'Action' -ComparisonType string -Operator eq -Value 'BLOCK' -BackgroundColor Salmon -Color Black
                New-TableCondition -Name 'Action' -ComparisonType string -Operator eq -Value 'GRANT' -BackgroundColor LightGreen -Color Black
                New-TableCondition -Name 'Action' -ComparisonType string -Operator eq -Value 'SESSION' -BackgroundColor LightBlue -Color Black
                New-TableCondition -Name 'Scope' -ComparisonType string -Operator eq -Value 'ADMINS' -BackgroundColor LightYellow -Color Black
            }
        }
    }

    # Display summary in console
    Write-Host "`nPolicy Summary:" -ForegroundColor Cyan
    $output | Group-Object CategoryCode | ForEach-Object {
        Write-Host "$($_.Name): $($_.Count) policies"
    }

    Write-Host "`nOutput files generated:" -ForegroundColor Green
    Write-Host "CSV: $OutputPath"
    Write-Host "HTML: $HTMLPath"
}

# Execute the function
Export-CAPolicy