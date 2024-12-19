# Install required modules if not already installed
function Install-RequiredModules {
    $modules = @(
        'Microsoft.Graph.Authentication',
        'Microsoft.Graph.Identity.SignIns',
        'Microsoft.Graph.Groups',
        'Microsoft.Graph.Users',
        'DCToolbox'
    )
    
    foreach ($module in $modules) {
        if (-not (Get-Module -ListAvailable -Name $module)) {
            Write-Host "Installing $module..." -ForegroundColor Yellow
            Install-Module -Name $module -Force -AllowClobber -Scope CurrentUser
        }
    }
}

function Initialize-CADeployment {
    # Install required modules
    Install-RequiredModules

    # Import required modules
    Import-Module Microsoft.Graph.Authentication
    Import-Module Microsoft.Graph.Identity.SignIns
    Import-Module Microsoft.Graph.Groups
    Import-Module Microsoft.Graph.Users
    Import-Module DCToolbox

    # Connect to Microsoft Graph with required permissions
    $requiredScopes = @(
        'Policy.ReadWrite.ConditionalAccess',
        'Policy.Read.All',
        'Directory.Read.All',
        'Application.Read.All',
        'Agreement.Read.All',
        'GroupMember.Read.All'
    )

    Write-Host "Connecting to Microsoft Graph..." -ForegroundColor Yellow
    Connect-DCMsGraphAsUser -Scopes $requiredScopes

    # Deploy baseline policies
    Write-Host "Starting Conditional Access baseline deployment..." -ForegroundColor Green
    
    # Parameters for deployment
    $deployParams = @{
        CreateDocumentation = $true    # Creates markdown documentation
        SkipReportOnlyMode = $false   # Keeps policies in report-only mode for safety
    }

    try {
        # Deploy the baseline policies
        Deploy-DCConditionalAccessBaselinePoC @deployParams

        Write-Host "Deployment completed successfully!" -ForegroundColor Green
        Write-Host @"

Important Next Steps:
1. Review the created policies in the Azure Portal
2. Check the documentation generated
3. Test the policies in report-only mode
4. Create and configure your break glass accounts
5. Add your break glass accounts to the exclusion group

Note: All policies are deployed in report-only mode by default for safety.
To enable them, use:
Set-DCConditionalAccessPoliciesReportOnlyMode -PrefixFilter 'GLOBAL - ' -SetToEnabled

"@ -ForegroundColor Yellow
    }
    catch {
        Write-Host "Error during deployment: $_" -ForegroundColor Red
        Write-Host "Please ensure you have Global Administrator or Security Administrator rights." -ForegroundColor Red
    }
}

# Execute the deployment
Initialize-CADeployment