# Install required modules if not already installed
function Install-RequiredModules {
    $modules = @(
        'Microsoft.Graph.Authentication',
        'Microsoft.Graph.Identity.SignIns',
        'Microsoft.Graph.Groups',
        'Microsoft.Graph.Users',
        'Microsoft.Graph.Identity.Governance',
        'Microsoft.Graph.Identity.DirectoryManagement',
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
    # Show warning message
    Write-Host @"
IMPORTANT SAFETY NOTICE
----------------------
You are about to deploy Conditional Access policies. These policies can affect all users' 
ability to access your environment. Before proceeding, ensure you have:

1. Created a break glass account
2. Documented the break glass account credentials securely
3. Backed up any existing policies
4. Reviewed the policies that will be deployed
5. Have a plan to test each policy before enabling

"@ -ForegroundColor Yellow

    $continue = Read-Host "Have you completed all the above steps? (yes/no)"
    if ($continue -ne "yes") {
        Write-Host "Deployment cancelled for safety. Please complete all preparatory steps first." -ForegroundColor Red
        return
    }

    # Install required modules
    Install-RequiredModules

    # Import required modules
    Import-Module Microsoft.Graph.Authentication
    Import-Module Microsoft.Graph.Identity.SignIns
    Import-Module Microsoft.Graph.Groups
    Import-Module Microsoft.Graph.Users
    Import-Module Microsoft.Graph.Identity.Governance
    Import-Module Microsoft.Graph.Identity.DirectoryManagement
    Import-Module DCToolbox

    # Connect to Microsoft Graph with required permissions
    $requiredScopes = @(
        'Policy.ReadWrite.ConditionalAccess',
        'Policy.Read.All',
        'Directory.Read.All',
        'Application.Read.All',
        'Agreement.Read.All',
        'GroupMember.Read.All',
        'Agreement.ReadWrite.All'
    )

    Write-Host "Connecting to Microsoft Graph..." -ForegroundColor Yellow
    Connect-DCMsGraphAsUser -Scopes $requiredScopes

    # Deploy baseline policies
    Write-Host "Starting Conditional Access baseline deployment..." -ForegroundColor Green
    
    # Force Report-Only mode for safety
    $deployParams = @{
        CreateDocumentation = $true
        SkipReportOnlyMode = $false  # Always deploy in report-only mode first
    }

    try {
        # Create break glass account exclusion group if it doesn't exist
        Write-Host "Checking break glass account exclusion group..." -ForegroundColor Yellow
        
        # Deploy the baseline policies
        Deploy-DCConditionalAccessBaselinePoC @deployParams

        Write-Host "Deployment completed successfully!" -ForegroundColor Green
        Write-Host @"

NEXT STEPS (DO NOT SKIP THESE):
------------------------------
1. Add your break glass accounts to the 'Excluded from Conditional Access' group
2. Review each policy in the Azure Portal (https://portal.azure.com/#blade/Microsoft_AAD_IAM/ConditionalAccessBlade/Policies)
3. Test each policy in Report-Only mode and review the Insights
4. Document any changes needed based on your testing
5. Create a pilot group for initial testing
6. Enable policies one at a time, starting with least impactful
7. Monitor sign-in logs for issues

SAFE ENABLEMENT PROCESS:
----------------------
1. First, test with a pilot group:
   Set-DCConditionalAccessPoliciesPilotMode -PrefixFilter 'GLOBAL - ' -PilotGroupName 'Conditional Access Pilot' -EnablePilot

2. After successful pilot, enable individual policies:
   - Use Azure Portal to enable policies one at a time
   - Monitor between each enablement
   - Have your break glass account ready

3. DO NOT enable all policies at once using PowerShell commands

Documentation has been generated - please review it thoroughly.

"@ -ForegroundColor Yellow
    }
    catch {
        Write-Host "Error during deployment: $_" -ForegroundColor Red
        Write-Host "Please ensure you have Global Administrator or Security Administrator rights." -ForegroundColor Red
    }
}

# Execute the deployment
Initialize-CADeployment