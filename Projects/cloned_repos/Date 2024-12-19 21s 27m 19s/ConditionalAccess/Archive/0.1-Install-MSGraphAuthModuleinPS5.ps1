# $Graph_Modulename = "microsoft.graph.Beta.Identity.SignIns"
$Graph_Modulename = "Microsoft.Graph.Beta.Identity.SignIns"
$module = Import-Module $Graph_Modulename -PassThru -ErrorAction Ignore

# If the module is not imported, install it
if (-not $module) {
    Write-Host "Installing module $Graph_Modulename"
    Install-Module $Graph_Modulename -Force -Scope AllUsers -ErrorAction Ignore
    Install-Module Microsoft.Graph.Groups -Force -Scope AllUsers -ErrorAction Ignore
    # Install-Module microsoft.graph.authentication -Force -ErrorAction Ignore
    Write-Host "Module $Graph_Modulename installed successfully"
}