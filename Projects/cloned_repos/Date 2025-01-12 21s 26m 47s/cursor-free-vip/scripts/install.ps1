# 檢查是否是通過權限提升啟動的
param(
    [switch]$Elevated
)

# 設置顏色主題
$Theme = @{
    Primary   = 'Cyan'
    Success   = 'Green'
    Warning   = 'Yellow'
    Error     = 'Red'
    Info      = 'White'
}

# ASCII Logo
$Logo = @"
   ██████╗██╗   ██╗██████╗ ███████╗ ██████╗ ██████╗      ██████╗ ██████╗  ██████╗   
  ██╔════╝██║   ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗     ██╔══██╗██╔══██╗██╔═══██╗  
  ██║     ██║   ██║██████╔╝███████╗██║   ██║██████╔╝     ██████╔╝██████╔╝██║   ██║  
  ██║     ██║   ██║██╔══██╗╚════██║██║   ██║██╔══██╗     ██╔═══╝ ██╔══██╗██║   ██║  
  ╚██████╗╚██████╔╝██║  ██║███████║╚██████╔╝██║  ██║     ██║     ██║  ██║╚██████╔╝  
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝  
"@

# 進度條函數
function Write-ProgressBar {
    param (
        [int]$Percent,
        [string]$Activity
    )
    $width = $Host.UI.RawUI.WindowSize.Width - 20
    $completed = [math]::Floor($width * ($Percent / 100))
    $remaining = $width - $completed
    $progressBar = "[" + ("█" * $completed) + ("-" * $remaining) + "]"
    Write-Host "`r$Activity $progressBar $Percent%" -NoNewline
}

# 美化輸出函數
function Write-Styled {
    param (
        [string]$Message,
        [string]$Color = $Theme.Info,
        [string]$Prefix = "",
        [switch]$NoNewline
    )
    $emoji = switch ($Color) {
        $Theme.Success { "✅" }
        $Theme.Error   { "❌" }
        $Theme.Warning { "⚠️" }
        default        { "ℹ️" }
    }
    
    $output = if ($Prefix) { "$emoji $Prefix :: $Message" } else { "$emoji $Message" }
    if ($NoNewline) {
        Write-Host $output -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $output -ForegroundColor $Color
    }
}

# 檢查管理員權限
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-NOT $isAdmin) {
    Write-Styled "需要管理員權限來安裝" -Color $Theme.Warning -Prefix "權限"
    Write-Styled "正在請求管理員權限..." -Color $Theme.Primary -Prefix "提升"
    
    # 顯示操作選項
    Write-Host "`n選擇操作:" -ForegroundColor $Theme.Primary
    Write-Host "1. 請求管理員權限" -ForegroundColor $Theme.Info
    Write-Host "2. 退出程序" -ForegroundColor $Theme.Info
    
    $choice = Read-Host "`n請輸入選項 (1-2)"
    
    if ($choice -ne "1") {
        Write-Styled "安裝已取消" -Color $Theme.Warning -Prefix "取消"
        Write-Host "`n按任意鍵退出..." -ForegroundColor $Theme.Info
        $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        exit
    }
    
    $pwshPath = if (Get-Command "pwsh" -ErrorAction SilentlyContinue) {
        (Get-Command "pwsh").Source
    } elseif (Test-Path "$env:ProgramFiles\PowerShell\7\pwsh.exe") {
        "$env:ProgramFiles\PowerShell\7\pwsh.exe"
    } else {
        "powershell.exe"
    }
    
    try {
        $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`" -Elevated -WindowStyle Normal"
        Start-Process -FilePath $pwshPath -Verb RunAs -ArgumentList $arguments
        Write-Host "`n請在新開啟的管理員權限視窗中繼續操作..." -ForegroundColor $Theme.Primary
        
        # 等待用戶確認
        Write-Host "`n按任意鍵退出此窗口..." -ForegroundColor $Theme.Info
        $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        exit
    }
    catch {
        Write-Styled "無法獲取管理員權限" -Color $Theme.Error -Prefix "錯誤"
        Write-Styled "請以管理員身份運行 PowerShell 後重試" -Color $Theme.Warning -Prefix "提示"
        Write-Styled "您可以：" -Color $Theme.Info
        Write-Host "1. 右鍵點擊 PowerShell，選擇「以系統管理員身分執行」" -ForegroundColor $Theme.Info
        Write-Host "2. 然後重新運行此安裝程序" -ForegroundColor $Theme.Info
        Write-Host "`n按任意鍵退出..." -ForegroundColor $Theme.Info
        $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
        exit 1
    }
}

# 如果是提升權限後的窗口，等待一下確保窗口可見
if ($Elevated) {
    Start-Sleep -Seconds 1
}

# 獲取版本號函數
function Get-LatestVersion {
    try {
        $latestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/yeongpin/cursor-free-vip/releases/latest"
        return @{
            Version = $latestRelease.tag_name.TrimStart('v')
            Assets = $latestRelease.assets
        }
    } catch {
        Write-Styled $_.Exception.Message -Color $Theme.Error -Prefix "錯誤"
        throw "無法獲取最新版本信息"
    }
}

# 顯示 Logo
Write-Host $Logo -ForegroundColor $Theme.Primary
$releaseInfo = Get-LatestVersion
$version = $releaseInfo.Version
Write-Host "Version $version" -ForegroundColor $Theme.Info
Write-Host "Created by YeongPin`n" -ForegroundColor $Theme.Info

# 設置 TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 創建臨時目錄
$TmpDir = Join-Path $env:TEMP ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

# 清理函數
function Cleanup {
    if (Test-Path $TmpDir) {
        Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
    }
}

# 主安裝函數
function Install-CursorFreeVIP {
    Write-Styled "開始安裝 Cursor Free VIP" -Color $Theme.Primary -Prefix "安裝"
    
    # 設置安裝目錄
    $InstallDir = "$env:ProgramFiles\CursorFreeVIP"
    if (!(Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    
    try {
        # 獲取最新版本
        Write-Styled "正在檢查最新版本..." -Color $Theme.Primary -Prefix "更新"
        $releaseInfo = Get-LatestVersion
        $version = $releaseInfo.Version
        Write-Styled "找到最新版本: $version" -Color $Theme.Success -Prefix "版本"
        
        # 查找對應的資源
        $asset = $releaseInfo.Assets | Where-Object { $_.name -eq "CursorFreeVIP_${version}_windows.exe" }
        if (!$asset) {
            Write-Styled "找不到檔案: CursorFreeVIP_${version}_windows.exe" -Color $Theme.Error -Prefix "錯誤"
            Write-Styled "可用的檔案:" -Color $Theme.Warning -Prefix "資訊"
            $releaseInfo.Assets | ForEach-Object {
                Write-Styled "- $($_.name)" -Color $Theme.Info
            }
            throw "找不到對應的安裝檔案"
        }
        
        # 下載
        Write-Styled "正在下載..." -Color $Theme.Primary -Prefix "下載"
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "PowerShell Script")
        
        $downloadPath = Join-Path $TmpDir "CursorFreeVIP.exe"
        $webClient.DownloadFile($asset.browser_download_url, $downloadPath)
        
        # 安裝
        Write-Styled "正在安裝到系統..." -Color $Theme.Primary -Prefix "安裝"
        Copy-Item -Path $downloadPath -Destination "$InstallDir\CursorFreeVIP.exe" -Force
        
        # 添加到 PATH
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($currentPath -notlike "*$InstallDir*") {
            [Environment]::SetEnvironmentVariable("Path", "$currentPath;$InstallDir", "Machine")
        }
        
        Write-Styled "安裝完成！" -Color $Theme.Success -Prefix "完成"
        Write-Styled "正在啟動程序..." -Color $Theme.Primary -Prefix "啟動"
        
        # 運行程序
        Start-Process "$InstallDir\CursorFreeVIP.exe"
        
    }
    catch {
        Write-Styled $_.Exception.Message -Color $Theme.Error -Prefix "錯誤"
        throw
    }
}

# 執行安裝
try {
    Install-CursorFreeVIP
}
catch {
    Write-Styled "安裝失敗" -Color $Theme.Error -Prefix "錯誤"
    Write-Styled $_.Exception.Message -Color $Theme.Error
}
finally {
    Cleanup
    Write-Host "`n按任意鍵退出..." -ForegroundColor $Theme.Info
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}