@echo off
setlocal EnableDelayedExpansion

:: Paths to store IPs
set "ENV_PATH=.env"  
set "IP_LIST_FILE=device_ips.txt"

:: Check if ADB is installed
adb version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ADB is not installed. Please install ADB first.
    exit /b 1
)

echo 🔌 Disconnecting old ADB connections...
adb disconnect

echo 📡 Setting ADB to TCP/IP mode on port 5555...
adb tcpip 5555

echo ⏳ Waiting for device to initialize...
timeout /t 3 /nobreak >nul

:: Get the device's IP address
for /f "tokens=2 delims= " %%A in ('adb shell ip addr show wlan0 ^| findstr /R "inet "') do set IP_FULL=%%A
for /f "delims=/" %%A in ("!IP_FULL!") do set IP=%%A

adb kill-server
adb start-server

:: Function to check and connect to ADB
:connect_to_adb
set "ip=%~1"
echo 🔄 Checking connectivity for %ip%...

ping -n 1 -w 1000 %ip% >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ %ip% is reachable, attempting ADB connection...
    adb connect %ip%:5555 >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Successfully connected to %ip%!
        
        findstr /x /c:"%ip%" "%IP_LIST_FILE%" >nul 2>&1 || (
            echo %ip%>>"%IP_LIST_FILE%"
            echo 📂 New IP saved in %IP_LIST_FILE%
        )
        exit /b 0
    ) else (
        echo ❌ ADB connection to %ip% failed.
    )
) else (
    echo ⚠️ %ip% is unreachable, skipping...
)
exit /b 1

:: **Prioritized IP Checking**
if not "%IP%"=="" call :connect_to_adb %IP% && exit /b 0

if exist "%ENV_PATH%" (
    echo ✅ Loading environment variables from %ENV_PATH%
    for /f "delims=" %%A in (%ENV_PATH%) do set %%A
    if not "%DEVICE_IP%"=="" call :connect_to_adb %DEVICE_IP% && exit /b 0
) else (
    echo ⚠️ .env file not found.
)

if exist "%IP_LIST_FILE%" (
    echo 🔍 Checking stored IPs from %IP_LIST_FILE%...
    for /f %%A in (%IP_LIST_FILE%) do call :connect_to_adb %%A && exit /b 0
)

echo ❌ No devices connected. Please check your network or try again.
exit /b 1
