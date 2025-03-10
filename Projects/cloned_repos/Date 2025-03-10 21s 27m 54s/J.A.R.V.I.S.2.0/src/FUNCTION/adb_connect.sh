#!/bin/bash

# Paths to store IPs
ENV_PATH=".env"               # Path to the .env file
IP_LIST_FILE="device_ips.txt" # File to store connected device IPs

# Check if ADB is installed
if ! command -v adb version &>/dev/null; then
    echo "❌ ADB is not installed. Please install ADB first."
    exit 1
fi


echo "🔌 Disconnecting old ADB connections..."
adb disconnect

echo "📡 Setting ADB to TCP/IP mode on port 5555..."
adb tcpip 5555

echo "⏳ Waiting for device to initialize..."
sleep 3

# Get the device's IP address (only works via USB)
IP_FULL=$(adb shell ip addr show wlan0 | grep "inet " | awk '{print $2}')
IP=${IP_FULL%%/*}  # Extract only the IP part before the slash

adb kill-server
adb start-server

# Function to check and connect to ADB
connect_to_adb() {
    local ip="$1"

    echo "🔄 Checking connectivity for $ip..."
    
    if ping -c 1 -W 1 "$ip" &>/dev/null; then
        echo "✅ $ip is reachable, attempting ADB connection..."
        
        if adb connect "$ip:5555" &>/dev/null; then
            echo "✅ Successfully connected to $ip!"

            # Store new working IP if not already saved
            if ! grep -Fxq "$ip" "$IP_LIST_FILE"; then
                echo "$ip" >> "$IP_LIST_FILE"
                echo "📂 New IP saved in $IP_LIST_FILE"
            fi
            
            return 0  # Stop further execution
        else
            echo "❌ ADB connection to $ip failed."
        fi
    else
        echo "⚠️ $ip is unreachable, skipping..."
    fi

    return 1  # Continue checking other IPs
}

# **Prioritized IP Checking**
if [[ -n "$IP" ]] && connect_to_adb "$IP"; then
    exit 0  # Stop on first success
    
elif [[ -f "$ENV_PATH" ]]; then
    echo "✅ Loading environment variables from $ENV_PATH"
    set -a
    source "$ENV_PATH"
    set +a
    if [[ -n "$DEVICE_IP" ]] && connect_to_adb "$DEVICE_IP"; then
        exit 0
    fi
else
    echo "⚠️ .env file not found."
fi

if [[ -f "$IP_LIST_FILE" ]]; then  
    echo "🔍 Checking stored IPs from $IP_LIST_FILE..."
    while read -r stored_ip; do
        if connect_to_adb "$stored_ip"; then
            exit 0  # Stop on first success
        fi
    done < "$IP_LIST_FILE"
fi

echo "❌ No devices connected. Please check your network or try again."
exit 1


#call : adb shell am start -a android.intent.action.CALL -d tel:+919876543210
# adb shell am start -n com.android.chrome/com.google.android.apps.chrome.Main
# screen record 
# adb shell screenrecord /sdcard/video.mp4
# adb pull /sdcard/video.mp4 .
#screen shot 
#adb shell screencap -p /sdcard/screen.png
#adb pull /sdcard/screen.png