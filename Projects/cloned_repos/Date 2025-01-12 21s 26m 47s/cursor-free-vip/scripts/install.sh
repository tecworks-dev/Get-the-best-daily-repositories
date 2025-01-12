#!/bin/bash

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logo
print_logo() {
    echo -e "${CYAN}"
    cat << "EOF"
   ██████╗██╗   ██╗██████╗ ███████╗ ██████╗ ██████╗      ██████╗ ██████╗  ██████╗   
  ██╔════╝██║   ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗     ██╔══██╗██╔══██╗██╔═══██╗  
  ██║     ██║   ██║██████╔╝███████╗██║   ██║██████╔╝     ██████╔╝██████╔╝██║   ██║  
  ██║     ██║   ██║██╔══██╗╚════██║██║   ██║██╔══██╗     ██╔═══╝ ██╔══██╗██║   ██║  
  ╚██████╗╚██████╔╝██║  ██║███████║╚██████╔╝██║  ██║     ██║     ██║  ██║╚██████╔╝  
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝  
EOF
    echo -e "${NC}"
}

# 檢查是否為 root 用戶
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}❌ 錯誤: 請使用 sudo 運行此腳本${NC}"
        exit 1
    fi
}

# 獲取最新版本
get_latest_version() {
    echo -e "${CYAN}ℹ️ 正在檢查最新版本...${NC}"
    local latest_release
    latest_release=$(curl -s https://api.github.com/repos/yeongpin/cursor-free-vip/releases/latest)
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 無法獲取最新版本信息${NC}"
        exit 1
    fi
    
    VERSION=$(echo "$latest_release" | grep -o '"tag_name": ".*"' | cut -d'"' -f4 | tr -d 'v')
    echo -e "${GREEN}✅ 找到最新版本: ${VERSION}${NC}"
}

# 檢測系統類型
detect_os() {
    if [[ "$(uname)" == "Darwin" ]]; then
        OS="mac"
    else
        OS="linux"
    fi
}

# 創建臨時目錄
create_temp_dir() {
    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' EXIT
}

# 下載並安裝
install_cursor_free_vip() {
    local install_dir="/usr/local/bin"
    local binary_name="CursorFreeVIP_${VERSION}_${OS}"
    local download_url="https://github.com/yeongpin/cursor-free-vip/releases/download/v${VERSION}/${binary_name}"
    
    echo -e "${CYAN}ℹ️ 正在下載...${NC}"
    if ! curl -L -o "${TMP_DIR}/${binary_name}" "$download_url"; then
        echo -e "${RED}❌ 下載失敗${NC}"
        exit 1
    fi
    
    echo -e "${CYAN}ℹ️ 正在安裝...${NC}"
    chmod +x "${TMP_DIR}/${binary_name}"
    mv "${TMP_DIR}/${binary_name}" "${install_dir}/cursor-free-vip"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 安裝完成！${NC}"
        echo -e "${CYAN}ℹ️ 您可以通過運行 'cursor-free-vip' 來啟動程序${NC}"
    else
        echo -e "${RED}❌ 安裝失敗${NC}"
        exit 1
    fi
}

# 主程序
main() {
    print_logo
    check_root
    get_latest_version
    detect_os
    create_temp_dir
    install_cursor_free_vip
}

# 運行主程序
main 