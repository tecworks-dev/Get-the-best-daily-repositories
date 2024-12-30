# WSTG-INFO-04: Enumerate Applications on Webserver

## Objective
To identify all applications hosted on the target web server and gather information about their versions, technologies, and configurations to uncover potential vulnerabilities.

## Key Steps

### 1. Enumerate Subdomains
Discover subdomains that may host additional applications.
- Tools:
  - [Sublist3r](https://github.com/aboul3la/Sublist3r):
    ```bash
    sublist3r -d targetdomain.com
    ```
  - [Amass](https://github.com/OWASP/Amass):
    ```bash
    amass enum -d targetdomain.com
    ```

### 2. Scan for Virtual Hosts
Identify virtual hosts hosted on the same server.
- Use `vhost` enumeration tools:
  - [FFUF](https://github.com/ffuf/ffuf):
    ```bash
    ffuf -u http://targetdomain.com -H "Host: FUZZ.targetdomain.com" -w wordlist.txt
    ```
  - [VHostScan](https://github.com/codingo/VHostScan)

### 3. Enumerate Web Directories and Files
Find directories and files that may lead to applications or additional functionality.
- Tools:
  - [Gobuster](https://github.com/OJ/gobuster):
    ```bash
    gobuster dir -u http://targetdomain.com -w wordlist.txt
    ```
  - [Dirsearch](https://github.com/maurosoria/dirsearch):
    ```bash
    python3 dirsearch.py -u http://targetdomain.com -e php,asp,html
    ```

### 4. Identify Installed Applications
Detect all hosted applications and frameworks.
- Tools:
  - [WhatWeb](https://github.com/urbanadventurer/WhatWeb):
    ```bash
    whatweb http://targetdomain.com
    ```
  - [Wappalyzer](https://www.wappalyzer.com/)

### 5. Search for Known Endpoints
Manually or programmatically search for endpoints that may lead to other applications.
- Common endpoints to check:
  - `/admin`
  - `/test`
  - `/backup`
  - `/staging`

### 6. Analyze Server Responses
Inspect HTTP response codes and headers for clues about hosted applications.
- Examples:
  - `200 OK`: Application is accessible
  - `403 Forbidden`: Application exists but requires permissions

### 7. Check for Multi-Tenancy
Determine if the server hosts multiple tenants or services.
- Analyze patterns in URLs and subdomains for indicators of multi-tenancy.

### 8. Document Findings
Maintain a detailed log of discovered applications:
- Application names
- Versions
- Locations (URLs, directories, subdomains)
- Associated technologies

## Tools and Resources
- **Tools**:
  - Sublist3r
  - Amass
  - Gobuster
  - Dirsearch
  - WhatWeb
  - Wappalyzer
- **Browser Extensions**:
  - BuiltWith
  - Wappalyzer

## Mitigation Recommendations
- Limit exposure of unnecessary or unused applications.
- Use virtual host configurations to restrict access to specific domains or subdomains.
- Regularly audit web server directories and applications.
- Implement proper access controls for sensitive applications.

---

**Next Steps:**
Proceed to [WSTG-INFO-05: Review Webpage Content for Information Leakage](./WSTG_INFO_05.md).
