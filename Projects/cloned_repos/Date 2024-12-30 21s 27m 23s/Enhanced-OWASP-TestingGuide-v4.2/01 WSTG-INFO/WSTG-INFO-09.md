# WSTG-INFO-09: Fingerprint Web Application

## Objective
To identify the specific web application being used by analyzing its behavior, structure, and responses. This information can help pinpoint known vulnerabilities and understand its functionality.

## Key Steps

### 1. Inspect HTTP Headers
Analyze HTTP response headers for application-specific identifiers.
- Headers to look for:
  - `X-Powered-By`
  - `Server`
- Example:
  ```
  X-Powered-By: WordPress
  Server: Apache/2.4.41 (Ubuntu)
  ```

### 2. Review HTML Source Code
Search the HTML source code for application-specific identifiers.
- Look for:
  - Comments indicating the application.
  - Specific meta tags.
- Example:
  ```html
  <!-- This site is powered by Drupal -->
  <meta name="generator" content="WordPress 5.8">
  ```

### 3. Search for Default Directories and Pages
Locate application-specific default directories or pages.
- Examples:
  - `/wp-admin` or `/wp-login.php` (WordPress)
  - `/administrator` (Joomla)
  - `/user/login` (Drupal)

### 4. Use Automated Tools
Leverage tools to fingerprint the application automatically.
- Tools:
  - [WhatWeb](https://github.com/urbanadventurer/WhatWeb):
    ```bash
    whatweb http://targetdomain.com
    ```
  - [Wappalyzer](https://www.wappalyzer.com/)
  - [BuiltWith](https://builtwith.com/)

### 5. Inspect JavaScript Files
Analyze linked JavaScript files for application-specific identifiers.
- Example:
  - `wp-content` or `wp-includes` (WordPress)
  - `sites/default` (Drupal)

### 6. Analyze Cookies
Check cookies set by the application for specific naming patterns.
- Examples:
  - `wordpress_logged_in`
  - `drupal_session`
  - `joomla_user`

### 7. Review Application Behavior
Interact with the application to understand its behavior and identify specific features or patterns.
- Examples:
  - Login pages
  - Error messages
  - Redirects

### 8. Document Findings
Maintain a record of identified web applications with:
- Application name and version (if possible).
- Evidence supporting the identification.
- Known vulnerabilities or risks.

## Tools and Resources
- **Browser Tools**:
  - Developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
- **Tools**:
  - WhatWeb
  - Wappalyzer
  - BuiltWith
- **Online Services**:
  - [Wappalyzer](https://www.wappalyzer.com/)
  - [Netcraft](https://www.netcraft.com/)

## Mitigation Recommendations
- Remove or obfuscate application-specific identifiers in HTTP headers and HTML source code.
- Regularly update the web application to patch known vulnerabilities.
- Limit exposure of default directories and files.

---

**Next Steps:**
Proceed to [WSTG-INFO-10: Map Application Architecture](./WSTG_INFO_10.md).
