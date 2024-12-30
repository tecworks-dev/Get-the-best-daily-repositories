# WSTG-INPV-19: Testing for Path Traversal

## Summary
Path Traversal vulnerabilities occur when an attacker is able to access files and directories outside the intended scope by manipulating file path inputs. This can lead to unauthorized file access, information leakage, or even arbitrary code execution.

---

## Objectives
- Identify inputs that allow file path manipulation.
- Test for directory traversal vulnerabilities.
- Assess the potential impact of accessing restricted files or directories.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate endpoints or functionality that accept file paths as input, such as:
     - File download features.
     - Log viewers.
     - URL parameters specifying file names.

   - Example request:
     ```http
     GET /view?file=report.txt HTTP/1.1
     Host: example.com
```     

2. **Inject Path Traversal Payloads:**
   - Modify file path inputs with traversal sequences to access restricted files:
     ```
     ../../etc/passwd
     ..\..\windows\system32\config\sam
     ```

   - Example request with payload:
     ```http
     GET /view?file=../../etc/passwd HTTP/1.1
     Host: example.com
```     

3. **Analyze Responses:**
   - Look for:
     - Unauthorized file content in the response.
     - Error messages revealing restricted file paths or server configuration details.

4. **Blind Path Traversal Testing:**
   - Use timing attacks or specific error messages to infer the existence of files.
   - Example:
     ```http
     GET /view?file=../../nonexistentfile HTTP/1.1
     Host: example.com
```     

---

### Automated Testing

Use tools like:
- **Burp Suite:** Intercept and modify file path inputs.
- **OWASP ZAP:** Scan for directory traversal vulnerabilities.
- **Nikto:** Identify common path traversal issues.

#### Example with OWASP ZAP:
- Use the active scanner to detect directory traversal vulnerabilities.

---

## Mitigation
- Validate and sanitize all file path inputs to prevent traversal sequences.
- Use an allowlist to restrict file access to specific directories.
- Implement secure APIs or frameworks for file handling.
- Set proper file permissions to minimize exposure.
- Avoid exposing detailed error messages to users.

---

## References
- OWASP Path Traversal Cheat Sheet
- CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)
  - [Nikto](https://cirt.net/Nikto2)
