# WSTG-CONF-02: Test Application Platform Configuration

## Objective
Ensure that the application platform (e.g., web server, application server) is configured securely to prevent unauthorized access, data leakage, or other vulnerabilities.

## Checklist

### 1. Verify Default Configuration
- Check if the application platform is running with default settings, which may expose unnecessary services or credentials.
- Look for default usernames/passwords in use (e.g., admin/admin).

### 2. Check for Unnecessary Services
- Identify services running on the platform.
- Disable or remove services that are not required for the application to function.

### 3. Validate Patch Management
- Ensure the platform is up-to-date with security patches and updates.
- Check the version of the platform and compare it with the latest stable release.

### 4. Analyze Configuration Files
- Review configuration files for:
  - Hardcoded credentials.
  - Sensitive information (e.g., API keys, tokens).
  - Improper permissions or access controls.

### 5. Verify HTTPS Implementation
- Ensure HTTPS is enabled and properly configured.
- Check for valid SSL/TLS certificates.

### 6. Analyze Directory Listings
- Confirm that directory listing is disabled on the platform to prevent unauthorized access to files.

### 7. Review Error Messages
- Validate that error messages do not disclose sensitive platform or application information.

### 8. Test for Misconfigured Headers
- Check HTTP headers for:
  - Missing or incorrect security headers (e.g., Content-Security-Policy, X-Frame-Options).
  - Leaked internal server information.

## Tools
- **Nmap**: For port and service enumeration.
- **Nikto**: For configuration weaknesses.
- **Burp Suite**: For analyzing headers and configurations.
- **SSL Labs**: For HTTPS testing.

---

Feel free to refine this file as needed or let me know if you'd like to proceed to the next! 
