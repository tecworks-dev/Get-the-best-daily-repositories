# WSTG-CONF-08: Test RIA Cross Domain Policy

## Objective
Ensure that Rich Internet Applications (RIA) such as Flash or Silverlight securely implement cross-domain policies to prevent unauthorized access to sensitive data.

## Checklist

### 1. Locate Cross-Domain Policy Files
- Identify the location of cross-domain policy files:
  - Flash: `crossdomain.xml`
  - Silverlight: `clientaccesspolicy.xml`

### 2. Analyze Cross-Domain Policy File Contents
- Verify that the files are present only if required.
- Review the `<allow-access-from>` directives:
  - Ensure `domain="*"` is not used to allow unrestricted access.
  - Restrict access to trusted domains only.

### 3. Validate HTTPS Usage
- Confirm that cross-domain policies are only served over HTTPS to prevent interception.
- Check the `secure="true"` attribute in policy files to enforce secure connections.

### 4. Test Unauthorized Data Access
- Attempt to access sensitive data or APIs from untrusted domains:
  - Confirm that unauthorized domains are blocked.

### 5. Check for Misconfigured Headers
- Verify that no wildcard (`*`) values are used in HTTP headers like `Access-Control-Allow-Origin` when serving RIA files.

### 6. Assess the Necessity of RIA Files
- Determine if the RIA functionality is still required. If not, remove associated files and configurations.

### 7. Inspect File Permissions
- Ensure that cross-domain policy files are not writable by unauthorized users.

## Tools
- **Burp Suite**: For intercepting and testing RIA-related requests.
- **cURL**: To fetch and analyze cross-domain policy files.
  - Example: `curl https://example.com/crossdomain.xml`
- **OWASP ZAP**: For testing cross-domain policies and configurations.

---

Let me know if you’d like to refine this section or move to the next file!
