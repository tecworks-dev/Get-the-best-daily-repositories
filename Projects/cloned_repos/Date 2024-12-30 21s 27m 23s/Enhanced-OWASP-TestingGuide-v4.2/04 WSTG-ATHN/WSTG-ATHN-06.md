# OWASP WSTG v4.0 - WSTG-ATHN-06

## Test Name: Testing for Browser Cache Weaknesses

### Overview
This test ensures that sensitive information, such as authentication credentials or session tokens, is not cached by the browser. Improper caching could allow unauthorized users to access sensitive data on shared or compromised systems.

---

### Objectives
- Verify that sensitive data is not stored in the browser cache.
- Assess the application’s use of caching directives to prevent unauthorized access.
- Identify scenarios where sensitive data can be exposed due to weak cache control policies.

---

### Test Steps

#### 1. **Inspect Cache-Control Headers**
   - **Scenario**: Ensure proper cache control headers are applied to sensitive pages and responses.
   - **Steps**:
     1. Use a browser’s developer tools or a proxy tool (e.g., Burp Suite) to capture HTTP responses.
     2. Check for cache-related headers, such as `Cache-Control`, `Pragma`, and `Expires`.
   - **Indicators**:
     - Sensitive pages or resources lack `Cache-Control: no-store, no-cache`.
     - Absence of `Pragma: no-cache` for older HTTP/1.0 clients.

#### 2. **Verify Browser Cache Behavior**
   - **Scenario**: Test how the browser caches sensitive data during user interactions.
   - **Steps**:
     1. Log in to the application and navigate through sensitive pages.
     2. Open the browser cache or inspect the storage via developer tools.
     3. Check if sensitive information, such as tokens or personal data, is cached.
   - **Indicators**:
     - Presence of cached sensitive data, such as HTML pages or JSON responses.

#### 3. **Test for History Storage of Sensitive Data**
   - **Scenario**: Check if sensitive data is stored in the browser history.
   - **Steps**:
     1. Navigate through sensitive pages, including login or account settings.
     2. Open the browser’s history and check for the URLs or page data.
   - **Indicators**:
     - URLs or page content exposing sensitive information in the browser history.

#### 4. **Analyze Autocomplete and Form Data**
   - **Scenario**: Ensure sensitive fields are not stored in browser autocomplete.
   - **Steps**:
     1. Test input fields, such as passwords or personal data, in forms.
     2. Observe whether autocomplete suggests previously entered values.
   - **Indicators**:
     - Autocomplete is enabled on sensitive fields (e.g., `autocomplete="on"`).

#### 5. **Assess File Downloads**
   - **Scenario**: Validate that temporary or downloaded files are not cached improperly.
   - **Steps**:
     1. Download files containing sensitive information from the application.
     2. Check the file storage path and behavior upon logout.
   - **Indicators**:
     - Sensitive files persist after logout or session expiration.

---

### Tools
- **Browser Developer Tools**: Inspect cache, storage, and network activity.
- **Burp Suite**: Capture and analyze HTTP responses.
- **Wireshark**: Monitor traffic for sensitive data exposure.
- **Postman**: Test API responses for cache directives.
- **Fiddler**: Debug HTTP caching behavior.

---

### Remediation
- Apply the `Cache-Control: no-store, no-cache` directive to sensitive resources.
- Use `Pragma: no-cache` for compatibility with older clients.
- Set `Expires: 0` to prevent browser caching.
- Disable autocomplete for sensitive input fields using `autocomplete="off"`.
- Ensure file downloads containing sensitive information are deleted or expire appropriately.
- Regularly test and audit cache policies for compliance.

---

### References
- [OWASP Testing Guide v4.0: Browser Cache Weaknesses](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)
- [OWASP Cheat Sheet: Sensitive Data Exposure](https://cheatsheetseries.owasp.org/cheatsheets/Sensitive_Data_Exposure_Cheat_Sheet.html)

---

### Checklist
- [ ] `Cache-Control` and `Pragma` headers are properly configured.
- [ ] Sensitive data is not stored in browser cache or history.
- [ ] Autocomplete is disabled for sensitive form fields.
- [ ] File downloads containing sensitive information are handled securely.
- [ ] Cache policies are regularly tested and reviewed.
