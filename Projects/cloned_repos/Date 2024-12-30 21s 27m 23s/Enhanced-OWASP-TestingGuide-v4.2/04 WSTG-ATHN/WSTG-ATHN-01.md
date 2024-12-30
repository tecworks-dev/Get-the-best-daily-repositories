# OWASP WSTG v4.0 - WSTG-ATHN-01

## Test Name: Testing for Credentials Transported Over an Encrypted Channel

### Overview
This test ensures that sensitive credentials, such as usernames and passwords, are transmitted securely over encrypted channels to prevent interception by attackers.

---

### Objectives
- Verify that the application uses secure protocols like HTTPS for transmitting credentials.
- Detect any instances where credentials are exposed in plaintext during transmission.
- Assess the effectiveness of encryption mechanisms.

---

### Test Steps

#### 1. **Verify the Use of HTTPS**
   - **Scenario**: Check if the application enforces HTTPS for sensitive operations like login, registration, or password recovery.
   - **Steps**:
     1. Access the login or registration page.
     2. Inspect the URL to confirm HTTPS usage.
     3. Observe browser indicators for valid SSL/TLS certificates.
   - **Indicators**:
     - Secure lock icon in the browser.
     - Valid and up-to-date SSL/TLS certificate.
     - No HTTP fallback for sensitive pages.

#### 2. **Inspect Credential Transmission**
   - **Scenario**: Analyze network traffic to confirm that credentials are transmitted securely.
   - **Steps**:
     1. Use tools like Burp Suite or Wireshark to intercept traffic during login.
     2. Observe if credentials are encrypted in transit.
     3. Check for sensitive data leaks in headers, parameters, or body.
   - **Indicators**:
     - Encrypted credentials in transmission.
     - No sensitive data exposed in plaintext.

#### 3. **Test for HTTP to HTTPS Redirection**
   - **Scenario**: Ensure HTTP requests are automatically redirected to HTTPS.
   - **Steps**:
     1. Attempt to access the application using HTTP.
     2. Observe if the application redirects to HTTPS.
   - **Indicators**:
     - Automatic redirection to HTTPS.
     - No sensitive operations accessible over HTTP.

#### 4. **Check SSL/TLS Configuration**
   - **Scenario**: Validate the strength and configuration of SSL/TLS.
   - **Steps**:
     1. Use tools like SSL Labs or OpenSSL to assess the server’s SSL/TLS configuration.
     2. Look for weak cipher suites, outdated protocols, or missing security headers.
   - **Indicators**:
     - Strong and modern cipher suites.
     - Absence of deprecated protocols like SSLv2, SSLv3, or TLS 1.0.
     - Presence of security headers like `Strict-Transport-Security`.

#### 5. **Analyze API Endpoints**
   - **Scenario**: Confirm that APIs transmitting credentials also use encryption.
   - **Steps**:
     1. Test API login or registration endpoints.
     2. Inspect requests for secure protocols and encrypted data.
   - **Indicators**:
     - API endpoints accessible only via HTTPS.
     - No credentials exposed in plaintext.

---

### Tools
- **Burp Suite**: Intercept and analyze traffic.
- **Wireshark**: Capture and inspect network packets.
- **SSL Labs**: Assess SSL/TLS configuration.
- **Postman**: Test API endpoints.
- **OpenSSL**: Analyze certificates and configurations.

---

### Remediation
- Enforce HTTPS for all sensitive operations and pages.
- Redirect all HTTP requests to HTTPS automatically.
- Use strong SSL/TLS configurations and avoid deprecated protocols.
- Regularly audit certificates to ensure they are valid and up to date.
- Secure API endpoints with encryption and ensure no sensitive data is exposed.

---

### References
- [OWASP Testing Guide v4.0: Credentials Transported Over an Encrypted Channel](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)

---

### Checklist
- [ ] HTTPS is enforced for all sensitive pages and operations.
- [ ] SSL/TLS certificates are valid and up to date.
- [ ] Credentials are encrypted in transit.
- [ ] HTTP requests are redirected to HTTPS.
- [ ] API endpoints use encryption for credential transmission.
