# WSTG-CLNT-10: Testing for Browser Security Headers

## Summary

Browser security headers enhance the security of web applications by instructing the browser how to handle certain types of content and interactions. Missing or improperly configured security headers can lead to vulnerabilities such as Cross-Site Scripting (XSS), clickjacking, or content sniffing attacks.

## Objective

To assess the presence and effectiveness of browser security headers and identify misconfigurations that could lead to potential security risks.

## How to Test

### Step 1: Identify Security Headers
1. Use tools like browser developer tools or interception proxies (e.g., Burp Suite, OWASP ZAP) to inspect HTTP responses.
2. Review the following headers:
   - **Content-Security-Policy (CSP)**: Prevents XSS and data injection attacks by restricting sources of content.
   - **X-Frame-Options**: Prevents clickjacking by restricting iframe embedding.
   - **Strict-Transport-Security (HSTS)**: Enforces HTTPS connections.
   - **X-Content-Type-Options**: Prevents MIME type sniffing.
   - **Referrer-Policy**: Controls how much referrer information is shared.
   - **Permissions-Policy**: Restricts access to browser features like geolocation, camera, etc.

---

### Step 2: Analyze Header Configuration
1. Check if the headers are present and properly configured:
   - **Content-Security-Policy**:
     - Verify the `default-src` directive and allowed sources for scripts, styles, images, etc.
     - Example: `Content-Security-Policy: default-src 'self';`
   - **X-Frame-Options**:
     - Ensure it is set to `DENY` or `SAMEORIGIN`.
     - Example: `X-Frame-Options: SAMEORIGIN`
   - **Strict-Transport-Security**:
     - Confirm it is configured with an appropriate `max-age` and includes `includeSubDomains`.
     - Example: `Strict-Transport-Security: max-age=31536000; includeSubDomains`
   - **X-Content-Type-Options**:
     - Ensure it is set to `nosniff`.
     - Example: `X-Content-Type-Options: nosniff`
   - **Referrer-Policy**:
     - Verify it restricts unnecessary referrer data.
     - Example: `Referrer-Policy: no-referrer`
   - **Permissions-Policy**:
     - Confirm unnecessary features are disabled.
     - Example: `Permissions-Policy: geolocation=(), camera=()`

2. Identify missing or weak configurations that could expose the application to attacks.

---

### Step 3: Perform Security Tests
1. **Test XSS Protection**:
   - Attempt to inject scripts and observe if the CSP prevents execution.
2. **Test Clickjacking Protection**:
   - Embed the application in an iframe and verify if `X-Frame-Options` or CSP blocks the embedding.
3. **Test HSTS Enforcement**:
   - Try accessing the application over HTTP and ensure it is redirected to HTTPS.
4. **Test MIME Sniffing Protection**:
   - Analyze responses for the `X-Content-Type-Options` header to ensure it prevents MIME type sniffing.

---

### Step 4: Analyze Results
1. Document missing or improperly configured headers.
2. Assess the potential risks, such as:
   - Exposure to XSS or clickjacking attacks.
   - Interception of unencrypted traffic.
   - Leakage of sensitive referrer information.

---

## Tools

- **Browser Developer Tools** for inspecting HTTP response headers.
- **Burp Suite** or **OWASP ZAP** for automated scanning and manual inspection.
- **Security Header Testing Tools**:
  - [SecurityHeaders.com](https://securityheaders.com)
  - [Mozilla Observatory](https://observatory.mozilla.org)

---

## Remediation

1. **Implement Missing Security Headers**:
   - Ensure all recommended headers are present and properly configured.

2. **Regularly Review Header Configurations**:
   - Periodically audit and update headers to align with security best practices.

3. **Use Default-Deny Policies**:
   - For CSP, use a `default-src 'none';` and explicitly allow trusted sources.

4. **Enforce HTTPS**:
   - Use HSTS to ensure all traffic is encrypted.

5. **Test After Deployment**:
   - Validate header effectiveness in live environments.

---

## References

- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)
- [Mozilla Developer Network - Security Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)

---
