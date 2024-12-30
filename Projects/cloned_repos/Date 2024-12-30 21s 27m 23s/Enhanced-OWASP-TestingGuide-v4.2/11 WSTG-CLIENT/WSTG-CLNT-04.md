# WSTG-CLNT-04: Testing for Cross-Origin Resource Sharing (CORS)

## Summary

Cross-Origin Resource Sharing (CORS) is a browser security feature that controls how resources on a web page are requested from another domain. Improperly configured CORS policies can expose sensitive data to unauthorized origins, leading to potential security risks such as data theft or malicious manipulation.

## Objective

To identify misconfigurations in the CORS policy that could allow unauthorized origins to access restricted resources or sensitive information.

## How to Test

### Step 1: Identify CORS Implementation
1. Review how the application implements CORS by inspecting HTTP headers in responses, such as:
   - `Access-Control-Allow-Origin`
   - `Access-Control-Allow-Credentials`
   - `Access-Control-Allow-Methods`
   - `Access-Control-Allow-Headers`

2. Analyze scenarios where the application handles cross-origin requests, such as:
   - API endpoints
   - Static resources (e.g., JavaScript or CSS files)
   - AJAX requests

---

### Step 2: Test for Misconfigured Headers
1. **Allowing Any Origin (`*`)**:
   - Verify if the `Access-Control-Allow-Origin` header is set to `*`.
   - This allows any origin to access the resources, posing a security risk.

2. **Dynamic Origin Reflection**:
   - Check if the server reflects the `Origin` header value dynamically.
   - Example: The server sets `Access-Control-Allow-Origin` to the value of the request’s `Origin` header.

3. **Credentials Misconfiguration**:
   - Validate if `Access-Control-Allow-Credentials: true` is set with an overly permissive `Access-Control-Allow-Origin`.
   - Example: Allowing credentials with `*` or reflected origins is insecure.

---

### Step 3: Exploit CORS Misconfigurations
1. **Send Malicious Cross-Origin Requests**:
   - Craft requests from an unauthorized origin to test access to sensitive resources.
   - Example: Use a custom `Origin` header to simulate an attacker’s website.

2. **Check for Sensitive Data Exposure**:
   - Analyze the responses to determine if sensitive data (e.g., tokens, user information) is exposed to unauthorized origins.

3. **Test Preflight Requests**:
   - Simulate `OPTIONS` preflight requests with custom headers or methods.
   - Example:
     ```http
     OPTIONS /api/resource HTTP/1.1
     Origin: https://malicious.com
     Access-Control-Request-Method: GET
     Access-Control-Request-Headers: Authorization
```     

---

### Step 4: Validate Exploitation
1. Confirm if the misconfiguration allows:
   - Unauthorized origins to access restricted data.
   - Exfiltration of sensitive information through malicious cross-origin requests.

2. Assess the impact on:
   - Confidentiality (e.g., exposure of sensitive resources)
   - Integrity (e.g., unauthorized modifications)

---

## Tools

- **Browser Developer Tools** for inspecting CORS headers and analyzing network requests
- **Burp Suite** or **OWASP ZAP** for crafting and testing cross-origin requests
- **CURL** or custom scripts to simulate malicious origins
- **Online Tools** like CORS testing utilities (e.g., Test CORS or CORS Exploiters)

---

## Remediation

1. **Restrict Allowed Origins**:
   - Use a strict allowlist of trusted origins for `Access-Control-Allow-Origin`.

2. **Avoid Reflecting Origin Headers**:
   - Do not dynamically set `Access-Control-Allow-Origin` to the value of the `Origin` header.

3. **Disable Credentials for Untrusted Origins**:
   - Ensure `Access-Control-Allow-Credentials` is not enabled unless absolutely necessary, and only for trusted origins.

4. **Implement Robust Preflight Validation**:
   - Validate preflight requests to ensure they comply with the expected headers and methods.

5. **Audit and Test Regularly**:
   - Regularly audit CORS configurations to ensure compliance with security best practices.

---

## References

- [OWASP Testing Guide - Testing for CORS Misconfigurations](https://owasp.org/www-project-testing/)
- [MDN Web Docs - HTTP Access Control (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)

---
