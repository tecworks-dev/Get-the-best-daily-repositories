# WSTG-CLNT-11: Testing for Cross-Origin Resource Sharing (CORS) Misconfigurations

## Summary

Cross-Origin Resource Sharing (CORS) is a mechanism that allows restricted resources on a web page to be requested from another domain. Misconfigured CORS policies can lead to unauthorized access to sensitive data or unintended actions performed on behalf of an attacker.

## Objective

To identify and exploit weaknesses in CORS configurations that allow unauthorized cross-origin requests, leading to data leakage, privilege escalation, or other security breaches.

## How to Test

### Step 1: Identify CORS Headers
1. Inspect HTTP responses for CORS-related headers:
   - `Access-Control-Allow-Origin`
   - `Access-Control-Allow-Methods`
   - `Access-Control-Allow-Headers`
   - `Access-Control-Allow-Credentials`

2. Determine if the headers are overly permissive, such as:
   - `Access-Control-Allow-Origin: *`
   - `Access-Control-Allow-Origin` reflecting user-provided values.
   - Misuse of `Access-Control-Allow-Credentials: true`.

---

### Step 2: Analyze CORS Policy
1. Check for common misconfigurations:
   - Wildcard (`*`) in `Access-Control-Allow-Origin` with sensitive data.
   - Dynamic origin reflection without validation.
   - Allowing credentials with overly permissive origins.
   - Misconfigured preflight responses.

2. Focus on endpoints that process sensitive data or perform critical operations.

---

### Step 3: Test Exploit Scenarios
1. **Test Cross-Origin Requests**:
   - Use tools like `curl`, Postman, or custom scripts to send cross-origin requests with a malicious `Origin` header:
     ```bash
     curl -H "Origin: https://attacker.com" -H "Access-Control-Request-Method: GET" -X OPTIONS https://target.com/api
     ```

2. **Simulate Exploitation**:
   - Create a malicious HTML page to perform unauthorized requests:
     ```html
     <script>
       fetch("https://target.com/api", {
         method: "GET",
         credentials: "include"
       }).then(response => response.text()).then(console.log);
     </script>
     ```

3. **Test Preflight Behavior**:
   - Send preflight `OPTIONS` requests with custom headers or methods to verify the server's behavior.

---

### Step 4: Analyze Results
1. Confirm if the application allows unauthorized access from untrusted origins.
2. Validate if sensitive data, such as cookies, tokens, or user information, can be accessed or manipulated by an attacker.

---

## Tools

- **Browser Developer Tools** for inspecting CORS headers and network requests.
- **Burp Suite** or **OWASP ZAP** for automated CORS testing.
- **CURL** or **Postman** for crafting and testing custom cross-origin requests.
- **CORS Testing Tools**:
  - [CORS Tester](https://github.com/s0md3v/Corsy)
  - [Online CORS Testers](https://www.test-cors.org)

---

## Remediation

1. **Restrict Allowed Origins**:
   - Avoid using `*` for `Access-Control-Allow-Origin`.
   - Specify a strict allowlist of trusted origins.

2. **Avoid Dynamic Reflection**:
   - Do not reflect the `Origin` header value dynamically without validation.

3. **Disallow Credentials with Wildcard Origins**:
   - Ensure `Access-Control-Allow-Credentials: true` is only used with specific origins, not `*`.

4. **Validate Preflight Requests**:
   - Properly validate preflight `OPTIONS` requests to allow only expected methods and headers.

5. **Audit CORS Configuration**:
   - Regularly review and test CORS policies to ensure they comply with security best practices.

---

## References

- [OWASP Testing Guide - Testing for CORS](https://owasp.org/www-project-testing/)
- [Mozilla Developer Network - CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [OWASP Secure Headers Project](https://owasp.org/www-project-secure-headers/)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)

---
