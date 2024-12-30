# WSTG-INPV-15: Testing for HTTP Header Injection

## Summary
HTTP Header Injection occurs when an attacker manipulates HTTP headers by injecting malicious content, leading to unauthorized actions such as HTTP response splitting, cache poisoning, or cross-site scripting (XSS).

---

## Objectives
- Identify endpoints that are vulnerable to HTTP header injection.
- Test for manipulation of HTTP headers and analyze the impact.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Analyze input fields, parameters, or headers that are reflected or processed in the HTTP headers.
   - Example headers to test:
     - `Location`
     - `Set-Cookie`
     - `User-Agent`

2. **Inject Malicious Payloads:**
   - Test for header injection by adding special characters or newline sequences to manipulate headers:
     - Example payloads:
       ```
       test
Set-Cookie: injected=value

       test
Location: https://malicious-site.com

       ```
   - Example request:
     ```http
     GET / HTTP/1.1
     Host: example.com
     User-Agent: test
Set-Cookie: injected=value
```     

3. **Observe Application Behavior:**
   - Check if the injected headers are reflected in the server response.
   - Analyze responses for signs of response splitting, cache poisoning, or other unintended effects.

4. **Test for Response Splitting:**
   - Inject payloads with 
 ```
    ` to split HTTP responses.
``` 


HTTP/1.1 200 OK
     ```
   - Verify if multiple responses are generated.

---

### Automated Testing

Use tools like:
- **Burp Suite:** Test HTTP headers for injection vulnerabilities.
- **OWASP ZAP:** Scan for header injection and response splitting issues.

---

## Mitigation
- Sanitize and validate all user input before including it in HTTP headers.
- Enforce a strict character set for header values.
- Use frameworks and libraries that automatically escape or sanitize HTTP headers.
- Implement security headers (e.g., `Content-Security-Policy`, `X-Frame-Options`) to reduce the impact of header injection attacks.

---

## References
- OWASP Testing Guide: HTTP Header Injection
- CWE-93: Improper Neutralization of CRLF Sequences in HTTP Headers ('HTTP Response Splitting')
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)
