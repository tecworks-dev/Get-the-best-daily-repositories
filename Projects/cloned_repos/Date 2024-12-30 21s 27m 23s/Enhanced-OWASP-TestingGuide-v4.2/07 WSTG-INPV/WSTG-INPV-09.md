# WSTG-INPV-09: Testing for HTTP Parameter Pollution (HPP)

## Summary
HTTP Parameter Pollution (HPP) occurs when an attacker manipulates or injects multiple HTTP parameters with the same name to exploit vulnerable applications that do not handle such cases properly. This vulnerability may lead to unexpected behaviors, such as bypassing authentication or causing other unintended effects.

---

## Objectives
- Identify and test for applications that are vulnerable to HPP.
- Validate if duplicated parameters are handled safely.
- Explore potential abuse scenarios, such as bypassing security mechanisms or injecting malicious payloads.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Map the application for all HTTP parameters. Focus on forms, query strings, and headers.
   - Example:

```http
     GET /search?q=test&filter=1 HTTP/1.1 

```



1. **Inject Duplicate Parameters:**
   - Modify requests to include duplicate parameters with different values.
```

     GET /search?q=test&q=malicious HTTP/1.1
 ``` 
 
   - Observe how the server processes the request. For example:
     - Does it accept the first or the last parameter?
     - Does it concatenate values?
     - Does it reject the request?

3. **Test with Special Characters:**
   - Inject variations of parameters with special characters.
     ```http
     GET /search?q=test&q[]=injection&q=<script>alert(1)</script> HTTP/1.1
```



4. **Analyze the Response:**
   - Determine if the server behaves unexpectedly or processes values insecurely.

---

### Automated Testing

Use tools like:
- Burp Suite (intruder with custom payloads)
- OWASP ZAP
- Fuzzers (e.g., wfuzz)

#### Example with Burp Suite:
- Intercept the request.
- Add duplicate parameters in the repeater.
- Observe the server's response to determine if any vulnerabilities exist.

---

## Mitigation
- Implement strict validation and sanitization for all parameters.
- Configure the application to handle duplicate parameters deterministically (e.g., accept only the first occurrence).
- Use security libraries and frameworks that automatically mitigate such issues.
- Implement Content Security Policy (CSP) to reduce the impact of HPP when combined with XSS.

---

## References
- OWASP Testing Guide
- CWE-235: Improper Handling of Parameters
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)
