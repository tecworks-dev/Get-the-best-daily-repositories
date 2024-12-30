# WSTG-INPV-16: Testing for Host Header Injection

## Summary
Host Header Injection occurs when an application processes an attacker-controlled `Host` header in an HTTP request without proper validation. This can lead to various attacks, including cache poisoning, password reset poisoning, and web cache deception.

---

## Objectives
- Identify vulnerabilities in the handling of the `Host` header.
- Assess the application's behavior when the `Host` header is manipulated.
- Determine potential impacts such as URL rewriting, phishing, or cache poisoning.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate endpoints where the `Host` header is used for redirections, URL generation, or internal logic.
   - Example request:
     ```http
     GET / HTTP/1.1
     Host: example.com
```     

2. **Inject Malicious Host Headers:**
   - Modify the `Host` header and observe the application's response.
     ```http
     GET / HTTP/1.1
     Host: attacker.com
```     
   - Test for scenarios like:
     - **Password Reset Poisoning:**
       ```http
       Host: attacker.com
```       
       Analyze the password reset email for malicious links.
     - **Cache Poisoning:**
       ```http
       Host: attacker.com
```       
       Verify if the cache stores the malicious `Host` value.

3. **Analyze Responses:**
   - Look for signs of URL rewriting or inclusion of the malicious `Host` in responses.
   - Example vulnerable response:
     ```html
     <a href="http://attacker.com/reset?token=abc123">Reset Password</a>
     ```

4. **Blind Testing:**
   - Use out-of-band techniques, such as DNS interactions, to confirm that the server processes the malicious `Host` header.
     ```http
     Host: attacker.com
```     

---

### Automated Testing

Use tools like:
- **Burp Suite:** Test and manipulate the `Host` header.
- **OWASP ZAP:** Scan for Host Header Injection vulnerabilities.
- **Custom Scripts:** Automate testing with specific payloads.

---

## Mitigation
- Validate and whitelist the `Host` header value.
- Use strict canonical hostnames for redirections and URL generation.
- Implement secure configurations for web servers and reverse proxies to prevent misuse of the `Host` header.
- Avoid relying on the `Host` header for critical functionality.

---

## References
- OWASP Host Header Injection Guide
- CWE-829: Inclusion of Functionality from Untrusted Control Sphere
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)