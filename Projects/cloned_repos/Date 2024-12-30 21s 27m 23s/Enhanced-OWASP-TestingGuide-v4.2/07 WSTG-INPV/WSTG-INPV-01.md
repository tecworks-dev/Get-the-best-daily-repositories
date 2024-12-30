# WSTG-INPV-01 - Testing for Reflected Cross-Site Scripting (XSS)

## Summary
Reflected Cross-Site Scripting (XSS) occurs when an application takes untrusted input, processes it, and includes it in the immediate response without proper validation or encoding. Attackers can exploit this vulnerability to execute malicious scripts in the context of the user's browser.

## Objective
To identify and validate the presence of reflected XSS vulnerabilities and assess their impact.

## Testing Procedure

### 1. Identify Input Points
- **Description**: Locate areas in the application where user input is accepted.
- **Steps**:
  1. Use the application to identify forms, query parameters, search fields, or other input points.
  2. Note all inputs that are reflected in the server's response.

### 2. Inject Payloads
- **Description**: Test inputs with payloads to identify reflected XSS vulnerabilities.
- **Steps**:
  1. Use simple XSS payloads like `<script>alert('XSS')</script>` or `<img src=x onerror=alert('XSS')>`.
  2. Inject these payloads into input fields or query parameters.
  3. Observe the server's response to check if the payload is executed.

### 3. Test Special Characters
- **Description**: Check how the application handles special characters in inputs.
- **Steps**:
  1. Input characters like `<`, `>`, `"`, `'`, `&`, and `;`.
  2. Analyze the response to identify improper encoding or sanitization.

### 4. Test for Context-Specific XSS
- **Description**: Test for XSS in different contexts like HTML, JavaScript, attributes, or CSS.
- **Steps**:
  1. Inject context-specific payloads (e.g., for attributes: `" onmouseover=alert('XSS')` or for JavaScript: `'); alert('XSS');//`).
  2. Observe if the payload executes in the respective context.

### 5. Check HTTP Responses
- **Description**: Analyze HTTP responses to identify reflected input.
- **Steps**:
  1. Capture responses using tools like Burp Suite or OWASP ZAP.
  2. Look for reflections of injected input in the response.
  3. Confirm if the input is properly sanitized or encoded.

### 6. Test for Encoding and Bypasses
- **Description**: Attempt to bypass filters by encoding payloads.
- **Steps**:
  1. Encode payloads using URL encoding, HTML entities, or Base64.
  2. Inject encoded payloads into inputs.
  3. Check if the application decodes the payload and executes it.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools
- XSS Payload Generators (e.g., XSSer)

## Remediation
1. Validate and sanitize all user inputs on the server side.
2. Encode output based on the context (e.g., HTML, JavaScript).
3. Implement Content Security Policy (CSP) to mitigate XSS.
4. Avoid reflecting untrusted input in server responses.
5. Educate developers on secure coding practices to prevent XSS.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
