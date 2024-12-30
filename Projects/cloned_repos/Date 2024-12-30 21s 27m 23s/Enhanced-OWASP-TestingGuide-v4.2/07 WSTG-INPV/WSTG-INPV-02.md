# WSTG-INPV-02 - Testing for Stored Cross-Site Scripting (XSS)

## Summary
Stored Cross-Site Scripting (XSS) occurs when an attacker injects malicious scripts into a web application that are then stored and later executed in the browser of users who access the stored data. This type of XSS is more severe because it can affect multiple users.

## Objective
To identify stored XSS vulnerabilities by injecting malicious scripts into fields where input is stored and later retrieved.

## Testing Procedure

### 1. Identify Input Points
- **Description**: Locate fields where user input is stored and later displayed (e.g., comments, profiles, feedback forms).
- **Steps**:
  1. Navigate the application and identify inputs linked to stored data.
  2. Verify if data from these fields is reflected back to the user or other users.

### 2. Inject Payloads
- **Description**: Test inputs with payloads to identify stored XSS vulnerabilities.
- **Steps**:
  1. Use XSS payloads such as `<script>alert('Stored XSS')</script>` or `<img src=x onerror=alert('Stored XSS')>`.
  2. Submit these payloads into input fields that store data.
  3. Check if the payload is executed when the stored data is retrieved or displayed.

### 3. Test Various Contexts
- **Description**: Verify XSS execution in different contexts (e.g., HTML, attributes, JavaScript).
- **Steps**:
  1. Inject context-specific payloads (e.g., for attributes: `" onmouseover=alert('XSS')` or for JavaScript: `'); alert('XSS');//`).
  2. Confirm if the payload is executed when stored data is rendered.

### 4. Verify Impact Across Users
- **Description**: Check if the stored payload impacts multiple users.
- **Steps**:
  1. Log in with a different user account.
  2. Access pages displaying the stored data.
  3. Confirm if the malicious payload executes for different users.

### 5. Analyze Stored Data
- **Description**: Inspect the backend storage of user inputs.
- **Steps**:
  1. Submit payloads and analyze database responses (if access is available).
  2. Confirm if the data is stored unencoded or unsanitized.

### 6. Test for Bypasses
- **Description**: Attempt to bypass filters by encoding payloads or using alternate syntax.
- **Steps**:
  1. Encode payloads using URL encoding, Base64, or HTML entities.
  2. Submit encoded payloads into input fields.
  3. Observe if the application decodes and executes the payload.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools
- XSS Payload Generators

## Remediation
1. Validate and sanitize all user inputs before storage.
2. Encode outputs based on the context (e.g., HTML, JavaScript, attributes).
3. Implement Content Security Policy (CSP) to restrict script execution.
4. Use frameworks or libraries that automatically handle output encoding.
5. Regularly review and sanitize stored data.
6. Educate developers on secure coding practices to prevent XSS.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
