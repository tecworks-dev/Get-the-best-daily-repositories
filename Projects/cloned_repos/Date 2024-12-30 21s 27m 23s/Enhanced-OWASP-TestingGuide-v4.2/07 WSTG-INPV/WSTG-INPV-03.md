# WSTG-INPV-03 - Testing for DOM-Based Cross-Site Scripting (XSS)

## Summary
DOM-Based Cross-Site Scripting (XSS) occurs when a web application’s client-side script processes data from an untrusted source in an unsafe way, directly modifying the DOM without proper validation or encoding. This type of XSS occurs entirely on the client-side.

## Objective
To identify DOM-based XSS vulnerabilities by analyzing how client-side scripts handle and manipulate untrusted data.

## Testing Procedure

### 1. Identify DOM Manipulation Points
- **Description**: Locate points where the application’s client-side scripts read and write data to the DOM.
- **Steps**:
  1. Use browser developer tools to inspect the application’s JavaScript code.
  2. Look for functions or methods such as `document.write()`, `innerHTML`, `outerHTML`, `location`, `eval()`, or `setTimeout()`.
  3. Identify input points that affect these DOM manipulation functions.

### 2. Inject Payloads
- **Description**: Test identified points with XSS payloads.
- **Steps**:
  1. Inject payloads like `<script>alert('DOM XSS')</script>` or `<img src=x onerror=alert('DOM XSS')>`.
  2. Input payloads into URL parameters, form fields, or other inputs that interact with the DOM.
  3. Observe if the payload is executed within the browser.

### 3. Analyze Client-Side Code
- **Description**: Review the application’s JavaScript code for unsafe practices.
- **Steps**:
  1. Look for dynamic data handling without proper encoding or validation.
  2. Trace the flow of untrusted data to determine if it reaches sensitive DOM functions.
  3. Verify if the application uses libraries or frameworks that sanitize inputs (e.g., DOMPurify).

### 4. Test for Context-Specific XSS
- **Description**: Verify execution of XSS in various contexts like HTML, attributes, or JavaScript.
- **Steps**:
  1. Inject context-specific payloads (e.g., `" onmouseover=alert('DOM XSS')` for attributes or `'); alert('DOM XSS');//` for JavaScript).
  2. Observe if the payload executes in the respective context.

### 5. Test for URL-Based Vulnerabilities
- **Description**: Check if the application uses data from the URL without sanitization.
- **Steps**:
  1. Modify URL fragments or query parameters with XSS payloads.
  2. Analyze if these values are used unsafely in DOM manipulations.

### 6. Inspect Third-Party Libraries
- **Description**: Assess the usage of third-party libraries or plugins for vulnerabilities.
- **Steps**:
  1. Identify third-party JavaScript libraries used by the application.
  2. Research known vulnerabilities in these libraries.
  3. Test if they handle untrusted data securely.

## Tools
- Burp Suite (DOM Invader)
- OWASP ZAP
- Browser Developer Tools
- XSS Payload Generators

## Remediation
1. Use secure JavaScript methods such as `textContent` or `setAttribute` instead of `innerHTML` or `eval()`.
2. Validate and sanitize all data before processing it in the DOM.
3. Implement a library like DOMPurify to sanitize untrusted data.
4. Avoid directly embedding untrusted input into client-side scripts.
5. Educate developers on secure coding practices to mitigate DOM-based XSS.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)
- [OWASP DOM-Based XSS Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/DOM_based_XSS_Prevention_Cheat_Sheet.html)
