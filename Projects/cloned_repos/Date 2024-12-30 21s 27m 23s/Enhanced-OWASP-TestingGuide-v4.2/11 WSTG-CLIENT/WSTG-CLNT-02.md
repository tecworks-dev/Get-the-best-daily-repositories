# WSTG-CLNT-02: Testing for JavaScript Execution

## Summary

JavaScript execution vulnerabilities occur when an attacker is able to inject or execute malicious JavaScript code within the client-side environment. These vulnerabilities often lead to issues such as Cross-Site Scripting (XSS), code execution, or unauthorized access to sensitive data.

## Objective

To identify and exploit scenarios where JavaScript execution is possible, allowing attackers to execute malicious scripts and compromise application functionality or user data.

## How to Test

### Step 1: Identify Potential Entry Points
1. Locate areas in the application where JavaScript code can be influenced by user input, such as:
   - Input fields
   - URL parameters
   - Query strings or fragments
   - Cookies and localStorage

2. Focus on dynamic features of the application that process or display user-controlled data.

---

### Step 2: Analyze JavaScript Code
1. Review the application for instances of JavaScript execution using:
   - `eval()` or similar functions
   - `setTimeout()`, `setInterval()`
   - Dynamic DOM updates via `innerHTML`, `outerHTML`, or `document.write()`

2. Pay attention to areas where user inputs are concatenated or inserted directly into JavaScript code or the DOM.

---

### Step 3: Perform Testing for JavaScript Execution
1. **Inject Malicious Scripts**:
   - Use common payloads to test for script execution, such as:
     ```javascript
     <script>alert('XSS')</script>
     "><img src=x onerror=alert(1)>
     ```
   - Inject payloads into all identified entry points.

2. **Inspect DOM Manipulation**:
   - Use browser developer tools to observe how inputs are processed.
   - Analyze the DOM structure for unexpected changes caused by injected inputs.

3. **Test JavaScript Functions**:
   - If JavaScript functions take user inputs, test with payloads like:
     ```javascript
     '; alert(1); var a = '
     ```

4. **Check API Responses**:
   - Intercept and analyze API responses for any signs of unsafe JavaScript handling.

---

### Step 4: Analyze Execution Behavior
1. Confirm if the payload is executed in the browser context.
2. Verify if the vulnerability allows for:
   - Persistent JavaScript execution
   - Stealing cookies, tokens, or other sensitive data
   - Exploitation across multiple users

---

## Tools

- **Browser Developer Tools** for DOM inspection and testing
- **Burp Suite** or **OWASP ZAP** for intercepting and modifying requests
- **Fuzzer Tools** for testing a wide range of JavaScript payloads
- **Payload Libraries** like XSS Hunter for advanced script payloads

---

## Remediation

1. **Sanitize User Inputs**:
   - Use libraries like DOMPurify to sanitize inputs before they are processed or displayed in the browser.

2. **Avoid Unsafe JavaScript Functions**:
   - Avoid using `eval()`, `setTimeout()` with string arguments, or `document.write()`.

3. **Apply Context-Aware Encoding**:
   - Encode outputs to match the context (HTML, JavaScript, etc.) where the data will be used.

4. **Implement Content Security Policy (CSP)**:
   - Use a restrictive CSP to mitigate the execution of unauthorized scripts.

5. **Regularly Review Client-Side Code**:
   - Conduct code reviews to identify unsafe JavaScript practices.

---

## References

- [OWASP Testing Guide - Testing for JavaScript Execution](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/)
- [Google Security - JavaScript Security Best Practices](https://developers.google.com/web/fundamentals/security)

---
