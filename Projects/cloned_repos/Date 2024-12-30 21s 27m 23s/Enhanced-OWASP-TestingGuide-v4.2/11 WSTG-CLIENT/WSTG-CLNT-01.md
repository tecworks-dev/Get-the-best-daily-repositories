# WSTG-CLNT-01: Testing for DOM-Based Cross-Site Scripting (XSS)

## Summary

DOM-based Cross-Site Scripting (XSS) occurs when malicious scripts are executed in the victim's browser due to unsafe handling of user-controlled input within the Document Object Model (DOM). Unlike traditional XSS, this vulnerability arises entirely in the client-side code without involving server-side input handling.

## Objective

To identify and exploit DOM-based XSS vulnerabilities by analyzing client-side scripts for unsafe manipulation of DOM elements using user-controlled inputs.

## How to Test

### Step 1: Identify Potential Entry Points
1. Review the application to locate:
   - Inputs that affect the DOM directly (e.g., URL parameters, hash fragments, HTML elements).
   - Features that reflect user input or interact dynamically with the DOM.

2. Common sources of user input:
   - `window.location`
   - `document.cookie`
   - `document.referrer`
   - `localStorage`, `sessionStorage`

---

### Step 2: Analyze JavaScript Code
1. Inspect JavaScript code to identify unsafe functions or methods, such as:
   - `innerHTML`, `outerHTML`
   - `document.write()`, `document.writeln()`
   - `eval()`, `setTimeout()`, `setInterval()`
   - `appendChild()`, `insertAdjacentHTML()`

2. Focus on scenarios where user input is inserted into the DOM without proper sanitization or encoding.

---

### Step 3: Test for DOM-Based XSS
1. **Use JavaScript Breakpoints**:
   - Set breakpoints in the browser’s developer tools to analyze DOM manipulation in real-time.
   - Example: Monitor sources like `window.location.search` or `document.referrer`.

2. **Inject Payloads into Inputs**:
   - Use common payloads to test for script execution:
     ```javascript
     <script>alert(1)</script>
     "><img src=x onerror=alert(1)>
     ```
   - Insert payloads into parameters such as:
     - Query strings (`?q=`)
     - Fragments (`#fragment`)
     - Cookies

3. **Analyze Reflection and Execution**:
   - Observe whether the payload is reflected and executed in the DOM.

4. **Simulate Real-World Scenarios**:
   - Test scenarios where input flows from one function to another without sanitization.
   - Example: A search query in the URL is displayed dynamically on the page.

---

### Step 4: Verify Exploitability
1. Determine if an attacker can exploit the issue remotely (e.g., by crafting a malicious link or embedding content).
2. Validate that the payload execution occurs without user intervention beyond basic interaction.

---

## Tools

- **Browser Developer Tools** for analyzing DOM and JavaScript execution
- **Burp Suite** with the DOM Invader extension for DOM XSS testing
- **OWASP ZAP** for passive scanning and manual analysis
- **Online Tools** like `DOM XSS Scanner` to detect unsafe DOM manipulations

---

## Remediation

1. **Sanitize and Encode User Inputs**:
   - Use libraries like DOMPurify to sanitize inputs before inserting them into the DOM.
   - Apply context-appropriate encoding (e.g., HTML, JavaScript, or URL encoding).

2. **Avoid Unsafe DOM Manipulations**:
   - Replace `innerHTML` with safer alternatives like `textContent` or `createElement()`.

3. **Validate Inputs on All Layers**:
   - Ensure input validation on both client and server sides.

4. **Use Content Security Policy (CSP)**:
   - Implement CSP to restrict script execution and mitigate the impact of XSS.

5. **Perform Regular Code Reviews**:
   - Review JavaScript code for unsafe patterns and apply secure coding practices.

---

## References

- [OWASP Testing Guide - DOM-Based XSS](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/)
- [Google Security - DOM XSS Prevention Cheat Sheet](https://developers.google.com/web/fundamentals/security/csp)

---
