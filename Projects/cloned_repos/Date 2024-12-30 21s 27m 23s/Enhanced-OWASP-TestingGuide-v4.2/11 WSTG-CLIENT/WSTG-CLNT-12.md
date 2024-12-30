Here’s the documentation for **WSTG-CLNT-12** in Markdown format:


# WSTG-CLNT-12: Testing for Cross-Origin Window Interaction

## Summary

Cross-Origin Window Interaction vulnerabilities occur when an attacker can interact with or manipulate content in a cross-origin window or iframe. This can lead to unauthorized access, sensitive data leakage, or unintended actions performed on behalf of the user.

## Objective

To identify and exploit weaknesses in cross-origin interactions between windows or iframes, such as:

- Data leakage through `postMessage` or DOM manipulation.
- Unauthorized control over cross-origin content.
- Security bypass through improper sandboxing or window interaction.

## How to Test

### Step 1: Identify Cross-Origin Interactions
1. Locate areas in the application where:
   - Cross-origin iframes or windows are used.
   - The application sends or receives data using `postMessage`.
   - Windows or frames interact with each other via JavaScript.

2. Inspect browser developer tools to identify:
   - Cross-origin frames loaded in the application.
   - Events triggered by `postMessage`.

---

### Step 2: Analyze `postMessage` Implementation
1. Review the application’s use of `postMessage`:
   - Inspect the origin validation in the `message` event handler.
   - Check if the handler properly validates the sender’s origin using `event.origin`.

2. Test for weaknesses:
   - Send messages from unauthorized origins.
   - Inject malicious payloads through `postMessage`.

3. Example malicious script to send a message:
   ```javascript
   const maliciousWindow = window.open('https://target.com');
   maliciousWindow.postMessage('maliciousData', '*');
```

---

### Step 3: Test for Frame and Window Manipulation

1. **Frame Redress**:
    
    - Attempt to manipulate the DOM of a cross-origin iframe.
    - Example:
        
        ```javascript
        const iframe = document.querySelector('iframe');
        iframe.contentWindow.document.body.innerHTML = 'Hacked!';
        ```
        
2. **Clickjacking Through Frames**:
    
    - Test if the cross-origin frame can be overlaid or interacted with invisibly.
3. **Untrusted Script Injection**:
    
    - Inject scripts or manipulate data in the context of the cross-origin window.

---

### Step 4: Test Sandboxing and Permissions

1. Inspect iframe sandbox attributes:
    
    - Verify the presence of `sandbox` and its restrictions (e.g., `allow-scripts`, `allow-forms`).
2. Test if the iframe can:
    
    - Execute scripts.
    - Interact with the parent window.
    - Access other cross-origin resources.
3. Example iframe sandboxing:
    
    ```html
    <iframe src="https://example.com" sandbox="allow-scripts"></iframe>
    ```
    

---

### Step 5: Analyze Results

1. Confirm if the cross-origin window or iframe can:
    
    - Leak sensitive data through `postMessage`.
    - Be manipulated to perform unauthorized actions.
    - Execute scripts or interact with the parent window.
2. Assess the potential impact of these vulnerabilities.
    

---

## Tools

- **Browser Developer Tools** for inspecting network requests, DOM, and JavaScript interactions.
- **Burp Suite** or **OWASP ZAP** for analyzing cross-origin requests and scripts.
- **Custom JavaScript Scripts** for testing `postMessage` and frame interactions.
- **Clickjacking Testing Tools** to simulate overlay attacks.

---

## Remediation

1. **Validate `postMessage` Origins**:
    
    - Always validate the sender’s origin in the `message` event handler:
        
        ```javascript
        window.addEventListener('message', (event) => {
          if (event.origin !== 'https://trusted.com') return;
          // Process message
        });
        ```
        
2. **Restrict Frame Permissions**:
    
    - Use the `sandbox` attribute for iframes to limit functionality:
        
        ```html
        <iframe src="https://example.com" sandbox="allow-scripts"></iframe>
        ```
        
3. **Prevent Cross-Origin Data Leakage**:
    
    - Avoid storing sensitive data in accessible cross-origin resources.
4. **Implement Clickjacking Protection**:
    
    - Use `X-Frame-Options` or `Content-Security-Policy: frame-ancestors` to restrict framing.
5. **Regularly Test Cross-Origin Interactions**:
    
    - Conduct periodic security testing to identify and address cross-origin vulnerabilities.

---

## References

- [OWASP Testing Guide - Cross-Origin Interaction Testing](https://owasp.org/www-project-testing/)
- [MDN Web Docs - Window.postMessage()](https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)
- [W3C - HTML5 Sandbox Attribute](https://html.spec.whatwg.org/multipage/iframe-embed-object.html#attr-iframe-sandbox)


