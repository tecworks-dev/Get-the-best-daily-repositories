Here’s the documentation for **WSTG-CLNT-08** in Markdown format:


# WSTG-CLNT-08: Testing for Clickjacking

## Summary

Clickjacking is a web application vulnerability where an attacker tricks users into performing unintended actions on a website without their knowledge by embedding the target site into an invisible or disguised iframe. This can lead to unauthorized actions such as submitting forms, clicking buttons, or revealing sensitive information.

## Objective

To identify whether the application is vulnerable to clickjacking by testing if it can be embedded within an iframe and determining if critical actions can be manipulated through hidden overlays.

## How to Test

### Step 1: Verify Frame Embedding
1. Test if the application allows itself to be embedded within an iframe.
2. Use a simple HTML page to attempt embedding:
   ```html
   <iframe src="https://target-website.com" width="800" height="600"></iframe>
```

3. Observe if the application renders inside the iframe:
    - If it does, the application may be vulnerable.
    - If it doesn’t, verify if HTTP headers like `X-Frame-Options` or `Content-Security-Policy` are in use.

---

### Step 2: Simulate a Clickjacking Attack

1. Create an attack scenario using HTML and CSS to overlay the target site:
    
    ```html
    <html>
    <head>
        <style>
            iframe {
                opacity: 0.01;
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 2;
            }
            button {
                position: relative;
                z-index: 1;
            }
        </style>
    </head>
    <body>
        <button>Click Me!</button>
        <iframe src="https://target-website.com"></iframe>
    </body>
    </html>
    ```
    
2. Test if clicking on the visible button triggers an action on the hidden iframe.
    

---

### Step 3: Test Critical Actions

1. Focus on sensitive or critical actions such as:
    
    - Submitting forms
    - Transferring funds
    - Changing account settings
2. Simulate click events on these actions using the embedded iframe to determine if the application executes the actions.
    

---

### Step 4: Analyze Results

1. Confirm if the application can be embedded within an iframe without restriction.
2. Determine if critical actions can be triggered by clickjacking techniques.
3. Assess the impact of potential exploitation:
    - Unauthorized actions performed by the victim.
    - Compromise of sensitive data or account control.

---

## Tools

- **Browser Developer Tools** for inspecting headers and testing iframe behavior.
- **Burp Suite** for analyzing responses and identifying missing security headers.
- **Custom HTML Pages** to simulate clickjacking attacks.
- **Clickjacking Testing Tools** such as X-FRAME-OPTIONS and CSP checkers.

---

## Remediation

1. **Implement X-Frame-Options Header**:
    
    - Add the `X-Frame-Options` header to responses:
        - `DENY`: Prevents all framing.
        - `SAMEORIGIN`: Allows framing only from the same origin.
2. **Use Content Security Policy (CSP)**:
    
    - Add a `frame-ancestors` directive to CSP to specify allowed origins:
        
        ```http
        Content-Security-Policy: frame-ancestors 'self';
```        
        
3. **Avoid Clickjacking-Prone UI Designs**:
    
    - Avoid critical actions relying solely on clicks without additional validation or confirmation.
4. **Implement User Confirmation**:
    
    - Require user interaction for sensitive actions (e.g., re-entering credentials or CAPTCHA).
5. **Regularly Test for Clickjacking**:
    
    - Conduct periodic assessments to ensure headers and CSP configurations are in place and functioning correctly.

---

## References

- [OWASP Testing Guide - Clickjacking Testing](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)
- [Mozilla Developer Network - X-Frame-Options](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-Frame-Options)

---



