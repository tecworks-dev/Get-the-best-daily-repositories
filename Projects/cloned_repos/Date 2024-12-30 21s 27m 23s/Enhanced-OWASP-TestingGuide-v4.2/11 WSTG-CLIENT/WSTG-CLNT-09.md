Here’s the documentation for **WSTG-CLNT-09** in Markdown format:


# WSTG-CLNT-09: Testing for WebSockets

## Summary

WebSockets are a protocol that enables bidirectional communication between the client and server. Improper implementation of WebSocket security can lead to vulnerabilities such as unauthorized access, data leakage, or man-in-the-middle (MITM) attacks.

## Objective

To identify and exploit weaknesses in the WebSocket implementation, including:

- Lack of authentication or authorization
- Insufficient input validation
- Data leakage through unsecured connections
- Susceptibility to injection attacks or manipulation

## How to Test

### Step 1: Identify WebSocket Endpoints
1. Inspect the application for WebSocket connections by analyzing:
   - Browser developer tools (Network tab → WS or WebSocket filters).
   - JavaScript code initiating WebSocket connections (e.g., `new WebSocket()`).

2. Record the WebSocket URLs and parameters.

---

### Step 2: Analyze WebSocket Handshake
1. Review the WebSocket handshake process by inspecting HTTP headers:
   - `Upgrade: websocket`
   - `Connection: Upgrade`
   - `Sec-WebSocket-Key`
   - `Sec-WebSocket-Version`

2. Verify if the WebSocket handshake is protected with HTTPS (`wss://`) to prevent MITM attacks.

---

### Step 3: Test Authentication and Authorization
1. Ensure that WebSocket connections require authentication and enforce user permissions.
2. Test scenarios such as:
   - Connecting without valid credentials.
   - Reusing or hijacking WebSocket sessions from another user.
   - Accessing resources or data beyond the user's permissions.

---

### Step 4: Test for Input Validation Issues
1. Inject malicious payloads into WebSocket messages to test for vulnerabilities:
   - SQL injection: `{"query": "SELECT * FROM users WHERE id='1' OR '1'='1';"}`
   - Cross-Site Scripting (XSS): `<script>alert('XSS')</script>`
   - Command injection: `{"command": "shutdown -h now"}`

2. Check the server’s response to determine if it processes or reflects the injected payloads.

---

### Step 5: Test for Data Leakage
1. Monitor WebSocket messages to identify sensitive information being transmitted:
   - Tokens
   - Credentials
   - Personally Identifiable Information (PII)

2. Verify if sensitive data is encrypted and not exposed in plaintext.

---

### Step 6: Test for Denial of Service (DoS)
1. Simulate high-frequency or oversized messages to test if the WebSocket server can handle the load:
   ```javascript
   for (let i = 0; i < 10000; i++) {
       ws.send("Flood message " + i);
   }
```

2. Observe the server’s behavior for crashes or degraded performance.

---

### Step 7: Analyze Results

1. Confirm if WebSocket vulnerabilities can be exploited.
2. Assess the impact, such as:
    - Unauthorized access to data or functionality.
    - Data leakage or exposure.
    - Service disruption due to resource exhaustion.

---

## Tools

- **Browser Developer Tools** for analyzing WebSocket connections and messages.
- **Burp Suite** or **OWASP ZAP** for intercepting and manipulating WebSocket traffic.
- **WebSocket Testing Tools** like `wscat` or custom scripts for automated testing.
- **Custom Payloads** for injecting malicious data.

---

## Remediation

1. **Enforce Secure Connections**:
    
    - Use `wss://` (WebSocket Secure) to encrypt WebSocket communications.
2. **Authenticate and Authorize WebSocket Connections**:
    
    - Require user authentication and validate permissions for each WebSocket connection.
3. **Implement Input Validation and Sanitization**:
    
    - Validate and sanitize all WebSocket inputs on the server side to prevent injection attacks.
4. **Limit Resource Usage**:
    
    - Set limits on message size, frequency, and connection count to prevent DoS attacks.
5. **Secure Sensitive Data**:
    
    - Avoid transmitting sensitive information over WebSockets unless encrypted and necessary.
6. **Log and Monitor WebSocket Activity**:
    
    - Implement logging and monitoring to detect suspicious WebSocket activity.

---

## References

- [OWASP Testing Guide - WebSocket Testing](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A02:2021 Cryptographic Failures](https://owasp.org/Top10/A02_2021-Cryptographic_Failures/)
- [Mozilla Developer Network - WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)


