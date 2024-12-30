# WSTG-SESS-09 - Testing for Session Hijacking

## Summary
Session hijacking involves stealing or using a legitimate user's session to gain unauthorized access to the application. This can occur due to weak session management, unencrypted communication, or exposure of session tokens.

## Objective
To determine if the application is vulnerable to session hijacking and identify potential mitigations.

## Testing Procedure

### 1. Capture Session Tokens
- **Description**: Identify where session tokens are stored and transmitted.
- **Steps**:
  1. Log in to the application and intercept HTTP requests using tools like Burp Suite or OWASP ZAP.
  2. Capture session tokens stored in cookies, headers, or local storage.
  3. Analyze how tokens are transmitted (e.g., via HTTP or HTTPS).

### 2. Test for Session Token Exposure
- **Description**: Verify if session tokens are exposed in URLs, logs, or client-side code.
- **Steps**:
  1. Examine URL parameters for session tokens.
  2. Inspect browser developer tools and JavaScript files for exposed tokens.
  3. Check if session tokens are included in application logs.

### 3. Simulate Network Sniffing
- **Description**: Capture session tokens over the network to test for encryption.
- **Steps**:
  1. Use a tool like Wireshark to monitor network traffic.
  2. Analyze captured traffic for session tokens transmitted over unencrypted HTTP.
  3. Confirm if HTTPS is enforced for all session-related communication.

### 4. Test for Token Replay
- **Description**: Replay captured session tokens to gain unauthorized access.
- **Steps**:
  1. Capture a valid session token.
  2. Use an HTTP client (e.g., Postman) to resend requests with the captured token.
  3. Observe if the application accepts the replayed token.

### 5. Test for Cross-Site Scripting (XSS)
- **Description**: Check if session tokens can be accessed via XSS vulnerabilities.
- **Steps**:
  1. Identify input fields or parameters susceptible to XSS.
  2. Inject scripts to steal session tokens (e.g., `document.cookie`).
  3. Observe if the script successfully retrieves the session token.

### 6. Test Session Timeout and Logout
- **Description**: Verify if session tokens are invalidated upon logout or after a timeout.
- **Steps**:
  1. Log in and capture the session token.
  2. Logout or remain idle until the session expires.
  3. Attempt to use the captured session token after logout or timeout.

## Tools
- Burp Suite
- OWASP ZAP
- Wireshark
- Postman
- Browser Developer Tools

## Remediation
1. Use secure, unpredictable session tokens.
2. Always transmit session tokens over HTTPS.
3. Store session tokens securely using `HttpOnly` and `Secure` flags.
4. Implement session expiration and logout mechanisms.
5. Protect against XSS to prevent token theft.
6. Avoid including session tokens in URLs or logs.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
