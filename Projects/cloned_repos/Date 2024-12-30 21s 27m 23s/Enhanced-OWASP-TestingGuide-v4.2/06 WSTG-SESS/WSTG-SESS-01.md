# WSTG-SESS-01 - Testing for Bypassing Session Management Schema

## Summary
Session management is a critical component in web applications. It ensures users are authenticated and authorized throughout their interaction with the application. Bypassing the session management schema can lead to unauthorized access, data leaks, and security breaches.

## Objective
To identify weaknesses in the session management schema that could allow attackers to bypass authentication or hijack sessions.

## Testing Procedure

### 1. Identify Session Tokens
- **Description**: Examine how session tokens are managed, transmitted, and stored.
- **Steps**:
  1. Log in to the application and intercept the session token using tools like Burp Suite or OWASP ZAP.
  2. Check for session tokens in cookies, headers, or URL parameters.
  3. Verify if the token format is predictable (e.g., sequential, timestamp-based).

### 2. Analyze Session Token Security
- **Description**: Determine if the session token is secure.
- **Steps**:
  1. Check for secure flag and HTTPOnly attributes in cookies.
  2. Verify if the token is encrypted or encoded.
  3. Confirm that tokens are invalidated upon logout.

### 3. Session Fixation Testing
- **Description**: Test if an attacker can force a victim to use a specific session token.
- **Steps**:
  1. Log in to the application and capture a valid session token.
  2. Logout and log in again to ensure a new token is issued.
  3. Attempt to reuse the old token and observe if it is still valid.

### 4. Session Hijacking
- **Description**: Check if an attacker can hijack an active session.
- **Steps**:
  1. Use tools like Wireshark to capture network traffic.
  2. Look for session tokens transmitted in plaintext (e.g., over HTTP).
  3. Replay the captured token to gain unauthorized access.

### 5. Session Timeout Enforcement
- **Description**: Validate if the application enforces a session timeout policy.
- **Steps**:
  1. Log in and remain idle for a period of time.
  2. Attempt to access protected resources after the timeout period.
  3. Verify if the session is invalidated.

### 6. Test for Cross-Site Script Inclusion (XSSI)
- **Description**: Determine if session tokens are vulnerable to XSSI attacks.
- **Steps**:
  1. Inject malicious scripts in input fields or URLs.
  2. Observe if the script can access session tokens.
  3. Check for Content Security Policy (CSP) implementation.

## Tools
- Burp Suite
- OWASP ZAP
- Wireshark
- Browser Developer Tools

## Remediation
1. Use secure, unpredictable session tokens.
2. Always set the secure and HTTPOnly flags for cookies.
3. Implement session timeout and logout functionality.
4. Ensure all session data is transmitted over HTTPS.
5. Regularly regenerate session tokens.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
