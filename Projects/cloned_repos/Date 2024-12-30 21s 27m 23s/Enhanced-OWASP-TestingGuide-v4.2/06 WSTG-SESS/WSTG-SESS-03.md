# WSTG-SESS-03 - Testing for Session Fixation

## Summary
Session fixation attacks occur when an attacker sets or manipulates a user's session ID before the user logs in. This allows the attacker to hijack the user’s authenticated session. Applications must ensure that a new session ID is issued after a successful login.

## Objective
To determine whether the application is vulnerable to session fixation attacks and ensure that a new session ID is issued upon authentication.

## Testing Procedure

### 1. Analyze Session ID Behavior
- **Description**: Identify how session IDs are assigned and managed before and after authentication.
- **Steps**:
  1. Log in to the application and capture the session ID using tools like Burp Suite or browser developer tools.
  2. Log out and log in again to observe if the session ID changes.
  3. Verify if the session ID remains constant during the login process.

### 2. Test for Pre-set Session ID
- **Description**: Check if the application allows the session ID to be pre-set.
- **Steps**:
  1. Capture a session ID before login.
  2. Use an HTTP client (e.g., Burp Suite, Postman) to set a predefined session ID in the `Cookie` header.
  3. Log in to the application and observe if the predefined session ID is retained.

### 3. Check for New Session ID on Login
- **Description**: Verify if a new session ID is issued upon authentication.
- **Steps**:
  1. Capture the session ID before login.
  2. Authenticate with valid credentials.
  3. Verify if the session ID is regenerated after login.

### 4. Test for Logout Behavior
- **Description**: Ensure the session ID is invalidated upon logout.
- **Steps**:
  1. Log in and capture the session ID.
  2. Log out and attempt to reuse the captured session ID.
  3. Observe if access is denied for the old session ID.

### 5. Test Session Fixation via URL Parameters
- **Description**: Check if the application accepts session IDs passed in URL parameters.
- **Steps**:
  1. Identify if the session ID is present in URL parameters.
  2. Attempt to modify or preset the session ID in the URL.
  3. Observe if the session ID is retained or regenerated.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools

## Remediation
1. Always regenerate the session ID after a successful login.
2. Invalidate the old session ID after logout.
3. Do not accept session IDs from URL parameters.
4. Use secure and unpredictable session IDs.
5. Implement HTTPOnly and Secure attributes for session cookies.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
