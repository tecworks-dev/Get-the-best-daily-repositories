# WSTG-SESS-06 - Testing for Logout Functionality

## Summary
Proper logout functionality ensures that a user's session is securely terminated upon logout, preventing unauthorized access to the application. Flaws in logout mechanisms can lead to session hijacking or unauthorized access.

## Objective
To verify that the logout functionality effectively terminates the user session and invalidates the session ID.

## Testing Procedure

### 1. Test Logout Behavior
- **Description**: Ensure that the application terminates the session upon logout.
- **Steps**:
  1. Log in to the application and capture the session ID using tools like Burp Suite or browser developer tools.
  2. Log out and attempt to access the application using the same session ID.
  3. Verify if the session is invalidated.

### 2. Check for Redirects After Logout
- **Description**: Confirm that users are redirected to a public page or the login page after logout.
- **Steps**:
  1. Log in and perform logout.
  2. Observe the behavior of the application after logout.
  3. Ensure no sensitive information is displayed post-logout.

### 3. Test Session Persistence
- **Description**: Verify that the session does not persist after logout.
- **Steps**:
  1. Log in and open multiple browser tabs or devices.
  2. Log out from one tab or device.
  3. Verify if the session remains active on other tabs or devices.

### 4. Check Token Revocation
- **Description**: Ensure that access and refresh tokens are revoked upon logout.
- **Steps**:
  1. Log in and capture any tokens issued.
  2. Log out and attempt to use the captured tokens to access protected resources.
  3. Confirm that the tokens are no longer valid.

### 5. Test Logout for All User States
- **Description**: Verify that logout works consistently across different user states (e.g., authenticated, idle).
- **Steps**:
  1. Log in and log out immediately to test for active sessions.
  2. Log in, remain idle for the session timeout period, and then log out.
  3. Observe if logout terminates all session-related data.

### 6. Verify Logout Mechanism
- **Description**: Check if the logout functionality is implemented securely.
- **Steps**:
  1. Analyze the logout request using tools like Burp Suite.
  2. Verify if the application clears session cookies.
  3. Confirm that the logout mechanism does not rely on client-side operations alone.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools

## Remediation
1. Invalidate the session ID upon logout.
2. Revoke all tokens (access and refresh) upon logout.
3. Clear session cookies and set the `Secure` and `HTTPOnly` attributes.
4. Redirect users to a non-sensitive page after logout.
5. Ensure that logout functionality works across all user states and devices.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
