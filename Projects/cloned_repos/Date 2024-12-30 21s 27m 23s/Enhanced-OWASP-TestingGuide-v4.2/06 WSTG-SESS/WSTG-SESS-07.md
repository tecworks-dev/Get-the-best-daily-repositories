# WSTG-SESS-07 - Testing Session Timeout

## Summary
Session timeout is a security control designed to reduce the risk of unauthorized access to an application when a user is inactive. Proper implementation ensures that inactive sessions are terminated after a specified period.

## Objective
To verify that the application enforces session timeouts and properly terminates inactive sessions.

## Testing Procedure

### 1. Identify Session Timeout Policy
- **Description**: Determine the session timeout policy implemented by the application.
- **Steps**:
  1. Review application documentation or security policies for session timeout details.
  2. Log in to the application and remain idle.
  3. Observe after how much time the session expires or the user is logged out.

### 2. Test Idle Session Timeout
- **Description**: Verify if the session is terminated after a period of inactivity.
- **Steps**:
  1. Log in to the application and remain idle for the specified timeout period.
  2. Attempt to perform actions after the timeout period.
  3. Confirm if the session is terminated and the user is prompted to log in again.

### 3. Test Active Session Handling
- **Description**: Ensure that active user sessions are not prematurely terminated.
- **Steps**:
  1. Log in to the application and perform actions periodically.
  2. Confirm if the session remains active without interruptions.

### 4. Check Persistent Sessions
- **Description**: Test if session timeout applies to "Remember Me" or persistent login features.
- **Steps**:
  1. Log in with the "Remember Me" feature enabled.
  2. Remain idle for the timeout period.
  3. Verify if the session is terminated or persists as expected.

### 5. Verify Token Expiry
- **Description**: Check if session tokens expire after the timeout period.
- **Steps**:
  1. Log in and capture session tokens (e.g., cookies, JWT).
  2. Remain idle for the timeout period.
  3. Attempt to reuse the session tokens after timeout and observe the response.

### 6. Test for Sliding Session Timeout
- **Description**: Verify if the session timeout is reset upon user activity.
- **Steps**:
  1. Log in to the application and perform actions periodically.
  2. Confirm if the session timeout period is extended with each user action.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools

## Remediation
1. Implement a session timeout policy aligned with application risk levels.
2. Invalidate sessions and associated tokens after the timeout period.
3. Provide users with clear feedback about session timeout.
4. Ensure that the session timeout resets upon legitimate user activity.
5. Test timeout functionality across all user states, including "Remember Me" sessions.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
