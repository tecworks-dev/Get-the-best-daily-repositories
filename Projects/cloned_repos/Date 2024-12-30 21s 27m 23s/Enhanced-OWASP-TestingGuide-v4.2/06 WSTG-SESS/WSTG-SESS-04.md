# WSTG-SESS-04 - Testing for Exposed Session Variables

## Summary
Exposing session variables in URLs, error messages, or client-side scripts can lead to information disclosure and session hijacking. Sensitive session-related data should never be exposed to the client-side or transmitted in insecure channels.

## Objective
To identify whether session variables are improperly exposed, leading to potential security vulnerabilities.

## Testing Procedure

### 1. Check for Session Variables in URLs
- **Description**: Examine if session variables are transmitted in URL parameters.
- **Steps**:
  1. Log in to the application and navigate through various pages.
  2. Observe the URL for session variables (e.g., session IDs, tokens).
  3. Attempt to modify the session variables in the URL and observe the behavior.

### 2. Analyze Client-Side Scripts
- **Description**: Check if session variables are exposed in client-side scripts.
- **Steps**:
  1. Use browser developer tools to review JavaScript files and inline scripts.
  2. Search for session variables, such as session IDs, tokens, or user information.
  3. Verify if any sensitive data is stored or processed on the client side.

### 3. Inspect HTML Source Code
- **Description**: Identify if session variables are exposed in the HTML source code.
- **Steps**:
  1. View the page source of the application using browser developer tools.
  2. Search for session variables in hidden fields, meta tags, or inline scripts.
  3. Confirm if sensitive information is present.

### 4. Test Error Messages
- **Description**: Check if session variables are leaked in error messages.
- **Steps**:
  1. Intentionally trigger errors by providing invalid input or modifying requests.
  2. Analyze error messages for exposed session-related data.
  3. Ensure no sensitive session data is included in the response.

### 5. Monitor Network Traffic
- **Description**: Observe how session variables are transmitted over the network.
- **Steps**:
  1. Use tools like Wireshark or Burp Suite to capture network traffic.
  2. Verify if session variables are sent over insecure channels (e.g., HTTP).
  3. Confirm if sensitive session data is included in request headers, parameters, or responses.

## Tools
- Burp Suite
- OWASP ZAP
- Wireshark
- Browser Developer Tools

## Remediation
1. Do not include session variables in URLs.
2. Store session variables securely on the server side.
3. Avoid exposing session-related data in client-side scripts or HTML source code.
4. Use HTTPS to secure all communications.
5. Handle errors gracefully without exposing sensitive session information.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
