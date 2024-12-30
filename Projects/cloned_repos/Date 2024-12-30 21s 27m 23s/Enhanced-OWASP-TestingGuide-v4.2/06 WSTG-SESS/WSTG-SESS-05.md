# WSTG-SESS-05 - Testing for Cross-Site Request Forgery (CSRF)

## Summary
Cross-Site Request Forgery (CSRF) is an attack where unauthorized commands are transmitted from a user that the application trusts. Properly implemented anti-CSRF mechanisms can prevent such attacks by ensuring that requests are initiated by authenticated users.

## Objective
To determine whether the application is vulnerable to CSRF attacks and to verify the effectiveness of its anti-CSRF mechanisms.

## Testing Procedure

### 1. Identify Actions Vulnerable to CSRF
- **Description**: Locate functionalities that could be exploited through CSRF attacks (e.g., account updates, fund transfers).
- **Steps**:
  1. Use the application and identify sensitive actions.
  2. Capture requests for these actions using tools like Burp Suite or browser developer tools.
  3. Note the presence or absence of CSRF tokens in the requests.

### 2. Test for CSRF Tokens
- **Description**: Verify if CSRF tokens are implemented and properly validated.
- **Steps**:
  1. Capture the request for a sensitive action.
  2. Look for CSRF tokens in headers, cookies, or hidden fields.
  3. Modify the token or remove it from the request.
  4. Replay the request and observe if the action is performed.

### 3. Inspect CSRF Token Properties
- **Description**: Ensure CSRF tokens are unique, unpredictable, and tied to the user session.
- **Steps**:
  1. Perform multiple sensitive actions and capture the requests.
  2. Verify if each request uses a unique CSRF token.
  3. Confirm that tokens are not predictable.

### 4. Test for Token Reuse
- **Description**: Check if the application allows the reuse of CSRF tokens.
- **Steps**:
  1. Capture a valid request with a CSRF token.
  2. Replay the request multiple times using the same token.
  3. Observe if the application accepts the reused token.

### 5. Check for SameSite Cookie Attribute
- **Description**: Verify if cookies are configured with the `SameSite` attribute to mitigate CSRF attacks.
- **Steps**:
  1. Capture cookies in the `Set-Cookie` header.
  2. Check if the `SameSite` attribute is set to `Lax` or `Strict`.
  3. Test cross-origin requests to confirm cookie behavior.

### 6. Test for CSRF Vulnerability Without Tokens
- **Description**: Create a simple CSRF attack scenario.
- **Steps**:
  1. Craft an HTML form that mimics a sensitive request.
  2. Host the form on a different domain or locally.
  3. Submit the form and observe if the action is performed without authentication.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools

## Remediation
1. Implement anti-CSRF tokens for all sensitive actions.
2. Use unique, unpredictable, and session-bound CSRF tokens.
3. Configure cookies with the `SameSite` attribute to `Lax` or `Strict`.
4. Validate the origin and referer headers in requests.
5. Educate developers on secure coding practices to mitigate CSRF.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
