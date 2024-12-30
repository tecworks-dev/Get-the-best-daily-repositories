# WSTG-SESS-02 - Testing for Cookies Attributes

## Summary
Cookies are widely used for session management, authentication, and tracking user activities. Improper cookie configurations can lead to security vulnerabilities such as session hijacking, cross-site scripting (XSS), and cross-site request forgery (CSRF).

## Objective
To evaluate cookie attributes and identify misconfigurations that could lead to security issues.

## Testing Procedure

### 1. Identify Cookies
- **Description**: Determine the cookies set by the application.
- **Steps**:
  1. Log in to the application and capture HTTP requests and responses using tools like Burp Suite or browser developer tools.
  2. Analyze all cookies set in the `Set-Cookie` header or JavaScript.
  3. Note the purpose and origin of each cookie.

### 2. Check Secure Attribute
- **Description**: Verify that the `Secure` attribute is set for cookies transmitted over HTTPS.
- **Steps**:
  1. Identify cookies in the `Set-Cookie` header.
  2. Confirm if the `Secure` attribute is present.
  3. Attempt to transmit the cookie over HTTP and verify if it is transmitted.

### 3. Check HTTPOnly Attribute
- **Description**: Ensure the `HTTPOnly` attribute is set to prevent client-side scripts from accessing cookies.
- **Steps**:
  1. Identify cookies and check if the `HTTPOnly` attribute is present.
  2. Use browser developer tools or JavaScript (`document.cookie`) to test if cookies can be accessed.

### 4. Check SameSite Attribute
- **Description**: Verify that the `SameSite` attribute is set to mitigate CSRF attacks.
- **Steps**:
  1. Identify cookies and check for the `SameSite` attribute.
  2. Confirm if it is set to `Strict`, `Lax`, or `None`.
  3. Test cross-origin requests to observe cookie behavior.

### 5. Check Expiration and Persistence
- **Description**: Analyze cookie expiration times and persistence settings.
- **Steps**:
  1. Identify session and persistent cookies.
  2. Verify if session cookies are properly invalidated upon logout.
  3. Check if persistent cookies have unnecessarily long expiration times.

### 6. Check for Excessive Cookies
- **Description**: Ensure the application is not setting excessive or unnecessary cookies.
- **Steps**:
  1. List all cookies and their purposes.
  2. Verify if any cookies are redundant or irrelevant.

### 7. Test for Insecure Transmission
- **Description**: Verify that cookies are not transmitted over unencrypted channels.
- **Steps**:
  1. Intercept network traffic using tools like Wireshark.
  2. Confirm that cookies are only sent over HTTPS.
  3. Test if cookies are sent over HTTP by forcing the application to use an insecure connection.

## Tools
- Burp Suite
- OWASP ZAP
- Wireshark
- Browser Developer Tools

## Remediation
1. Always set the `Secure` attribute for cookies.
2. Use the `HTTPOnly` attribute to prevent client-side access to cookies.
3. Set the `SameSite` attribute to `Strict` or `Lax` where applicable.
4. Limit cookie expiration to an appropriate timeframe.
5. Avoid setting unnecessary or excessive cookies.
6. Transmit cookies only over HTTPS.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
