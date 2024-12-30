# OWASP WSTG v4.0 - WSTG-ATHN-05

## Test Name: Testing for Vulnerable Remember Me Functionality

### Overview
This test evaluates the "Remember Me" functionality for security flaws that may allow unauthorized access to user accounts or compromise sensitive data.

---

### Objectives
- Verify the security of "Remember Me" tokens.
- Identify vulnerabilities in token storage and management.
- Ensure robust session handling when "Remember Me" is enabled.

---

### Test Steps

#### 1. **Analyze Token Behavior**
   - **Scenario**: Examine how the application generates and validates "Remember Me" tokens.
   - **Steps**:
     1. Enable "Remember Me" during login.
     2. Capture the token stored in cookies or local storage.
     3. Analyze the token’s structure and content.
   - **Indicators**:
     - Tokens contain predictable or sensitive information (e.g., user ID, session ID).
     - Lack of encryption or signing.

#### 2. **Inspect Token Lifetime and Expiry**
   - **Scenario**: Validate that tokens have a reasonable expiration policy.
   - **Steps**:
     1. Log in with "Remember Me" enabled.
     2. Wait for the token to expire (or set the system clock forward).
     3. Attempt to use the expired token.
   - **Indicators**:
     - Tokens do not expire or remain valid indefinitely.
     - Expiration policies are inconsistent or poorly enforced.

#### 3. **Test for Token Replay**
   - **Scenario**: Determine if the same token can be reused across multiple sessions.
   - **Steps**:
     1. Log in on one device with "Remember Me" enabled.
     2. Use the same token on a different device or browser.
     3. Observe whether access is granted without re-authentication.
   - **Indicators**:
     - Tokens can be reused across devices or sessions.
     - No mechanisms to revoke or invalidate tokens upon misuse.

#### 4. **Check Secure Storage of Tokens**
   - **Scenario**: Ensure tokens are securely stored on the client-side.
   - **Steps**:
     1. Inspect where the token is stored (e.g., cookies, local storage, session storage).
     2. Verify if secure flags (e.g., `HttpOnly`, `Secure`) are set for cookies.
   - **Indicators**:
     - Tokens stored in insecure locations like local storage or visible in the DOM.
     - Cookies lack secure attributes (e.g., `Secure`, `HttpOnly`).

#### 5. **Validate Logout and Session Revocation**
   - **Scenario**: Confirm that "Remember Me" tokens are invalidated upon logout.
   - **Steps**:
     1. Log in with "Remember Me" enabled.
     2. Log out and attempt to reuse the token.
   - **Indicators**:
     - Tokens remain valid after logout.
     - No mechanism to revoke tokens upon account changes.

---

### Tools
- **Burp Suite**: Intercept and analyze tokens.
- **Postman**: Test API endpoints for token validation.
- **JWT.io**: Decode and inspect JSON Web Tokens (if used).
- **Fiddler**: Monitor and modify HTTP requests.
- **Browser Developer Tools**: Inspect client-side token storage.

---

### Remediation
- Use secure, cryptographically signed, and encrypted tokens.
- Implement reasonable expiration policies for tokens.
- Ensure tokens are unique per session and user.
- Store tokens securely (e.g., cookies with `HttpOnly` and `Secure` flags).
- Revoke tokens upon logout or account changes.
- Avoid storing sensitive information in tokens.

---

### References
- [OWASP Testing Guide v4.0: Vulnerable Remember Me Functionality](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)

---

### Checklist
- [ ] Tokens are secure, encrypted, and signed.
- [ ] Tokens have a reasonable and enforced expiration policy.
- [ ] Tokens are unique per user and session.
- [ ] Secure storage mechanisms are used for tokens.
- [ ] Tokens are invalidated upon logout or misuse.
- [ ] No sensitive data is stored in tokens.
