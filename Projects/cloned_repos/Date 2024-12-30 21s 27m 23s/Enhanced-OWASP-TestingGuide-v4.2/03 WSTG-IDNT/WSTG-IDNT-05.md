# OWASP WSTG v4.0 - WSTG-IDNT-05

## Test Name: Testing for Vulnerable Remember Me and User Sessions

### Overview
This test evaluates the implementation of "Remember Me" functionality and user session management to ensure they are secure and do not expose users to session hijacking or other vulnerabilities.

---

### Objectives
- Verify the security of "Remember Me" tokens and their storage.
- Assess the session management mechanism for vulnerabilities.
- Identify insecure practices in session handling and token generation.

---

### Test Steps

#### 1. **Analyze Remember Me Token Behavior**
   - **Scenario**: Check how "Remember Me" functionality is implemented.
   - **Steps**:
     1. Enable "Remember Me" during login.
     2. Inspect cookies, local storage, or other storage mechanisms for tokens.
     3. Analyze the structure and content of the token.
   - **Indicators**:
     - Tokens stored in plaintext or predictable formats.
     - Tokens containing sensitive information (e.g., passwords, user data).
     - Tokens lacking encryption or signatures.

#### 2. **Test Token Predictability**
   - **Scenario**: Determine if "Remember Me" tokens are predictable or guessable.
   - **Steps**:
     1. Generate multiple "Remember Me" tokens for different accounts.
     2. Compare patterns or structures in the tokens.
     3. Attempt to predict or forge a valid token.
   - **Indicators**:
     - Sequential or predictable tokens.
     - Tokens without cryptographic randomness.

#### 3. **Inspect Token Expiration**
   - **Scenario**: Ensure "Remember Me" tokens have a reasonable expiration policy.
   - **Steps**:
     1. Note the expiration time for the token.
     2. Test whether expired tokens are still valid.
     3. Check if expiration policies comply with application security requirements.
   - **Indicators**:
     - Tokens that never expire.
     - Inconsistent or overly long expiration durations.

#### 4. **Assess Session Management Mechanisms**
   - **Scenario**: Analyze session handling and protection against hijacking.
   - **Steps**:
     1. Log in and capture the session ID.
     2. Test session fixation and hijacking by reusing or sharing session tokens.
     3. Observe session behaviors, such as revocation and invalidation on logout.
   - **Indicators**:
     - Session tokens not invalidated on logout.
     - Reused or shared tokens remain valid.

#### 5. **Evaluate Multi-Device and Logout Behavior**
   - **Scenario**: Validate session handling across multiple devices and during logout.
   - **Steps**:
     1. Log in from multiple devices using "Remember Me."
     2. Test whether logging out from one device invalidates other sessions.
     3. Observe whether session invalidation applies universally.
   - **Indicators**:
     - Active sessions remain valid after logout from other devices.
     - Logout does not invalidate "Remember Me" tokens.

---

### Tools
- **Burp Suite**: Analyze and manipulate cookies and tokens.
- **Postman**: Test API endpoints and session behaviors.
- **OWASP ZAP**: Scan for session-related vulnerabilities.
- **Fiddler**: Inspect and modify HTTP requests and responses.

---

### Remediation
- Use strong encryption and digital signatures for "Remember Me" tokens.
- Ensure tokens are unpredictable and generated with cryptographic randomness.
- Implement reasonable expiration policies and automatic invalidation of expired tokens.
- Invalidate session tokens on logout and ensure consistency across devices.
- Avoid storing sensitive information in tokens or client-side storage.
- Regularly audit session and "Remember Me" implementations for compliance with security best practices.

---

### References
- [OWASP Testing Guide v4.0: Vulnerable Remember Me and User Sessions](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

---

### Checklist
- [ ] "Remember Me" tokens are encrypted and signed securely.
- [ ] Tokens are unique, random, and unpredictable.
- [ ] Tokens have a defined and reasonable expiration policy.
- [ ] Sessions are invalidated consistently on logout across devices.
- [ ] Sensitive data is not exposed in tokens or client-side storage.
