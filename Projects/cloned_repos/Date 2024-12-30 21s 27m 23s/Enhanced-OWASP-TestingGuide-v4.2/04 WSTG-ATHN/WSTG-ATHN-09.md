# OWASP WSTG v4.0 - WSTG-ATHN-09

## Test Name: Testing for Weak Password Change or Reset Functionality

### Overview
This test evaluates the robustness of the password change and reset functionality to ensure that attackers cannot exploit weaknesses to compromise user accounts.

---

### Objectives
- Verify the security of the password change and reset process.
- Ensure that proper validation and authentication mechanisms are in place.
- Detect vulnerabilities that could allow unauthorized access to accounts.

---

### Test Steps

#### 1. **Analyze Password Change Workflow**
   - **Scenario**: Verify that authenticated users can securely change their passwords.
   - **Steps**:
     1. Log in to a user account.
     2. Navigate to the password change section.
     3. Test the password change process with valid and invalid inputs.
   - **Indicators**:
     - Lack of current password validation.
     - Acceptance of weak or commonly used passwords.
     - No confirmation of successful password change.

#### 2. **Inspect Password Reset Workflow**
   - **Scenario**: Assess the security of the password reset functionality.
   - **Steps**:
     1. Initiate the password reset process using a valid email or username.
     2. Capture and analyze the reset link or token.
     3. Test if expired or reused tokens are invalidated.
   - **Indicators**:
     - Reset links or tokens do not expire or are predictable.
     - Lack of identity verification during the reset process.

#### 3. **Test for Brute Force on Password Reset Tokens**
   - **Scenario**: Check if the application protects against brute force attacks on reset tokens.
   - **Steps**:
     1. Generate multiple password reset requests.
     2. Attempt to brute force the token using automation tools.
   - **Indicators**:
     - Tokens are short, predictable, or not rate-limited.
     - Application does not invalidate tokens after multiple failed attempts.

#### 4. **Evaluate Error Messaging**
   - **Scenario**: Assess if error messages reveal sensitive information.
   - **Steps**:
     1. Initiate password reset with invalid or unregistered email addresses.
     2. Observe error messages displayed by the application.
   - **Indicators**:
     - Messages confirm account existence (e.g., "Email not found").
     - Detailed errors aid attackers in crafting targeted attacks.

#### 5. **Inspect Post-Reset Behavior**
   - **Scenario**: Ensure proper session handling and token invalidation post-reset.
   - **Steps**:
     1. Complete the password reset process.
     2. Test if previous sessions or tokens remain valid.
   - **Indicators**:
     - Active sessions are not invalidated after a password reset.
     - Old tokens can still be used after a new password is set.

---

### Tools
- **Burp Suite**: Intercept and analyze password reset tokens.
- **Postman**: Test password reset API endpoints.
- **Hydra or Medusa**: Attempt brute force attacks on reset tokens.
- **Fiddler**: Debug and inspect HTTP requests during password workflows.
- **JWT.io**: Decode and inspect JSON Web Tokens (if used).

---

### Remediation
- Require current password validation for password changes.
- Use strong, unpredictable, and short-lived reset tokens.
- Implement rate limiting and CAPTCHA for password reset requests.
- Ensure error messages do not disclose sensitive information.
- Invalidate all active sessions and reset tokens after password changes.
- Regularly test and audit password workflows for vulnerabilities.

---

### References
- [OWASP Testing Guide v4.0: Weak Password Change or Reset Functionality](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)

---

### Checklist
- [ ] Current password validation is required for changes.
- [ ] Password reset tokens are strong, unpredictable, and expire after use.
- [ ] Error messages do not disclose sensitive information.
- [ ] Rate limiting and CAPTCHA are implemented for password resets.
- [ ] All active sessions and reset tokens are invalidated after password changes.