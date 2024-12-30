# OWASP WSTG v4.0 - WSTG-IDNT-03

## Test Name: Testing for Weak or Guessable Passwords

### Overview
This test identifies whether users or the system employ weak or guessable passwords that can be exploited by attackers through brute force or dictionary attacks.

---

### Objectives
- Ensure the application enforces strong password policies.
- Detect any use of weak, default, or guessable passwords.
- Identify vulnerabilities in password-related features.

---

### Test Steps

#### 1. **Review Password Policy**
   - **Scenario**: Check if the application enforces a strong password policy during user registration or password updates.
   - **Steps**:
     1. Attempt to create or update a password using weak values (e.g., "password123," "12345678").
     2. Observe if the application rejects weak passwords.
   - **Indicators**:
     - Password policy enforces minimum length, complexity, and avoids common passwords.
     - Users are informed of policy requirements clearly.

#### 2. **Test for Default or Guessable Passwords**
   - **Scenario**: Validate whether default or common passwords are being used.
   - **Steps**:
     1. Identify valid usernames or accounts.
     2. Attempt to log in using default or common passwords (e.g., "admin:admin," "user:password").
   - **Indicators**:
     - Successful login using default or guessable credentials.
     - Accounts lack mandatory password change on first login.

#### 3. **Conduct Brute Force Testing**
   - **Scenario**: Attempt to brute force user passwords using automation tools.
   - **Steps**:
     1. Use tools like Hydra, Medusa, or Burp Suite to test a range of passwords for a known username.
     2. Monitor application responses to identify successful logins or weak protections.
   - **Indicators**:
     - Application allows unlimited password attempts without lockout.
     - Lack of monitoring or rate-limiting for brute force attacks.

#### 4. **Analyze Password Reset Mechanism**
   - **Scenario**: Check the security of the password reset functionality.
   - **Steps**:
     1. Trigger the "Forgot Password" flow.
     2. Ensure reset links or tokens are unique, expire after a short period, and require strong verification.
   - **Indicators**:
     - Tokens are predictable or valid for extended periods.
     - Reset flows allow weak or previously compromised passwords.

#### 5. **Inspect Configuration Files and Back-End Systems**
   - **Scenario**: Ensure sensitive systems do not store passwords in plain text or use weak encryption.
   - **Steps**:
     1. Check for password storage mechanisms in configuration files or database schemas.
     2. Confirm encryption standards and salting are used.
   - **Indicators**:
     - Passwords are stored in plain text or use weak hash algorithms like MD5.

---

### Tools
- **Hydra or Medusa**: Automate brute force testing.
- **Burp Suite**: Analyze application responses and password flows.
- **John the Ripper**: Test password strength offline.
- **OWASP ZAP**: Identify weak password mechanisms during scans.
- **Wordlists**: Use lists of common and default passwords.

---

### Remediation
- Enforce strong password policies, including minimum length, complexity, and avoidance of common passwords.
- Mandate password changes for default accounts upon first login.
- Implement account lockout and rate-limiting mechanisms.
- Use secure hashing algorithms like bcrypt, PBKDF2, or Argon2 with proper salting.
- Regularly audit and update password reset flows to ensure secure token usage.

---

### References
- [OWASP Testing Guide v4.0: Weak or Guessable Passwords](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

---

### Checklist
- [ ] Password policy enforces strong complexity and length requirements.
- [ ] Default or guessable passwords are not used.
- [ ] Brute force protection mechanisms are in place.
- [ ] Password reset flows are secure and enforce strong new passwords.
- [ ] Passwords are stored securely using proper hashing algorithms.
