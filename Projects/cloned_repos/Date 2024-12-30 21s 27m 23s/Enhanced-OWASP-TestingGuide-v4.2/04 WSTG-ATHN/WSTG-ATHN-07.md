# OWASP WSTG v4.0 - WSTG-ATHN-07

## Test Name: Testing for Weak Password Policy

### Overview
This test evaluates the application's password policy to ensure it enforces strong and secure passwords that minimize the risk of unauthorized access.

---

### Objectives
- Verify that the application enforces a robust password policy during user registration and password changes.
- Detect weak password requirements that could lead to account compromise.
- Assess compliance with industry standards for password security.

---

### Test Steps

#### 1. **Analyze Password Requirements**
   - **Scenario**: Assess the strength of password policies enforced during registration or password updates.
   - **Steps**:
     1. Attempt to create a password using weak values (e.g., "password123", "12345678").
     2. Observe if the application rejects weak passwords.
   - **Indicators**:
     - Password policies allow weak or commonly used passwords.
     - Absence of checks for complexity, length, or reuse.

#### 2. **Test for Minimum Length and Complexity**
   - **Scenario**: Ensure that the application enforces adequate length and complexity requirements.
   - **Steps**:
     1. Attempt to use short passwords (e.g., "12345").
     2. Test passwords lacking complexity (e.g., no uppercase, lowercase, numbers, or special characters).
   - **Indicators**:
     - Acceptance of passwords shorter than 8 characters.
     - No enforcement of complexity rules.

#### 3. **Check for Password Reuse**
   - **Scenario**: Verify if the application prevents reuse of previously used passwords.
   - **Steps**:
     1. Change the password to a new value.
     2. Attempt to reuse the old password.
   - **Indicators**:
     - Acceptance of previously used passwords.

#### 4. **Evaluate Error Messaging**
   - **Scenario**: Review how the application communicates password policy requirements.
   - **Steps**:
     1. Enter a weak password during registration or change.
     2. Observe the error messages displayed.
   - **Indicators**:
     - Vague or unclear error messages regarding password strength.

#### 5. **Inspect Temporary Password Mechanisms**
   - **Scenario**: Assess the security of temporary passwords used for account recovery.
   - **Steps**:
     1. Trigger the password recovery process.
     2. Inspect the format and strength of temporary passwords.
   - **Indicators**:
     - Predictable or weak temporary passwords.
     - Temporary passwords not expiring after use.

---

### Tools
- **Burp Suite**: Intercept and modify password-related requests.
- **Postman**: Test API endpoints for password strength enforcement.
- **Custom Wordlists**: Validate common weak passwords.
- **John the Ripper**: Test password hashes offline (if accessible).
- **OWASP ZAP**: Scan for password-related vulnerabilities.

---

### Remediation
- Enforce strong password policies with a minimum length of 12 characters and a mix of uppercase, lowercase, numbers, and special characters.
- Implement checks against commonly used or breached passwords.
- Prevent reuse of previously used passwords.
- Use clear and detailed error messages to guide users.
- Ensure temporary passwords are strong, random, and expire after one use.
- Regularly review and update password policies to align with industry standards (e.g., NIST).

---

### References
- [OWASP Testing Guide v4.0: Weak Password Policy](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-3/)
- [Have I Been Pwned? Passwords](https://haveibeenpwned.com/Passwords)

---

### Checklist
- [ ] Password policy enforces minimum length and complexity requirements.
- [ ] Commonly used and breached passwords are rejected.
- [ ] Previously used passwords cannot be reused.
- [ ] Temporary passwords are strong, random, and expire after use.
- [ ] Error messages clearly communicate password policy requirements.
