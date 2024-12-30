# OWASP WSTG v4.0 - WSTG-ATHN-03

## Test Name: Testing for Weak Lock Out Mechanism

### Overview
This test evaluates the lockout mechanisms implemented by the application to protect against brute force attacks. A weak or improperly configured lockout mechanism can leave user accounts vulnerable to unauthorized access.

---

### Objectives
- Verify that the application implements a robust lockout mechanism after a predefined number of failed login attempts.
- Detect potential bypasses or weaknesses in the lockout mechanism.
- Assess the user experience and security balance in the lockout implementation.

---

### Test Steps

#### 1. **Test Lockout After Multiple Failed Login Attempts**
   - **Scenario**: Confirm that the application locks user accounts after repeated failed login attempts.
   - **Steps**:
     1. Attempt to log in with invalid credentials multiple times.
     2. Observe the system’s response after a predefined number of failed attempts.
   - **Indicators**:
     - Account is locked or temporarily disabled.
     - Consistent lockout message displayed to the user.

#### 2. **Check Lockout Reset Mechanisms**
   - **Scenario**: Validate the lockout duration and reset mechanisms.
   - **Steps**:
     1. Trigger the lockout by exceeding the allowed failed login attempts.
     2. Wait for the lockout period to expire, if applicable.
     3. Test if the lockout is reset after a successful login.
   - **Indicators**:
     - Lockout period matches application security policy.
     - Mechanisms for lockout reset are secure and cannot be bypassed.

#### 3. **Test for Lockout Bypass Techniques**
   - **Scenario**: Identify methods to bypass the lockout mechanism.
   - **Steps**:
     1. Attempt login from multiple IP addresses or devices.
     2. Test whether valid login attempts reset the lockout.
     3. Analyze if other authentication methods (e.g., SSO, OAuth) bypass lockout.
   - **Indicators**:
     - Lockout is enforced across IPs and devices.
     - Lockout applies universally to all authentication methods.

#### 4. **Inspect Application Responses During Lockout**
   - **Scenario**: Analyze how the application communicates the lockout status.
   - **Steps**:
     1. Trigger a lockout by exceeding the allowed failed attempts.
     2. Intercept and analyze HTTP responses, headers, and codes.
   - **Indicators**:
     - Responses confirm lockout without exposing sensitive information.
     - No discrepancies or detailed error messages that aid attackers.

#### 5. **Evaluate Multi-Account Lockout Behavior**
   - **Scenario**: Assess if attempts on multiple accounts trigger lockouts inconsistently.
   - **Steps**:
     1. Attempt failed logins on multiple user accounts in rapid succession.
     2. Observe lockout behavior for each account.
   - **Indicators**:
     - Lockout is applied consistently across all accounts.
     - No evidence of shared thresholds that compromise individual accounts.

---

### Tools
- **Burp Suite**: Intercept and analyze login attempts.
- **Hydra or Medusa**: Test brute force scenarios.
- **Postman**: Test and verify lockout behaviors via API endpoints.
- **Wireshark**: Monitor traffic for lockout-related responses.

---

### Remediation
- Implement account lockout after a predefined number of failed attempts.
- Use progressive delays or CAPTCHA challenges instead of full account lockout where appropriate.
- Enforce consistent lockout policies across all devices, IPs, and authentication methods.
- Log and monitor lockout events to identify potential abuse.
- Ensure lockout messages do not disclose sensitive details or differentiate between valid and invalid accounts.
- Regularly test and audit lockout mechanisms for potential bypass techniques.

---

### References
- [OWASP Testing Guide v4.0: Weak Lock Out Mechanism](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-3/)

---

### Checklist
- [ ] Lockout occurs after a predefined number of failed login attempts.
- [ ] Lockout duration is reasonable and consistent.
- [ ] Lockout applies across IPs, devices, and authentication methods.
- [ ] Lockout mechanisms cannot be bypassed.
- [ ] Lockout messages do not reveal sensitive information.
- [ ] Lockout events are logged and monitored for abuse.
