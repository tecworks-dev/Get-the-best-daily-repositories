# OWASP WSTG v4.0 - WSTG-IDNT-02

## Test Name: Testing for Account Lockout

### Overview
This test assesses whether the application implements mechanisms to prevent brute force attacks by locking accounts after a specified number of failed login attempts.

---

### Objectives
- Verify if account lockout is implemented.
- Ensure lockout mechanisms are effective and user-friendly.
- Identify potential bypasses or weaknesses in the account lockout mechanism.

---

### Test Steps

#### 1. **Test for Acc ount Lockout Implementation**
   - **Scenario**: Check if the application locks accounts after repeated failed login attempts.
   - **Steps**:
     1. Identify a valid username.
     2. Attempt multiple incorrect logins using the username.
     3. Observe the system response after each attempt.
   - **Indicators**:
     - Account is locked after a predefined number of failed attempts.
     - User receives a lockout notification.
     - Further login attempts result in a consistent lockout message.

#### 2. **Test Lockout Duration and Reset Mechanisms**
   - **Scenario**: Validate the lockout duration and any mechanisms to unlock the account.
   - **Steps**:
     1. Lock the account using failed login attempts.
     2. Wait for the lockout period to expire, if applicable.
     3. Test any account recovery options, such as "Forgot Password."
     4. Attempt to unlock using administrative interfaces, if possible.
   - **Indicators**:
     - Lockout duration matches documented policy.
     - Recovery or unlock mechanisms are secure and not easily exploitable.

#### 3. **Attempt to Bypass Lockout Mechanisms**
   - **Scenario**: Investigate if the lockout can be bypassed.
   - **Steps**:
     1. Attempt logins from multiple IP addresses or devices.
     2. Test whether valid login attempts reset the lockout counter.
     3. Observe if any other authentication mechanisms bypass the lockout.
   - **Indicators**:
     - Lockout applies consistently across IPs and devices.
     - Valid logins do not reset the lockout counter for unrelated users.

#### 4. **Inspect Application Responses**
   - **Scenario**: Analyze HTTP responses during lockout testing.
   - **Steps**:
     1. Use tools like Burp Suite to monitor server responses.
     2. Check for detailed error messages or headers indicating lockout state.
     3. Look for inconsistencies in server behavior.
   - **Indicators**:
     - Responses confirm lockout without exposing unnecessary details.
     - No sensitive information is leaked in headers or error messages.

---

### Tools
- **Burp Suite**: Proxy for analyzing HTTP responses.
- **Hydra or Medusa**: Test brute force capabilities.
- **Browser Developer Tools**: Inspect client-side behaviors.
- **Wordlists**: Test common passwords and usernames.

---

### Remediation
- Implement account lockout after a predefined number of failed login attempts.
- Configure lockout duration to balance security and user experience.
- Ensure lockout applies across all devices and IPs.
- Monitor for lockout bypass attempts and implement logging mechanisms.
- Provide secure and user-friendly recovery options.

---

### References
- [OWASP Testing Guide v4.0: Account Lockout](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Cheat Sheet: Authentication](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

---

### Checklist
- [ ] Lockout occurs after a predefined number of failed login attempts.
- [ ] Lockout duration is documented and implemented securely.
- [ ] Lockout applies across devices and IPs.
- [ ] No sensitive information is disclosed during lockout.
- [ ] Recovery mechanisms are secure and effective.
