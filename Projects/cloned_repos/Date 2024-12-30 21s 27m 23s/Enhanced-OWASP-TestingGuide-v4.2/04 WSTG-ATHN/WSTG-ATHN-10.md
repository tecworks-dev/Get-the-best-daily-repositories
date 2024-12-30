# OWASP WSTG v4.0 - WSTG-ATHN-10

## Test Name: Testing for Insufficient Session Expiration

### Overview
This test ensures that user sessions are appropriately expired after logout, inactivity, or a predefined duration to prevent unauthorized access.

---

### Objectives
- Verify that user sessions are terminated after a defined period of inactivity.
- Ensure sessions are invalidated on user logout.
- Detect vulnerabilities that allow continued access after session expiration.

---

### Test Steps

#### 1. **Test for Session Expiration on Logout**
   - **Scenario**: Verify that active sessions are terminated upon user logout.
   - **Steps**:
     1. Log in to the application.
     2. Capture the session ID or token.
     3. Log out and attempt to use the captured session ID/token to access the application.
   - **Indicators**:
     - Session remains valid after logout.
     - No server-side invalidation of session tokens.

#### 2. **Test for Idle Session Timeout**
   - **Scenario**: Ensure the application invalidates sessions after a period of inactivity.
   - **Steps**:
     1. Log in and perform actions to initiate a session.
     2. Leave the session idle for the defined timeout period.
     3. Attempt to perform an action after the timeout period.
   - **Indicators**:
     - Session remains valid despite exceeding the inactivity timeout.
     - Application does not enforce session expiration.

#### 3. **Test for Absolute Session Timeout**
   - **Scenario**: Verify that sessions expire after a maximum lifespan, regardless of activity.
   - **Steps**:
     1. Log in and perform actions periodically to keep the session active.
     2. Wait for the absolute timeout period to elapse.
     3. Attempt to perform an action after the absolute timeout period.
   - **Indicators**:
     - Session does not expire after the absolute timeout period.
     - Lack of enforcement of a maximum session duration.

#### 4. **Inspect Session Management Mechanisms**
   - **Scenario**: Evaluate how session expiration is managed on the client and server sides.
   - **Steps**:
     1. Analyze cookies or tokens to check for expiration attributes.
     2. Intercept and inspect HTTP responses during login and logout flows.
   - **Indicators**:
     - Session expiration relies solely on client-side mechanisms (e.g., JavaScript).
     - Missing or incorrect `expires` and `max-age` attributes in cookies.

#### 5. **Test Multi-Device Session Behavior**
   - **Scenario**: Assess session expiration across multiple devices.
   - **Steps**:
     1. Log in to the same account on two devices.
     2. Log out on one device and attempt to access the application on the other.
   - **Indicators**:
     - Sessions remain valid on other devices after logout.
     - No synchronization of session invalidation across devices.

---

### Tools
- **Burp Suite**: Capture and manipulate session tokens.
- **Postman**: Test session expiration for APIs.
- **Fiddler**: Debug and inspect session-related requests.
- **Browser Developer Tools**: Analyze cookies and storage mechanisms.
- **Wireshark**: Monitor session behavior at the network level.

---

### Remediation
- Ensure all sessions are invalidated on user logout.
- Implement server-side enforcement of idle and absolute session timeouts.
- Synchronize session invalidation across all devices and endpoints.
- Use secure cookies with proper `expires` or `max-age` attributes.
- Regularly audit session management mechanisms for compliance with security standards.

---

### References
- [OWASP Testing Guide v4.0: Insufficient Session Expiration](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-3/)

---

### Checklist
- [ ] Sessions are invalidated upon user logout.
- [ ] Idle sessions expire after a predefined period of inactivity.
- [ ] Sessions have an absolute expiration time.
- [ ] Session expiration is enforced server-side.
- [ ] Multi-device session synchronization is implemented.
