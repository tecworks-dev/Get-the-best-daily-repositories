# OWASP WSTG v4.0 - WSTG-ATHZ-02

## Test Name: Testing for Bypassing Authorization Schema

### Overview
This test evaluates whether an application’s authorization mechanisms can be bypassed, allowing unauthorized users to access restricted resources or perform unauthorized actions.

---

### Objectives
- Identify flaws in the authorization logic.
- Detect vulnerabilities that allow unauthorized access or privilege escalation.
- Ensure proper role-based and resource-based access controls are enforced.

---

### Test Steps

#### 1. **Test for Direct Access to Restricted Resources**
   - **Scenario**: Verify if restricted resources can be accessed without proper authorization.
   - **Steps**:
     1. Identify URLs, endpoints, or resources that should be restricted.
     2. Attempt to access these resources directly without authentication or with lower privileges.
   - **Indicators**:
     - Access is granted to restricted resources without proper authorization.
     - No redirection to a login or access denied page.

#### 2. **Inspect Parameter-Based Access Controls**
   - **Scenario**: Check if authorization depends solely on client-side parameters.
   - **Steps**:
     1. Intercept requests and modify parameters such as `user_id`, `role`, or `access_level`.
     2. Observe the application’s response.
   - **Indicators**:
     - Unauthorized actions are allowed by manipulating parameters.
     - No server-side validation of parameter values.

#### 3. **Test for Horizontal Privilege Escalation**
   - **Scenario**: Check if users can access data or resources belonging to other users.
   - **Steps**:
     1. Log in as a normal user.
     2. Modify requests to target data or resources belonging to other users (e.g., change `user_id`).
   - **Indicators**:
     - Ability to view, modify, or delete other users’ data.
     - No checks on resource ownership.

#### 4. **Test for Vertical Privilege Escalation**
   - **Scenario**: Verify if lower-privileged users can perform administrative actions.
   - **Steps**:
     1. Log in with a lower-privileged account.
     2. Attempt actions or access resources intended for higher-privileged users (e.g., administrators).
   - **Indicators**:
     - Successful execution of administrative actions.
     - Access to resources meant for higher-privileged roles.

#### 5. **Evaluate Role-Based Access Controls (RBAC)**
   - **Scenario**: Ensure roles are correctly enforced for access control.
   - **Steps**:
     1. Test actions or resources specific to various roles.
     2. Verify that users with inappropriate roles cannot perform restricted actions.
   - **Indicators**:
     - Misconfigured roles grant excessive permissions.
     - Lack of segregation of duties.

#### 6. **Inspect Access Tokens or Session Data**
   - **Scenario**: Analyze the security of access tokens or session data used for authorization.
   - **Steps**:
     1. Capture and decode tokens or session cookies.
     2. Test if tampering with these values affects authorization.
   - **Indicators**:
     - Tokens lack integrity checks or are not cryptographically signed.
     - Authorization depends solely on client-side data.

---

### Tools
- **Burp Suite**: Intercept and modify requests to test authorization logic.
- **Postman**: Test API endpoints for bypass scenarios.
- **Fiddler**: Debug and inspect HTTP requests.
- **OWASP ZAP**: Automate tests for authorization vulnerabilities.
- **JWT.io**: Decode and manipulate JSON Web Tokens (if used).

---

### Remediation
- Enforce strict server-side access control mechanisms.
- Validate all user actions against their roles and permissions.
- Implement proper integrity checks and cryptographic signing for tokens.
- Regularly audit access controls and roles for misconfigurations.
- Log and monitor unauthorized access attempts to detect potential attacks.

---

### References
- [OWASP Testing Guide v4.0: Bypassing Authorization Schema](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)

---

### Checklist
- [ ] Direct access to restricted resources is denied without proper authorization.
- [ ] Server-side validation is enforced for all parameters.
- [ ] Horizontal privilege escalation is not possible.
- [ ] Vertical privilege escalation is not possible.
- [ ] Role-based access controls are implemented and tested.
- [ ] Tokens and session data are secured with integrity checks.