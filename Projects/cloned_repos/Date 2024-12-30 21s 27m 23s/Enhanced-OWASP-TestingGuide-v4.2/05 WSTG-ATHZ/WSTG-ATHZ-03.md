# OWASP WSTG v4.0 - WSTG-ATHZ-03

## Test Name: Testing for Privilege Escalation

### Overview
This test identifies vulnerabilities that allow a user to elevate their privileges beyond what they are authorized for, potentially gaining administrative or other unauthorized access.

---

### Objectives
- Verify that the application enforces proper access controls to prevent privilege escalation.
- Detect vertical or horizontal privilege escalation vulnerabilities.
- Ensure roles and permissions are strictly implemented and validated.

---

### Test Steps

#### 1. **Test for Horizontal Privilege Escalation**
   - **Scenario**: Check if users can access resources or perform actions intended for other users.
   - **Steps**:
     1. Log in as a standard user.
     2. Modify requests to access resources or data belonging to other users (e.g., change `user_id` or `account_id` in parameters).
   - **Indicators**:
     - Access to other users’ resources or data without authorization.
     - No server-side validation of resource ownership.

#### 2. **Test for Vertical Privilege Escalation**
   - **Scenario**: Verify if lower-privileged users can perform administrative actions.
   - **Steps**:
     1. Log in with a lower-privileged account.
     2. Attempt to access administrative functions or modify higher-privileged resources.
   - **Indicators**:
     - Ability to perform actions or access resources meant for administrators.
     - Lack of role-based validation.

#### 3. **Inspect Role-Based Access Control (RBAC)**
   - **Scenario**: Evaluate the implementation of RBAC mechanisms.
   - **Steps**:
     1. Test actions available to various roles (e.g., user, moderator, admin).
     2. Verify if roles are properly enforced for each action or resource.
   - **Indicators**:
     - Misconfigured roles granting excessive permissions.
     - Lack of segregation of duties between roles.

#### 4. **Analyze Multi-Step Processes**
   - **Scenario**: Check if multi-step processes can be manipulated to escalate privileges.
   - **Steps**:
     1. Identify workflows with multiple steps (e.g., account upgrades, admin approvals).
     2. Attempt to bypass intermediate steps or approvals.
   - **Indicators**:
     - Ability to complete restricted actions without required approvals.
     - Lack of validation in multi-step processes.

#### 5. **Inspect Client-Side Role Validation**
   - **Scenario**: Test if roles or permissions are enforced client-side instead of server-side.
   - **Steps**:
     1. Intercept and modify requests to alter role or permission attributes.
     2. Test if altered attributes grant unauthorized access.
   - **Indicators**:
     - Role or permission checks performed only client-side.
     - Unauthorized actions allowed after modifying client-side attributes.

---

### Tools
- **Burp Suite**: Intercept and manipulate requests for privilege escalation tests.
- **Postman**: Test API endpoints for role and permission validation.
- **OWASP ZAP**: Scan for access control vulnerabilities.
- **Fiddler**: Debug and analyze HTTP requests.
- **JWT.io**: Decode and manipulate JSON Web Tokens (if used).

---

### Remediation
- Enforce strict server-side validation for all roles and permissions.
- Implement proper role-based access control mechanisms and test them regularly.
- Validate ownership and permissions for each user action.
- Use secure multi-step processes with validation at each step.
- Avoid relying solely on client-side role or permission enforcement.
- Log and monitor privilege escalation attempts to detect potential abuse.

---

### References
- [OWASP Testing Guide v4.0: Privilege Escalation](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [OWASP Top 10: Broken Access Control](https://owasp.org/Top10/A01_2021-Broken_Access_Control/)

---

### Checklist
- [ ] Horizontal privilege escalation is not possible.
- [ ] Vertical privilege escalation is not possible.
- [ ] Role-based access control is implemented and tested.
- [ ] Multi-step processes are secure and validated.
- [ ] Role and permission checks are enforced server-side.
