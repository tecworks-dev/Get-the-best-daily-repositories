# OWASP WSTG v4.0 - WSTG-ATHZ-04

## Test Name: Testing for Insecure Direct Object References (IDOR)

### Overview
This test evaluates whether the application exposes direct access to objects, such as files, database entries, or resources, without proper authorization checks. IDOR vulnerabilities allow attackers to manipulate references to access unauthorized data or functionality.

---

### Objectives
- Detect insecure implementations of direct object references.
- Verify that the application enforces authorization checks for referenced objects.
- Prevent unauthorized access or manipulation of sensitive resources.

---

### Test Steps

#### 1. **Identify Object References**
   - **Scenario**: Discover endpoints or parameters referencing specific objects.
   - **Steps**:
     1. Analyze URLs, API endpoints, and request parameters for object references (e.g., `user_id`, `file_id`, `order_id`).
     2. Note patterns or predictable structures in the references.
   - **Indicators**:
     - Use of IDs, filenames, or other direct object identifiers in requests.
     - Predictable or sequential object references.

#### 2. **Manipulate Object References**
   - **Scenario**: Attempt to access unauthorized objects by modifying references.
   - **Steps**:
     1. Change object identifiers in the request to another valid value.
     2. Observe if the application grants access to the modified reference.
   - **Indicators**:
     - Successful access to objects without proper authorization.
     - No validation of user ownership or permissions for the object.

#### 3. **Test for Sensitive Object Exposure**
   - **Scenario**: Check if sensitive objects, such as administrative or private files, can be accessed.
   - **Steps**:
     1. Modify object references to target sensitive or administrative resources.
     2. Analyze server responses for exposure of sensitive data.
   - **Indicators**:
     - Unauthorized access to private or administrative objects.
     - Lack of proper authorization checks.

#### 4. **Inspect Role-Based Object Access**
   - **Scenario**: Verify that role-based permissions are enforced for object references.
   - **Steps**:
     1. Log in with accounts of varying privilege levels (e.g., user, admin).
     2. Test access to objects restricted to higher-privileged roles.
   - **Indicators**:
     - Lower-privileged accounts can access objects meant for higher roles.
     - No differentiation in access controls based on roles.

#### 5. **Analyze Error Messaging**
   - **Scenario**: Review error messages returned when accessing unauthorized objects.
   - **Steps**:
     1. Attempt to access invalid or unauthorized object references.
     2. Observe the error messages for sensitive information.
   - **Indicators**:
     - Error messages reveal object existence or structure.
     - Detailed messages aid in enumerating valid object references.

---

### Tools
- **Burp Suite**: Intercept and modify requests to test IDOR vulnerabilities.
- **Postman**: Test API endpoints for object reference manipulation.
- **OWASP ZAP**: Scan for direct object reference vulnerabilities.
- **Fiddler**: Debug and inspect HTTP requests.
- **Custom Scripts**: Automate object reference testing for predictable patterns.

---

### Remediation
- Enforce strict server-side authorization checks for all object references.
- Use indirect references, such as tokens or hashed identifiers, instead of direct object references.
- Validate ownership and permissions for each object request.
- Avoid revealing object details in error messages.
- Regularly test and audit endpoints for IDOR vulnerabilities.

---

### References
- [OWASP Testing Guide v4.0: Insecure Direct Object References (IDOR)](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)

---

### Checklist
- [ ] Object references are indirect and not easily manipulated.
- [ ] Authorization checks are enforced server-side for all objects.
- [ ] Role-based permissions are correctly implemented for object access.
- [ ] Error messages do not reveal sensitive information about objects.
- [ ] Endpoints are regularly tested and audited for IDOR vulnerabilities.
