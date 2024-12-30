# OWASP WSTG v4.0 - WSTG-ATHN-02

## Test Name: Testing for Default Credentials

### Overview
This test identifies the presence of default credentials in the application or its associated systems. Default credentials can be exploited by attackers to gain unauthorized access.

---

### Objectives
- Identify accounts that use default credentials.
- Ensure that systems enforce the change of default credentials during initial setup.
- Detect vulnerabilities arising from poorly managed credentials.

---

### Test Steps

#### 1. **Identify Default Credentials**
   - **Scenario**: Check if the application or underlying systems use default credentials.
   - **Steps**:
     1. Review application and system documentation for default credentials.
     2. Search online for publicly available default credentials related to the application or system.
     3. Attempt to log in with known default credentials (e.g., "admin:admin," "user:password").
   - **Indicators**:
     - Successful login with default credentials.
     - Presence of unused or hidden default accounts.

#### 2. **Test for Hardcoded Credentials**
   - **Scenario**: Analyze application components for hardcoded credentials.
   - **Steps**:
     1. Inspect application source code, configuration files, and scripts for embedded credentials.
     2. Use tools to decompile or reverse engineer binaries to identify hardcoded credentials.
   - **Indicators**:
     - Credentials stored in plaintext or hardcoded within code.
     - Use of predictable credentials.

#### 3. **Inspect Administrative Interfaces**
   - **Scenario**: Assess the security of administrative interfaces for default or weak credentials.
   - **Steps**:
     1. Access administrative login pages or panels.
     2. Attempt to use common default credentials.
   - **Indicators**:
     - Default credentials allow administrative access.
     - Weak credentials used by default in admin interfaces.

#### 4. **Scan and Test Third-Party Components**
   - **Scenario**: Evaluate third-party software or systems for default credentials.
   - **Steps**:
     1. Identify third-party components integrated into the application.
     2. Test each component with known or default credentials.
   - **Indicators**:
     - Third-party components accessible with default credentials.
     - Lack of enforced credential updates for third-party tools.

#### 5. **Check Documentation and Deployment Scripts**
   - **Scenario**: Verify whether deployment scripts or documentation inadvertently expose default credentials.
   - **Steps**:
     1. Review deployment scripts and logs for default credentials.
     2. Analyze documentation for any mention of default credentials.
   - **Indicators**:
     - Default credentials included in deployment scripts or setup guides.

---

### Tools
- **Hydra or Medusa**: Automate credential brute forcing.
- **Strings**: Extract embedded strings from binaries or files.
- **Burp Suite**: Test for hardcoded credentials in requests.
- **Nmap**: Identify services that may use default credentials.
- **Search Engines**: Look up default credentials for specific software or systems.

---

### Remediation
- Remove or disable default accounts and credentials.
- Enforce mandatory credential changes during initial setup.
- Use strong and unique passwords for all accounts, including administrative ones.
- Audit code and configurations to remove hardcoded credentials.
- Regularly review and update documentation to ensure no default credentials are exposed.
- Monitor and manage third-party components to ensure secure credential practices.

---

### References
- [OWASP Testing Guide v4.0: Default Credentials](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Credential Stuffing Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Credential_Stuffing_Prevention_Cheat_Sheet.html)

---

### Checklist
- [ ] Default credentials are removed or disabled.
- [ ] Hardcoded credentials are not present in code or configurations.
- [ ] Administrative interfaces require strong and unique credentials.
- [ ] Third-party components do not use default credentials.
- [ ] Deployment scripts and documentation do not expose credentials.
