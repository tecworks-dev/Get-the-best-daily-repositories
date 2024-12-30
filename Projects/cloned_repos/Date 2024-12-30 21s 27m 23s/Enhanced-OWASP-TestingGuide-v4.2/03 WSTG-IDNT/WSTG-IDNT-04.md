# OWASP WSTG v4.0 - WSTG-IDNT-04

## Test Name: Testing for Insecure Password Storage

### Overview
This test evaluates whether the application securely stores user passwords in a way that protects them against unauthorized access or compromise.

---

### Objectives
- Verify that passwords are stored securely using robust encryption or hashing mechanisms.
- Identify weak or misconfigured password storage mechanisms.
- Ensure compliance with industry best practices for password storage.

---

### Test Steps

#### 1. **Inspect Application Configuration and Backend**
   - **Scenario**: Identify how passwords are stored in the application backend.
   - **Steps**:
     1. Access the application configuration files or database schema.
     2. Verify the password storage mechanism used.
   - **Indicators**:
     - Passwords stored in plain text.
     - Weak or obsolete hashing algorithms (e.g., MD5, SHA1).
     - Lack of proper salting.

#### 2. **Analyze Authentication Data in Transit**
   - **Scenario**: Confirm that passwords are not inadvertently exposed during transit.
   - **Steps**:
     1. Intercept traffic using tools like Burp Suite.
     2. Analyze login requests for unencrypted passwords.
   - **Indicators**:
     - Passwords sent in plain text or visible in logs.
     - Lack of HTTPS during sensitive operations.

#### 3. **Test for Weak Hashing Algorithms**
   - **Scenario**: Evaluate the strength of the password hashing algorithm.
   - **Steps**:
     1. Identify the hashing algorithm (e.g., through reverse engineering or inspecting configuration).
     2. Check if it meets modern cryptographic standards (e.g., bcrypt, Argon2).
   - **Indicators**:
     - Use of weak or fast algorithms that are vulnerable to brute force attacks.

#### 4. **Validate Password Salting**
   - **Scenario**: Check for the presence and uniqueness of salting in password storage.
   - **Steps**:
     1. Analyze stored password hashes.
     2. Verify if salts are unique for each user and stored securely.
   - **Indicators**:
     - Lack of salting or reuse of the same salt across accounts.

#### 5. **Examine Backup Files and Logs**
   - **Scenario**: Identify if sensitive password data is exposed in backups or logs.
   - **Steps**:
     1. Access system backups or log files.
     2. Search for instances of plaintext passwords or weakly hashed passwords.
   - **Indicators**:
     - Passwords stored in logs or backup files in an insecure manner.

---

### Tools
- **Burp Suite**: Intercept and analyze traffic.
- **Hashcat or John the Ripper**: Evaluate the strength of password hashes.
- **Database Management Tools**: Inspect stored passwords directly.
- **Strings**: Search for plaintext passwords in binaries or files.

---

### Remediation
- Always store passwords using strong, modern hashing algorithms (e.g., bcrypt, Argon2, or PBKDF2).
- Implement unique salts for each password to prevent rainbow table attacks.
- Ensure passwords are never logged in plaintext.
- Use TLS/SSL to protect password transmission over the network.
- Regularly audit backups and logs to ensure they do not contain sensitive data.
- Update legacy systems to use secure password storage mechanisms.

---

### References
- [OWASP Testing Guide v4.0: Insecure Password Storage](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-3/)

---

### Checklist
- [ ] Passwords are stored using strong and modern hashing algorithms.
- [ ] Unique salts are implemented for each password.
- [ ] Passwords are not stored in plaintext or weakly encrypted.
- [ ] Passwords are never exposed in transit, logs, or backups.
- [ ] Password storage mechanisms comply with industry best practices.
