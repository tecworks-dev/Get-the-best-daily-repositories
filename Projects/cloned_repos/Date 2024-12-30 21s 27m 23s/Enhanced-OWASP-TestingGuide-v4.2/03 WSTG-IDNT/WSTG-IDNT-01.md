# OWASP WSTG v4.0 - WSTG-IDNT-01

## Test Name: Testing for User Enumeration and Guessable User Account

### Overview
This test aims to determine whether user accounts or related information can be enumerated, which attackers could leverage to target specific users or accounts.

---

### Objectives
- Identify vulnerabilities that allow attackers to enumerate valid usernames.
- Assess if guessable usernames are present, which could lead to brute force attacks.

---

### Test Steps

#### 1. **Enumerate Users via Login Mechanism**
   - **Scenario**: Test if the application discloses valid usernames or accounts based on login error messages.
   - **Steps**:
     1. Access the login page.
     2. Enter a valid username with an incorrect password.
     3. Note the error message.
     4. Enter an invalid username with any password.
     5. Compare the error messages.
   - **Indicators**:
     - Differing error messages (e.g., "Invalid username" vs. "Incorrect password").
     - Response time differences for valid vs. invalid usernames.

#### 2. **Enumerate Users via Forgot Password Functionality**
   - **Scenario**: Check if the password recovery process reveals valid usernames or email addresses.
   - **Steps**:
     1. Navigate to the "Forgot Password" page.
     2. Input a valid username or email.
     3. Observe the response.
     4. Input an invalid username or email and compare the response.
   - **Indicators**:
     - Specific error messages like "Email not found" or "User does not exist."
     - Account recovery instructions only sent to valid accounts.

#### 3. **Enumerate Users via Registration Mechanism**
   - **Scenario**: Validate if registration reveals existing usernames or emails.
   - **Steps**:
     1. Attempt to register with an existing username or email.
     2. Observe if the system discloses information about existing accounts.
   - **Indicators**:
     - Messages such as "Username already taken" or "Email already in use."

#### 4. **Inspect Application Responses and Metadata**
   - **Scenario**: Examine application behavior and HTTP responses for clues.
   - **Steps**:
     1. Perform actions like login attempts or form submissions.
     2. Analyze HTTP responses, headers, and codes using tools (e.g., Burp Suite, Postman).
     3. Look for hidden fields, JavaScript code, or metadata exposing user details.
   - **Indicators**:
     - Inclusion of usernames or account-related information in responses.
     - Hidden fields or comments revealing user information.

#### 5. **Leverage Default or Guessable Usernames**
   - **Scenario**: Attempt to identify if the application uses default or easily guessable usernames.
   - **Steps**:
     1. Test with common/default usernames like "admin," "user," "test," etc.
     2. Observe if any default accounts exist and their behavior upon access.
   - **Indicators**:
     - Successful logins or access using default credentials.

---

### Tools
- **Burp Suite**: Proxy and inspect HTTP responses.
- **Postman**: Test APIs and application responses.
- **Nmap**: Scan services for usernames exposed via protocols.
- **Wordlists**: Use lists for default usernames.
  
---

### Remediation
- Standardize error messages to prevent differentiation.
- Implement rate limiting to block enumeration attempts.
- Use CAPTCHAs for critical actions like login and registration.
- Avoid disclosing user details in HTTP responses or error messages.
- Regularly audit for default accounts or guessable usernames.

---

### References
- [OWASP Testing Guide v4.0: User Enumeration and Guessable User Account](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Cheat Sheet: Authentication](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

---

### Checklist
- [ ] Error messages are uniform and non-descriptive.
- [ ] Forgot Password functionality does not disclose user existence.
- [ ] Registration does not reveal existing accounts.
- [ ] No default or guessable usernames in the system.
- [ ] Application responses and metadata are sanitized.
