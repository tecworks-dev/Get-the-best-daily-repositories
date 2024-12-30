# OWASP WSTG v4.0 - WSTG-ATHN-08

## Test Name: Testing for Weak Security Question/Answer

### Overview
This test evaluates the security of the question/answer mechanism used for account recovery. Weak security questions can be easily guessed or researched, leading to unauthorized account access.

---

### Objectives
- Verify that security questions and answers are robust and not easily guessable.
- Assess the implementation of the security question mechanism for weaknesses.
- Ensure that the mechanism does not expose sensitive information.

---

### Test Steps

#### 1. **Review Available Security Questions**
   - **Scenario**: Examine the default security questions provided by the application.
   - **Steps**:
     1. Attempt to create or update an account with security questions enabled.
     2. Review the list of available questions.
   - **Indicators**:
     - Questions based on publicly available information (e.g., "What is your mother’s maiden name?").
     - Use of generic or common questions.

#### 2. **Test for Predictable Answers**
   - **Scenario**: Assess if security answers can be easily guessed or researched.
   - **Steps**:
     1. Analyze the answers to common questions (e.g., "What is your favorite color?").
     2. Test if brute force techniques can guess answers.
   - **Indicators**:
     - Answers with limited variability (e.g., "Yes/No" or single-word answers).
     - Publicly known or easily researchable answers.

#### 3. **Check for Multiple Question Support**
   - **Scenario**: Verify if the application supports multiple security questions to enhance security.
   - **Steps**:
     1. Attempt to set up or reset an account using multiple questions.
     2. Observe whether multiple answers are required.
   - **Indicators**:
     - Application relies on a single question and answer.
     - No option to combine questions for additional security.

#### 4. **Test for Secure Storage of Answers**
   - **Scenario**: Confirm that security answers are stored securely in the backend.
   - **Steps**:
     1. Capture traffic or inspect server responses during security question setup.
     2. Verify if answers are encrypted or hashed.
   - **Indicators**:
     - Answers stored in plaintext or without hashing.
     - Lack of proper encryption or security mechanisms.

#### 5. **Analyze the Recovery Flow**
   - **Scenario**: Evaluate the recovery process for potential weaknesses.
   - **Steps**:
     1. Initiate the account recovery process.
     2. Observe how security questions are presented and validated.
   - **Indicators**:
     - No rate limiting for answering security questions.
     - Error messages reveal details about valid answers or accounts.

---

### Tools
- **Burp Suite**: Intercept and analyze traffic during the recovery process.
- **Fiddler**: Debug and inspect HTTP requests related to security questions.
- **Custom Wordlists**: Test common answers and brute force possibilities.
- **Wireshark**: Monitor traffic for plaintext answers.

---

### Remediation
- Use customizable security questions with user-defined questions and answers.
- Ensure questions and answers are unique and not based on public information.
- Hash and encrypt answers securely in the backend.
- Implement rate limiting and CAPTCHA to prevent brute force attacks.
- Use multi-factor authentication (MFA) for additional security during account recovery.
- Regularly review and update security question policies.

---

### References
- [OWASP Testing Guide v4.0: Weak Security Question/Answer](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Sensitive Data Exposure Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Sensitive_Data_Exposure_Cheat_Sheet.html)

---

### Checklist
- [ ] Security questions are customizable and unique to the user.
- [ ] Answers are securely hashed and encrypted.
- [ ] Recovery flows implement rate limiting and CAPTCHA mechanisms.
- [ ] No sensitive data is exposed during the recovery process.
- [ ] Security policies are regularly reviewed and updated.