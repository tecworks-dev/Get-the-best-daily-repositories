# WSTG-BUSL-02: Test for Integrity Checks

## Summary

Testing for integrity checks ensures that the application validates the integrity of data during transmission and storage. Weaknesses in data integrity validation can result in unauthorized modifications, tampering, or corruption of critical data.

## Objectives

1. Identify data flows and storage points where integrity validation is required.
2. Ensure data is protected against unauthorized modification.
3. Validate the implementation of integrity checks.

## How to Test

### 1. Understand Data Flows and Storage

- Review the application's workflows and identify:
  - Data transmitted between components (e.g., client-server, APIs).
  - Critical data stored in databases, files, or other storage mechanisms.
  - Sensitive data elements requiring integrity protection (e.g., user credentials, transaction data).

### 2. Test Data Integrity During Transmission

- Intercept and modify data in transit:
  - Use tools to manipulate HTTP/HTTPS traffic, API requests, or other communication channels.
  - Observe if the application detects and prevents tampered data.
- Check for cryptographic integrity mechanisms, such as:
  - Message Authentication Codes (MACs)
  - Digital signatures

#### Tools:
- Burp Suite
- Mitmproxy
- Postman
- Custom scripts

### 3. Test Data Integrity at Rest

- Access stored data and attempt unauthorized modifications:
  - Alter database entries or configuration files.
  - Modify sensitive files on the filesystem.
- Verify if the application detects and handles tampering attempts.
- Check for encryption and hashing mechanisms applied to stored data.

#### Tools:
- Database inspection tools
- File editors
- Hex editors

### 4. Validate Use of Cryptographic Integrity Mechanisms

- Ensure cryptographic mechanisms are implemented:
  - Check if MACs or HMACs are applied to critical data.
  - Verify the use of strong hashing algorithms (e.g., SHA-256 or higher).
  - Confirm digital signatures are used for non-repudiation where applicable.
- Validate that cryptographic keys are managed securely.

#### Tools:
- OpenSSL
- Python Cryptography libraries
- Custom scripts

### 5. Test Error Handling and Alerts

- Observe how the application responds to integrity violations:
  - Ensure proper error messages are displayed without exposing sensitive information.
  - Confirm logging and alerting mechanisms are in place for tampering attempts.

### 6. Check Business Logic Integrity

- Review workflows for logical integrity checks:
  - Verify the consistency of multi-step processes (e.g., transaction rollbacks).
  - Ensure no partial updates or logical inconsistencies occur.

## Remediation

1. Implement Integrity Validation:
   - Use MACs, HMACs, or digital signatures to validate data integrity.
   - Apply cryptographic mechanisms to both data in transit and data at rest.
2. Secure Cryptographic Keys:
   - Store keys in secure hardware modules or encrypted storage.
   - Regularly rotate and retire old keys.
3. Enforce Error Handling:
   - Implement consistent error responses for tampering attempts.
   - Log and monitor suspicious activity for further investigation.
4. Conduct Regular Audits:
   - Periodically review data integrity mechanisms.
   - Perform integrity checks during code reviews and testing cycles.

## References

- OWASP Testing Guide: Data Validation Testing
- [OWASP Cryptographic Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html)
- [NIST Cryptographic Standards](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines)
