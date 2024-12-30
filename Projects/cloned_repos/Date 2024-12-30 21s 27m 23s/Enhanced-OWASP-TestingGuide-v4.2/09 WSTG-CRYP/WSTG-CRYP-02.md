# WSTG-CRYP-02: Test for Padding Oracle

## Summary

Padding oracle vulnerabilities allow attackers to decrypt, modify, or forge encrypted data without access to the encryption key. These issues arise from improper handling of cryptographic padding errors in block cipher modes like CBC.

## Objectives

1. Identify whether the application is vulnerable to padding oracle attacks.
2. Understand the impact of potential exploitation, including data decryption and unauthorized data manipulation.

## How to Test

### 1. Identify Cryptographic Functionality

- Locate functionalities using cryptographic operations, such as:
  - User authentication tokens
  - Encrypted cookies
  - API request/response payloads
  - Sensitive data stored or transmitted

### 2. Analyze Responses for Padding Errors

- Test for distinguishable error messages:
  - Monitor server responses for cryptographic operations that might reveal padding-related error messages.
  - Look for differences in responses for invalid ciphertexts versus valid ones.
- Verify if the application responds with a specific error (e.g., "Padding is invalid" or "Decryption failed") when manipulated ciphertext is submitted.

#### Tools:
- Burp Suite
- Padding Oracle Exploit Tool (POET)
- Custom scripts

### 3. Perform Ciphertext Manipulation

- Modify encrypted data systematically to identify patterns:
  - Use a valid ciphertext and tamper with individual bytes.
  - Submit the manipulated ciphertext and observe server responses.
- Repeat the process to detect if the application leaks information about the padding structure.

#### Tools:
- OpenSSL
- Python (PyCrypto or Cryptography libraries)
- Custom-built scripts

### 4. Exploit the Padding Oracle (if vulnerability exists)

- Demonstrate the impact of the vulnerability:
  - Decrypt sensitive information.
  - Forge valid encrypted data.
- Validate the extent of data exposure or unauthorized data modification.

#### Example:
1. Capture a valid encrypted token (e.g., a session token).
2. Tamper with the ciphertext and monitor server responses to identify padding-related errors.
3. Use padding oracle exploitation techniques to recover plaintext data or craft a valid token.

### 5. Evaluate Mitigations

- Check if proper mitigations are in place:
  - Use authenticated encryption (e.g., AES-GCM or ChaCha20-Poly1305) instead of CBC mode.
  - Ensure consistent error messages for encryption-related issues.
  - Limit exposure of encrypted data to external actors.

## Remediation

1. Use modern cryptographic algorithms and modes:
   - Replace CBC mode with authenticated encryption schemes (e.g., AES-GCM).
2. Handle all decryption errors generically:
   - Ensure error messages do not reveal information about the nature of the failure.
3. Implement rigorous input validation:
   - Restrict user-controlled data from being processed in cryptographic operations.
4. Conduct regular cryptographic security reviews:
   - Audit the implementation of cryptographic functions.

## References

- OWASP Cryptographic Storage Cheat Sheet
- [Practical Padding Oracle Attacks](https://www.usenix.org/legacy/events/sec02/full_papers/vaudenay/vaudenay.pdf)
- [Padding Oracle Exploit Tool (POET)](https://github.com/GDSSecurity/Poet)
