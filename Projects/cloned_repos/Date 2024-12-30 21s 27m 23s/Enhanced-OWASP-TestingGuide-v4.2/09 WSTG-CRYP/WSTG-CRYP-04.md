# WSTG-CRYP-04: Test for Weak Encryption

## Summary

Testing for weak encryption ensures that sensitive data is protected using secure algorithms, key lengths, and configurations. Weak encryption can lead to data breaches, unauthorized access, or other security vulnerabilities.

## Objectives

1. Identify the use of insecure or outdated cryptographic algorithms.
2. Ensure the implementation of secure encryption practices.
3. Validate the protection of sensitive data at rest and in transit.

## How to Test

### 1. Identify Cryptographic Implementations

- Locate where encryption is applied:
  - Sensitive data storage (databases, files).
  - Data transmission (APIs, web applications).
  - Encrypted cookies, tokens, or keys.

### 2. Review Algorithms and Key Lengths

- Check for the use of secure encryption algorithms:
  - Avoid weak algorithms such as DES, 3DES, RC4, and MD5.
  - Use strong algorithms like AES (128-bit, 256-bit) and SHA-2 or SHA-3 for hashing.
- Validate key lengths:
  - Ensure the use of adequate key lengths (e.g., AES-128 or higher).
  - Avoid short or hardcoded keys.

#### Tools:
- OpenSSL
- Crypto libraries (e.g., PyCrypto, Cryptography in Python)
- Custom scripts

### 3. Analyze Key Management Practices

- Verify the secure generation and storage of cryptographic keys:
  - Ensure keys are generated using secure random number generators.
  - Confirm that keys are stored securely (e.g., in hardware security modules or encrypted storage).
  - Avoid embedding keys in source code or configuration files.

#### Tools:
- Code review tools
- Static analysis tools (e.g., SonarQube)

### 4. Evaluate Encryption Implementations

- Review encryption and decryption processes:
  - Ensure proper use of initialization vectors (IVs) and nonces (random and unique for each encryption operation).
  - Validate the use of authenticated encryption (e.g., AES-GCM or ChaCha20-Poly1305).

#### Tools:
- Code reviews
- Custom test scripts

### 5. Test for Known Weaknesses

- Look for vulnerabilities in encryption implementations:
  - Check for side-channel vulnerabilities (e.g., timing attacks, padding oracle attacks).
  - Test for improper padding or configuration issues.

#### Tools:
- TestSSL.sh
- Nmap (`ssl-enum-ciphers` script)
- Burp Suite

### 6. Verify Secure Data Storage

- Confirm that sensitive data at rest is encrypted securely:
  - Review database and file encryption mechanisms.
  - Ensure proper key management for encrypted data.

#### Tools:
- Database inspection tools
- Forensic tools

## Remediation

1. Use Modern Algorithms:
   - Replace weak algorithms (e.g., DES, RC4) with secure alternatives like AES or RSA with appropriate key lengths.
2. Enforce Proper Key Management:
   - Generate keys securely and store them in secure environments.
   - Rotate keys periodically and decommission old keys.
3. Implement Authenticated Encryption:
   - Use AES-GCM or ChaCha20-Poly1305 for encryption.
4. Regularly Audit Cryptographic Practices:
   - Perform regular code reviews and security assessments.
   - Update cryptographic libraries to patch known vulnerabilities.
5. Educate Developers:
   - Train developers on cryptographic best practices and secure implementation.

## References

- OWASP Cryptographic Storage Cheat Sheet
- [NIST Cryptographic Standards](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
