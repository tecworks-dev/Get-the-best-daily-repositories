# WSTG-CRYP-01
# WSTG-CRYP-01: Test for Weak SSL/TLS Ciphers, Insufficient Transport Layer Protection

## Summary

Testing for weak SSL/TLS ciphers and insufficient transport layer protection ensures that sensitive data in transit is adequately encrypted to prevent interception or tampering by attackers.

## Objectives

1. Verify the use of strong SSL/TLS configurations.
2. Identify weak ciphers, protocols, or configurations that may expose data to risks.
3. Confirm that all sensitive data transmissions are encrypted and secure.

## How to Test

### 1. Identify Entry Points

- Determine all entry points where SSL/TLS is implemented, such as:
  - Web applications
  - APIs
  - Mobile applications

### 2. Check SSL/TLS Certificate

- Validate the SSL/TLS certificate:
  - Ensure the certificate is valid and not expired.
  - Confirm the certificate is issued by a trusted Certificate Authority (CA).
  - Verify the domain name matches the certificate's Subject Name (CN) or Subject Alternative Names (SAN).

#### Tools:
- OpenSSL
- Online tools like [SSL Labs](https://www.ssllabs.com/)

### 3. Analyze SSL/TLS Configuration

- Identify supported SSL/TLS versions and ensure only secure versions are enabled:
  - TLS 1.2 or higher is recommended.
  - Disable SSL 2.0, SSL 3.0, TLS 1.0, and TLS 1.1.
- Check for secure cipher suites:
  - Avoid weak ciphers such as RC4, DES, or 3DES.
  - Prefer modern ciphers like AES-GCM.
- Confirm Perfect Forward Secrecy (PFS) is enabled:
  - Look for ciphers like ECDHE and DHE.

#### Tools:
- Nmap (`ssl-enum-ciphers` script)
- sslyze
- TestSSL.sh

### 4. Ensure Proper Implementation of HSTS

- Verify the use of HTTP Strict Transport Security (HSTS):
  - Check for the `Strict-Transport-Security` header in HTTP responses.
  - Ensure the `max-age` directive is set and sufficient.
  - Include the `includeSubDomains` directive if applicable.

#### Tools:
- Browser developer tools
- curl or wget

### 5. Evaluate Security Headers

- Verify security headers related to SSL/TLS:
  - `X-Content-Type-Options`
  - `X-Frame-Options`
  - `X-XSS-Protection`

### 6. Test for Known Vulnerabilities

- Test for known SSL/TLS vulnerabilities:
  - Heartbleed
  - BEAST
  - POODLE
  - DROWN
  - BREACH
  - CRIME

#### Tools:
- Nessus
- Qualys SSL Labs
- Burp Suite (Pro version)

## Remediation

1. Use a strong SSL/TLS configuration:
   - Enforce TLS 1.2 or higher.
   - Disable weak ciphers and protocols.
   - Enable PFS.
2. Regularly review and update SSL/TLS configurations.
3. Implement and enforce HSTS for all web applications.
4. Monitor and remediate SSL/TLS vulnerabilities promptly.
5. Maintain valid and trusted SSL/TLS certificates.

## References

- OWASP Transport Layer Protection Cheat Sheet
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [SSL Labs Best Practices](https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices)
