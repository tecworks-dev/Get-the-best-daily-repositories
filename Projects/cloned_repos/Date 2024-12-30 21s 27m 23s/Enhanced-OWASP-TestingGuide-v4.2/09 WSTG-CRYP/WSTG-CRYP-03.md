# WSTG-CRYP-03: Test for Sensitive Information Sent via Unencrypted Channels

## Summary

Testing for sensitive information sent via unencrypted channels ensures that all sensitive data in transit is adequately protected from eavesdropping or interception by malicious actors.

## Objectives

1. Identify unencrypted communication channels transmitting sensitive information.
2. Verify the implementation of proper encryption protocols for sensitive data.

## How to Test

### 1. Identify Communication Channels

- Determine all channels where sensitive data is transmitted, such as:
  - Web applications (HTTP/HTTPS)
  - Mobile applications (API requests)
  - Backend server communication
  - Email or messaging systems

### 2. Intercept Network Traffic

- Use network monitoring tools to capture and analyze traffic:
  - Look for sensitive data transmitted in plaintext (e.g., passwords, PII, session tokens).
  - Identify any HTTP traffic for data that should be sent over HTTPS.

#### Tools:
- Wireshark
- Burp Suite
- Mitmproxy
- tcpdump

### 3. Check SSL/TLS Usage

- Verify if SSL/TLS is implemented:
  - Ensure HTTPS is used for web applications.
  - Confirm APIs use HTTPS for requests and responses.
  - Check if email systems use secure protocols (e.g., SMTPS, IMAPS, POP3S).

#### Tools:
- OpenSSL
- Qualys SSL Labs
- Browser developer tools

### 4. Analyze Security Headers

- Check for HTTP Strict Transport Security (HSTS):
  - Ensure the `Strict-Transport-Security` header is present in HTTP responses.
  - Verify the `max-age` directive is set appropriately.
  - Confirm the use of `includeSubDomains` where applicable.

#### Tools:
- curl
- wget
- Browser developer tools

### 5. Test Downgrade Attacks

- Verify resistance to protocol downgrade attacks:
  - Check if secure channels (e.g., HTTPS) can be downgraded to insecure protocols (e.g., HTTP).

#### Tools:
- sslstrip
- Custom scripts

## Remediation

1. Enforce Encryption:
   - Ensure all sensitive data is transmitted over secure channels (e.g., HTTPS, TLS, SMTPS).
2. Implement HSTS:
   - Add the `Strict-Transport-Security` header to enforce HTTPS.
3. Disable Insecure Protocols:
   - Prevent fallback to insecure protocols (e.g., HTTP, SSL 3.0, TLS 1.0/1.1).
4. Conduct Regular Security Audits:
   - Periodically review and test communication channels to ensure proper encryption is enforced.
5. Educate Developers:
   - Provide training on secure communication practices.

## References

- OWASP Transport Layer Protection Cheat Sheet
- [HTTP Strict Transport Security (HSTS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Strict-Transport-Security)
- [SSL/TLS Deployment Best Practices](https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices)
