# WSTG-CONF-05: Enumerate Infrastructure and Application Admin Interfaces

## Objective
Identify and evaluate the security of infrastructure and application administration interfaces to prevent unauthorized access and potential exploitation.

## Checklist

### 1. Discover Admin Interfaces
- Use automated tools to identify potential admin portals:
  - Common paths: `/admin`, `/administrator`, `/login`, `/wp-admin`.
  - Custom naming conventions: `/controlpanel`, `/manage`.
- Analyze HTTP headers and responses for clues indicating an admin interface.

### 2. Verify Access Control Mechanisms
- Test whether admin interfaces require authentication.
- Attempt to access sensitive functionalities without proper credentials.

### 3. Assess Authentication Mechanisms
- Ensure secure authentication is enforced:
  - Multi-factor authentication (MFA).
  - Strong password policies.
  - Rate-limiting and lockout mechanisms for failed login attempts.

### 4. Analyze Admin Interface Features
- Check for unnecessary or overly privileged functionalities exposed in the interface.
- Test for any debug or developer options left enabled.

### 5. Evaluate HTTP Headers
- Inspect headers to ensure sensitive information is not leaked.
- Check for proper configuration of security headers (e.g., `X-Frame-Options`, `Content-Security-Policy`).

### 6. Scan for Unused or Deprecated Interfaces
- Identify older or unused admin portals and verify they are disabled or removed.

### 7. Test for Directory Listings
- Confirm directory listing is disabled for admin-related directories.

### 8. Check for Default Credentials
- Test common default username/password combinations for known platforms.
- Examples:
  - `admin/admin`
  - `root/root`

## Tools
- **Dirb/Gobuster**: For brute-forcing admin interface paths.
- **Burp Suite**: For analyzing responses and testing authentication mechanisms.
- **Nmap**: To scan for admin-related services or ports.
- **WhatWeb/Wappalyzer**: For fingerprinting admin panels and technologies.

---

Let me know if you'd like any updates or are ready for the next file!
