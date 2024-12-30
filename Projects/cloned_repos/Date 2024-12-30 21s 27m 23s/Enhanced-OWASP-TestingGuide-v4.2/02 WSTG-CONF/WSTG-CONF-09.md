# WSTG-CONF-09: Test File Permission

## Objective
Ensure that file permissions are configured securely to prevent unauthorized access, modification, or deletion of sensitive files.

## Checklist

### 1. Identify Accessible Files
- Enumerate files and directories accessible via the web server.
- Verify that sensitive files (e.g., configuration files, credentials) are not exposed.

### 2. Validate Read Permissions
- Confirm that files intended for public access do not reveal sensitive data.
- Ensure files like `.env`, `config.php`, and `database.yml` are not readable by unauthorized users.

### 3. Check Write Permissions
- Verify that writable directories (e.g., `uploads/`) are restricted:
  - Ensure only authorized users can upload files.
  - Prevent attackers from uploading malicious files or overwriting existing files.

### 4. Inspect Execute Permissions
- Ensure that executable files are limited to intended scripts only.
- Prevent unauthorized users from executing arbitrary code or commands.

### 5. Review Directory Listings
- Confirm that directory indexing is disabled to prevent unauthorized file enumeration.

### 6. Verify Permissions of Backup Files
- Check that backup files (e.g., `.bak`, `.old`) have restricted access or are removed from the server.

### 7. Test Access Control Mechanisms
- Test for proper implementation of access control rules:
  - Ensure files are accessible only to authenticated and authorized users.

### 8. Analyze Permissions of Hidden Files
- Verify the security of hidden files (e.g., `.htaccess`, `.git/`) to prevent unauthorized access.

## Tools
- **Nmap**: For identifying accessible files and directories.
- **Burp Suite**: To inspect file responses and permissions.
- **Nikto**: For automated file and directory enumeration.
- **Gobuster/Dirb**: To brute-force directories and files for permission misconfigurations.

---

Let me know if you'd like to make changes or proceed to the next file!
