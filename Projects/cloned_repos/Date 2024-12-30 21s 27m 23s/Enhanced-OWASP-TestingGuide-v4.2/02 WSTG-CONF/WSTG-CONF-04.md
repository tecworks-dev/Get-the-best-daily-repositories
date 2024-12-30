# WSTG-CONF-04: Review Old Backup and Unreferenced Files for Sensitive Information

## Objective
Ensure that old backup files, unreferenced files, or leftover test files do not expose sensitive information or functionality.

## Checklist

### 1. Identify Backup and Temporary Files
- Search for backup files commonly left behind:
  - `.bak`, `.backup`, `.tmp`, `.old`
  - `intitle:"index of" "backup.zip"` search for bakcup files via index. 
  - File duplicates with tilde (e.g., `index.php~`).
- Use tools to enumerate files in directories.

### 2. Look for Unreferenced Files
- Review publicly accessible directories for unlinked files.
- Check for test, debug, or old versions of application files.
- Check the index page for sensitive files `intitle:"index of"`.
- Perform Fuzzing based on the specific technology (e.g. `Apache Tomcat`).
  ```sh
  dirsearch -u https://sub2.sub1.domain.com -x 403,404,500,400,502,503,429 -w /usr/share/seclists/Discovery/Web-Content/ApacheTomcat.fuzz.txt
  ```

### 3. Test Access Control
- Verify whether sensitive files are accessible without authentication.
- Examples:
  - Source code files (e.g., `.php`, `.asp`).
  - Environment files (`.env`).

### 4. Analyze Contents of Exposed Files
- Inspect accessible files for:
  - Hardcoded credentials.
  - API keys or tokens.
  - Database connection strings.
  - Sensitive configurations.

### 5. Review Hidden Directories
- Check for directories or files not meant to be publicly accessible:
  - `.git/`
  - `.svn/`
  - `/backup/`
  - Use [DotGit](https://github.com/davtur19/DotGit) extension.

### 6. Check Web Application Logs
- Look for unprotected log files (e.g., `access.log`, `error.log`, `debug.log`).

## Tools
- **Dirb/Gobuster**: For directory and file enumeration.
- **Burp Suite**: For manual testing and crafting specific requests.
- **DotGit**: An extension for checking if .git is exposed in visited websites.
- **Nmap**: To scan for open directories and files.

---

Let me know if you'd like to refine this section or proceed to the next!
