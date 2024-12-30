# Registration Vulnerability Checklist**

## Objective
To identify and test potential vulnerabilities in the registration process that could be exploited by attackers.

## Key Steps

### 1. Account Registration with Existing Email and Password Change
Test registering a new account with an existing email and change the password, ensuring that the new account can be logged in successfully.

### 2. DOS Attack on Name/Password Field
Test by entering a long string in the password field to trigger a "500 Internal Server Error."

### 3. Status Code Manipulation
Change the status code from 403 to 200, and manipulate the response to change an error message into a correct one (e.g., changing a bad response to a correct one).

### 4. Using Email as Username or Password
Check if the system allows using the email address as a username or password, which could lead to issues with the victim’s email being used as a username.

### 5. Third-Party Registration and Email Overlap
Test registering via third-party authentication (e.g., Google) and then try to register with the same email using a password method. This could lead to account takeover by either method.

### 6. Combining Email Addresses
Test combining email addresses like **victim@gmail.com@attacker.com** to see if the system allows this as a valid email.

### 7. Direct File Access via Path
Check if the application allows access to user profiles via a direct path (e.g., `/username`). This could be exploited by registering with filenames like `index.php` or `login.php`, which might replace important system pages with the attacker’s profile.

### 8. Creating an Email with Corporate Domain
Test if creating an email with the company domain can lead to privilege escalation, potentially exposing admin privileges.

### 9. SSTI Vulnerability in Name or Username
Test for **Server-Side Template Injection (SSTI)** vulnerabilities in the name or username fields.

### 10. Capitalizing Email and Username
Test if capitalizing letters in email and username causes any issues, such as account takeover or email deletion.

### 11. Special Characters in Email
Test by adding special characters before or after the email address, such as `%00`, `%09`, `%20`, `0d%0a`, and `\n%` to check for vulnerabilities.

### 12. Password Reset Page Exploitation
If there is no registration page but a password reset page exists, try adding an account and check if sending a password reset request to a non-existing account can open the admin dashboard.

### 13. SQL Injection in Name or Username
Test for SQL Injection in the name or username fields, e.g., using payloads like `'; --` to check for errors and also test blind injection techniques.
---
