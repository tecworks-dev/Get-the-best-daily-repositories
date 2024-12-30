# Login Vulnerability Checklist

## Objective
To identify and test potential vulnerabilities in the login process that could be exploited by attackers.

## Key Steps

### 1. Session Fixation via Logout Request
Send a logout request with the `user_id` changed to the victim’s ID and see if the session cookie or auth token for the victim is exposed in the response. Try using this token to log into the victim’s account.

### 2.Brute Force Attack
Test for the ability to perform brute force attacks if there is no limit on login attempts, targeting accounts like admin or other known emails once identified on the site.

### 3. Response Manipulation for Authentication Bypass
Modify response status codes or parameters in the response to bypass authentication or gain unauthorized access to accounts.

### 4. Content Type Manipulation
Change the content type to JSON, XML, or URL-encoded, then send a login request with these types. Observe any unintended data leakage or other issues that may arise from manipulating the content type.

### 5. **Method and Content Type Manipulation**
Start by changing the request method to a different one, such as POST or GET, and experiment with it. Also, modify the content type to JSON, XML, or URL-encoded, and send a login request using these combinations. Observe any errors or issues that occur in response.

### 6. Redirect After Login Exploit
Verify if you can change the redirect URL after login using parameters in the URL. This might allow the attacker to steal tokens or cookies if the user is redirected to a malicious website.


### 7. Session Persistence After Logout or Password Change
Check if sessions remain valid after logging out or changing the password, allowing attackers to use old cookies or session tokens to regain access.

### 8. Malformed Input Testing (Email/Password)
Test for issues in input validation by entering only an email or password, or entering boolean values like `true`, `false`, or `null` in either field to check for potential security flaws.


### 9. Login via HTTP Instead of HTTPS
Check if login credentials are transmitted over an unencrypted HTTP connection. This could expose sensitive data if the user visits the page from a public or shared network (MITM attack).

### 10. CSRF in login
Test if the email confirmation link can be used for CSRF attacks by tricking the user into logging into a fake account, then monitoring the victim’s actions and stealing sensitive data like credit card information.

### 11. SQL Injection in Username or Email
Test for SQL injection vulnerabilities in the username or email fields (e.g., `'; --`) and try various payloads, including blind injection techniques, to gain unauthorized access.

### 12. NoSQL Injection to Bypass Login
Test for NoSQL injection vulnerabilities in the login form, particularly in the username or email fields. Try using common NoSQL payloads, such as `{"$ne": null}` or `{"$gt": ""}`, to manipulate the query logic and bypass authentication. Observe if you can authenticate without valid credentials by exploiting NoSQL injection in the backend database.

### 13. LDAP Injection to Bypass Login
Test for LDAP injection vulnerabilities by manipulating the login fields (e.g., username or email) to alter the LDAP query structure. Try payloads such as `*)(uid=*))(|(uid=*` to bypass authentication logic. Observe if the query allows unauthorized access by injecting malicious LDAP queries into the login process.

### 14. XPath Injection to Bypass Login
Test for XPath injection vulnerabilities in the login form, particularly in the username or email fields. Attempt injecting XPath payloads, such as `' or '1'='1` or `' or 'x'='x`, to alter the XPath query and bypass authentication. Observe whether the server returns unintended results or grants access without proper authentication, allowing unauthorized login.
