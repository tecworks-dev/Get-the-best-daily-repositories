# WSTG-BUSL-05: Testing for Integrity of Business Logic

## Summary

The integrity of business logic ensures that an application enforces its intended rules and processes without manipulation. Weaknesses in business logic integrity may allow attackers to bypass workflows, modify data, or perform unauthorized actions. Testing for integrity ensures that the application consistently enforces business rules, even under malicious attempts.

## Objective

To identify vulnerabilities in the integrity of business logic that allow an attacker to:

- Bypass or modify critical workflows
- Perform unauthorized actions
- Corrupt or manipulate data without detection

## How to Test

### Step 1: Understand the Business Logic
1. Map the application’s business logic by reviewing documentation or analyzing workflows.
2. Identify critical rules or constraints that the application enforces, such as:
   - User authorization for specific actions
   - Constraints on data input or processing
   - Dependencies between processes

---

### Step 2: Identify Test Scenarios
Focus on scenarios where the integrity of the business logic could be compromised, such as:

1. **Enforcement of Rules**:
   - Ensure the application enforces its rules consistently across all workflows.
   - Example: An admin-only action should not be accessible to regular users via alternate interfaces.

2. **Input Validation**:
   - Verify that all inputs are validated correctly to enforce the intended logic.

3. **Cross-Function Dependencies**:
   - Test if changes in one function affect the integrity of other business processes.

4. **Business Rule Bypass**:
   - Look for ways to bypass rules, such as through direct API calls or parameter tampering.

---

### Step 3: Perform Integrity Testing
1. **Manipulate User Roles or Permissions**:
   - Test if lower-privileged users can perform actions restricted to higher-privileged users.
   - Example: Modify API requests to escalate user privileges.

2. **Tamper with Workflow Data**:
   - Modify parameters or session data to bypass validation.
   - Example: Change a discount value in a request to receive a larger discount than allowed.

3. **Check for Lack of Dependencies**:
   - Test if required conditions or dependencies are enforced.
   - Example: Submit a purchase request without fulfilling prerequisites like payment.

4. **Simulate Incomplete Workflows**:
   - Attempt to execute operations without completing previous steps.
   - Example: Submit a confirmation request without placing an order.

5. **Audit Logging and Monitoring**:
   - Verify if unauthorized actions are logged and monitored.
   - Example: Perform an invalid operation and check if it is logged appropriately.

---

### Step 4: Analyze Results
1. Identify and document scenarios where the application fails to enforce business rules.
2. Assess the potential impact, such as:
   - Data corruption
   - Unauthorized actions
   - Financial loss or reputational damage

---

## Tools

- **Proxy Tools** (e.g., Burp Suite, OWASP ZAP) for intercepting and modifying requests
- **Testing Scripts** (e.g., Python scripts using `requests` library) to automate and test workflow manipulations
- **Logging Analysis Tools** to verify that unauthorized actions are detected

---

## Remediation

1. **Validate Business Logic at All Layers**:
   - Enforce business rules at the client, server, and database layers.

2. **Implement Robust Input Validation**:
   - Ensure all input parameters are validated against business rules.

3. **Monitor and Log Critical Actions**:
   - Log unauthorized attempts and implement alerts for anomalous activities.

4. **Conduct Regular Reviews and Tests**:
   - Perform periodic integrity testing to identify and mitigate vulnerabilities.

5. **Follow the Principle of Least Privilege**:
   - Ensure that users and processes only have the permissions necessary to perform their roles.

---

## References

- [OWASP Testing Guide - Business Logic Testing](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A05:2021 Security Misconfiguration](https://owasp.org/Top10/A05_2021-Security_Misconfiguration/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
