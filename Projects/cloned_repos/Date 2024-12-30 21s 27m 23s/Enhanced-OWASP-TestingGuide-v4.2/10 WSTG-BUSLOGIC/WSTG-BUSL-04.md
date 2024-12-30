# WSTG-BUSL-04: Testing for Business Logic Process Timing

## Summary

Timing vulnerabilities in business logic occur when applications rely on specific timing or sequence of events to enforce rules or constraints. Exploiting these weaknesses can allow attackers to bypass security controls, cause unexpected behavior, or access unauthorized resources.

## Objective

To identify and exploit weaknesses in the application's business logic related to timing or sequence of processes, such as:

- Race conditions
- Improper handling of delayed or repeated requests
- Exploiting asynchronous processes for unintended outcomes

## How to Test

### Step 1: Understand the Application Workflow
1. Map the business processes and workflows within the application.
   - Identify time-sensitive operations or sequences of events.
   - Examples: Payment processing, token expiration, multi-step forms.

2. Determine if the application uses asynchronous processing, distributed systems, or relies on user actions within specific time windows.

---

### Step 2: Identify Test Scenarios
Focus on scenarios that may involve timing or sequencing issues, such as:
- Concurrent modification of resources
- Time-bound operations (e.g., session expiration, temporary tokens)
- Sequential dependencies (e.g., approvals, escalations)

---

### Step 3: Perform Timing-Based Tests
1. **Test for Race Conditions**:
   - Simulate concurrent requests for time-sensitive operations.
   - Example: Attempt to withdraw more money than allowed by submitting multiple requests simultaneously.

2. **Delay Request Responses**:
   - Intentionally delay the submission of responses to test if the application handles delayed inputs properly.
   - Example: Leave a token-based form open and attempt submission after the token has expired.

3. **Replay or Repeat Requests**:
   - Repeat requests for time-sensitive operations to test for unintended consequences.
   - Example: Replay a coupon application request to receive multiple discounts.

4. **Modify Timestamps**:
   - Manipulate timestamps in requests or cookies to bypass timing restrictions.
   - Example: Change an expiration date to prolong session validity.

---

### Step 4: Analyze Results
1. Document observed impacts, such as:
   - Bypassing security measures
   - Exploiting timing gaps for unauthorized actions
   - Causing data inconsistency or corruption

2. Identify business impact scenarios, including financial loss, reputation damage, or regulatory violations.

---

## Tools

- **Proxy Tools** (e.g., Burp Suite, OWASP ZAP) for intercepting and modifying requests
- **Concurrency Testing Tools** (e.g., Turbo Intruder, custom scripts) to simulate race conditions
- **Delay Simulation Tools** to test delayed responses (e.g., browser developer tools, interception proxies)

---

## Remediation

1. **Implement Atomic Transactions**:
   - Use database or application-level locks to prevent race conditions.

2. **Use Strong Timestamp Validation**:
   - Reject requests with manipulated or expired timestamps.

3. **Employ Idempotent Operations**:
   - Design APIs and processes to handle repeated requests gracefully.

4. **Enforce Sequential Integrity**:
   - Validate that actions are performed in the correct sequence.

5. **Test Regularly for Timing Weaknesses**:
   - Perform stress tests and concurrent execution tests to identify timing vulnerabilities.

---

## References

- [OWASP Top Ten - A01:2021 Broken Access Control](https://owasp.org/Top10/A01_2021-Broken_Access_Control/)
- [OWASP Guide on Race Conditions](https://owasp.org/www-community/vulnerabilities/Race_Conditions)
- [NIST SP 800-53 - Security and Privacy Controls](https://csrc.nist.gov/publications/sp800-53)
