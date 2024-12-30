# WSTG-BUSL-06: Testing for Circumvention of Workflows

## Summary

Workflow circumvention vulnerabilities occur when an application fails to enforce the correct sequence of actions, allowing attackers to bypass or manipulate workflows. These vulnerabilities can lead to unauthorized access, data corruption, or bypassing critical business rules.

## Objective

To identify vulnerabilities in workflows where an attacker can:

- Skip mandatory steps in a process
- Perform actions out of sequence
- Circumvent restrictions imposed by the application

## How to Test

### Step 1: Understand the Workflow
1. Map out the business workflows, including:
   - Key steps required to complete processes.
   - Dependencies and validations between steps.

2. Identify critical workflows where circumvention could have significant impacts, such as:
   - Registration and authentication
   - Payment processing
   - Order placement and fulfillment
   - Approval and escalation processes

---

### Step 2: Analyze Workflow Enforcement
Review how the application enforces workflow integrity, such as:

1. **Session Tracking**:
   - Determine if the application tracks session states between workflow steps.

2. **State Validation**:
   - Identify if each step validates the state of the workflow.

3. **Parameter Dependencies**:
   - Check if the application relies on client-side parameters to enforce workflows.

---

### Step 3: Perform Circumvention Tests
1. **Skip Steps**:
   - Bypass mandatory steps by accessing endpoints directly.
   - Example: Attempt to submit a payment request without completing an order.

2. **Reorder Steps**:
   - Execute steps in a different sequence than intended.
   - Example: Approve an action before completing the prerequisite review.

3. **Parameter Manipulation**:
   - Modify parameters in requests to skip validations.
   - Example: Change a "step_completed" parameter to proceed to the next step without completing the current one.

4. **Replay Requests**:
   - Test if previously valid requests can be replayed to bypass workflows.
   - Example: Resubmit an approval request after a denial to override the decision.

5. **Tamper with State Data**:
   - Manipulate cookies, session data, or hidden form fields to alter the workflow state.

---

### Step 4: Analyze Results
1. Identify instances where:
   - Steps were bypassed or reordered.
   - Validation was insufficient to enforce workflow integrity.

2. Document the business impact, such as:
   - Financial loss due to payment circumvention.
   - Security risks from bypassing authentication or authorization.
   - Data corruption from incomplete processes.

---

## Tools

- **Proxy Tools** (e.g., Burp Suite, OWASP ZAP) for intercepting and modifying requests
- **Automated Workflow Testing Tools** for systematically testing step bypasses
- **Custom Scripts** (e.g., Python or Postman) to test parameter tampering and state manipulation

---

## Remediation

1. **Implement Server-Side Workflow Enforcement**:
   - Enforce sequence and dependencies of steps on the server side.
   - Use server-side session management to track workflow states.

2. **Validate State Transitions**:
   - Verify that all workflow transitions follow the defined sequence.

3. **Secure Parameters and States**:
   - Avoid relying on client-side parameters for workflow validation.
   - Use cryptographically secure tokens for session and workflow state tracking.

4. **Log and Monitor Workflow Activities**:
   - Record workflow steps and monitor for anomalies.

5. **Perform Regular Testing**:
   - Conduct routine tests to identify and mitigate workflow circumvention vulnerabilities.

---

## References

- [OWASP Testing Guide - Business Logic Testing](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A01:2021 Broken Access Control](https://owasp.org/Top10/A01_2021-Broken_Access_Control/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
