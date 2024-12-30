## WSTG-CONF-11: Test Cloud Storage

### Objective
The goal is to identify misconfigurations and vulnerabilities in cloud storage services used by the application. Misconfigured cloud storage can expose sensitive data or lead to unauthorized access.

### Testing Approach

1. **Check for Publicly Accessible Buckets**
   - Verify if storage buckets (e.g., AWS S3, Azure Blob Storage, Google Cloud Storage) are publicly accessible without authentication.
   - Tools: `awscli`, `gcloud`, `azcli`, or third-party tools like `s3scanner`.

2. **Test Permissions**
   - Validate that permissions for the storage bucket are appropriately set:
     - Restrict `READ`, `WRITE`, and `LIST` permissions to authorized users only.
   - Tools: Cloud provider-specific CLI or web console.

3. **Search for Sensitive Data**
   - Check for sensitive files stored in the cloud (e.g., backups, credentials, environment variables).
   - Tools: Manual review, automated scanners like TruffleHog or GitLeaks.

4. **Analyze Storage Bucket Policies**
   - Review bucket policies to ensure they enforce least privilege.
   - Ensure no wildcard `*` permissions are granted.
   - Tools: Policy analyzers or manual inspection.

5. **Test Encryption Settings**
   - Verify that data is encrypted at rest and in transit:
     - Check for usage of encryption keys (e.g., AWS KMS).
     - Confirm that HTTPS is used for data transfer.
   - Tools: Manual review or cloud provider-specific analyzers.

6. **Verify Access Logging**
   - Check if logging is enabled for cloud storage to track access and changes.
   - Ensure logs are sent to a secure location.

7. **Evaluate CORS Configuration**
   - Ensure Cross-Origin Resource Sharing (CORS) settings are restrictive and do not allow arbitrary domains.
   - Tools: Manual review or cloud configuration tools.

### Tools and Resources
- **AWS CLI**
  - Example: `aws s3api list-buckets`
- **Azure CLI**
  - Example: `az storage blob list`
- **Google Cloud SDK**
  - Example: `gsutil ls`
- **Third-Party Tools**
  - `s3scanner`, `bucket-stream`, `TruffleHog`

### Recommendations
- Implement strict access control for cloud storage.
- Regularly audit storage policies and permissions.
- Encrypt sensitive data at rest and in transit.
- Enable access logging and monitor for anomalies.

---

