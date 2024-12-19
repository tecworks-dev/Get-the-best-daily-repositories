Certainly! Here’s a documentation note for future reference, advising on how to maintain and customize the script, specifically focusing on the `Load-SignInLogs`, `Add-Result`, and `Process-DeviceItem` functions:

---

### **Script Maintenance and Customization Guide**

#### **1. Customizing the `Load-SignInLogs` Function**

The `Load-SignInLogs` function is responsible for loading and parsing sign-in logs from a JSON file. This function extracts specific properties from the JSON data and includes them in the `signInLog` object. 

- **When to Update:**
  - If you need to include additional properties from the JSON file (e.g., `location`, `clientAppUsed`), you should modify the `Load-SignInLogs` function to capture these properties.
  - If certain properties are no longer needed in the report, you can remove them from this function to simplify the `signInLog` object.

- **How to Update:**
  - Locate the section where each property is extracted, for example:
    ```powershell
    $deviceDetail = [PSCustomObject]@{
        DeviceId       = $element.GetProperty("deviceDetail").GetProperty("deviceId").GetString()
        DisplayName    = $element.GetProperty("deviceDetail").GetProperty("displayName").GetString()
        OperatingSystem = $element.GetProperty("deviceDetail").GetProperty("operatingSystem").GetString()
        IsCompliant    = $element.GetProperty("deviceDetail").GetProperty("isCompliant").GetBoolean()
        TrustType      = $element.GetProperty("deviceDetail").GetProperty("trustType").GetString()
    }
    ```
  - Add or remove lines within these blocks to include or exclude properties.

#### **2. Customizing the `Add-Result` Function**

The `Add-Result` function constructs a result object that is eventually added to the processing context. This result object is what gets exported in the final report.

- **When to Update:**
  - If you’ve updated the `Load-SignInLogs` function to include new properties, you’ll need to update `Add-Result` to reflect these changes.
  - Conversely, if you’ve removed properties from `Load-SignInLogs`, you should also remove the corresponding fields in `Add-Result` to avoid unnecessary data processing.

- **How to Update:**
  - Update the `Add-Result` function by modifying the `[PSCustomObject]` construction block:
    ```powershell
    $result = [PSCustomObject]@{
        DeviceName             = $deviceName
        UserName               = $Item.UserDisplayName
        DeviceEntraID          = $DeviceId
        UserEntraID            = $Item.UserId
        DeviceOS               = $Item.DeviceDetail.OperatingSystem
        OSVersion              = $osVersion
        DeviceComplianceStatus = $complianceStatus
        DeviceStateInIntune    = $DeviceState
        TrustType              = $Item.DeviceDetail.TrustType
        UserLicense            = $userLicense
        SignInStatus           = $signInStatus   # New property for Sign-In Status
        City                   = $Item.Location.City  # New property for City
        State                  = $Item.Location.State # New property for State
        CountryOrRegion        = $Item.Location.CountryOrRegion # New property for Country/Region
    }
    ```
  - Add new properties as needed, or remove those no longer required.

#### **3. Customizing the `Process-DeviceItem` Function**

The `Process-DeviceItem` function is where logic is applied to decide whether to process a sign-in log entry or skip it based on specific conditions, such as error codes or the presence of specific properties.

- **When to Update:**
  - Update this function when you want to change the conditions under which a sign-in log is processed or skipped. For example, you might want to skip logs with specific error codes or filter logs based on device compliance status.
  
- **How to Update:**
  - Locate the conditional logic where decisions are made, for example:
    ```powershell
    if ($Item.Status.ErrorCode -ne 0) {
        Write-EnhancedLog -Message "Sign-in attempt failed for user $($Item.UserDisplayName) with ErrorCode: $($Item.Status.ErrorCode) - $($Item.Status.FailureReason)" -Level "WARNING"
        return
    }
    ```
  - Adjust the conditions to reflect your new requirements.
  - Modify or add logging statements as necessary to capture the reasons for skipping or processing specific entries.

#### **Final Note:**
Whenever you update any of these functions, make sure to thoroughly test the script to ensure that the changes work as expected and that all necessary data is being captured or excluded based on the updates.

---

This guide should help maintain the script and ensure that any future changes can be made systematically, without losing track of why those changes are being implemented.