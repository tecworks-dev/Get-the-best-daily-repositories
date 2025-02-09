## Custom Backend URL Support Plan

### High Priority Tasks
1. TypeScript Changes (src/purchases.ts):
   - Update `PurchasesConfiguration` interface to include optional `url` property
   - Rename `apiKey` to `apiToken` for better context with custom backends
   - Modify `configure` method to handle new `url` parameter
   - Update type definitions for native module calls

2. iOS Implementation (ios/RNPurchases.m):
   - Update `setupPurchases` method to accept `url` parameter
   - Modify RCPurchases configuration to use custom URL
   - Update backend request handling to use provided URL
   - Add validation for URL format

3. Android Implementation (android/src/main/java/com/wildberry/purchases/react/RNPurchasesModule.java):
   - Update `setupPurchases` method to accept `url` parameter
   - Modify CommonKt.configure call to include URL
   - Update backend request handling to use provided URL
   - Add validation for URL format

### Testing Requirements
- Unit tests for URL validation
- Integration tests with custom backend
- Example app updates to demonstrate custom URL usage
- Error handling tests for invalid URLs

### Documentation Updates
- Update README with custom backend setup instructions
- Add migration guide for existing users
- Document URL format requirements
- Add example configuration snippets

### Implementation Notes
- Maintain backward compatibility for existing users
- Ensure proper error handling for URL-related issues
- Consider adding URL validation utilities
- Add logging for backend URL configuration