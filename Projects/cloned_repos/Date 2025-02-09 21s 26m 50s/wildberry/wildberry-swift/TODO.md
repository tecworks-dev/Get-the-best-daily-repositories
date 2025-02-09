# Wildberry Swift SDK Configuration Update

## High-Level Tasks

### 1. Update Configuration Methods
- [ ] Modify `Purchases.configure` methods to accept custom URL
- [ ] Add new `with(url:)` method in `Configuration.Builder`
- [ ] Store provided URL for network requests

### 2. Backend and Network Updates
- [ ] Modify `Backend` class to accept baseURL in initialization
- [ ] Update `HTTPClient` to use provided baseURL
- [ ] Remove hardcoded RevenueCat API endpoints
- [ ] Replace all instances of https://api.wildberry.com with configurable URL

### 3. API Key Integration
- [ ] Verify API key handling in Authorization headers
- [ ] Adapt authentication for custom backend

### 4. Test Suite Updates
- [ ] Update `PurchasesTests`
- [ ] Modify `BackendIntegrationTests`
- [ ] Update `Fastfile` for api_key_integration_tests
- [ ] Add new tests for custom URL configuration

### 5. Documentation Updates
- [ ] Update README.md with new configuration instructions
- [ ] Update Sources/DocCDocumentation/DocCDocumentation.docc/RevenueCat.md
- [ ] Create migration guide for existing users

## File Changes Required

### Configuration Updates
- `Sources/Purchasing/Purchases/Purchases.swift`
  - Add URL configuration support
  - Update configuration builder

### Backend Updates
- `Sources/Backend/Backend.swift`
  - Add baseURL parameter
  - Update network request construction

### Networking Updates
- `Sources/Networking/HTTPClient/HTTPRequestPath.swift`
  - Implement dynamic URL construction
  - Remove hardcoded endpoints

### Test Updates
- Update relevant test files to support custom URL configuration
- Add new test cases for URL validation