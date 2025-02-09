# Wildberry Flutter Plugin TODO List

## High Priority

### API Configuration Refactoring
- [ ] Modify `PurchasesConfiguration` class
  - Remove `apiKey` field
  - Add `serverUrl` field
  - Remove `AmazonConfiguration` class
  - Update configuration constructor and documentation

### Native Code Updates
- [ ] iOS Updates (`PurchasesFlutterPlugin.m`)
  - [ ] Update `setupPurchases` method to use `serverUrl`
  - [ ] Modify configuration method chain to use new parameters
  - [ ] Remove deprecated store-specific parameters
  - [ ] Update verification mode handling

- [ ] Android Updates (`PurchasesFlutterPlugin.java`)
  - [ ] Remove `useAmazon` boolean logic
  - [ ] Update `setupPurchases` method for `serverUrl`
  - [ ] Clean up store-specific parameters
  - [ ] Update configuration builder pattern

## Medium Priority

### Dart Code Updates
- [ ] Update `purchases_flutter.dart`
  - [ ] Refactor setup method with new configuration
  - [ ] Remove platform-specific logic
  - [ ] Remove deprecated `setAllowSharingStoreAccount`
  - [ ] Clean up Amazon-specific code

### Testing & Documentation
- [ ] Update test suite
  - [ ] Modify existing unit tests for new configuration
  - [ ] Add new test cases for `serverUrl` functionality
  - [ ] Update mock responses in tests
- [ ] Update example apps
  - [ ] Modify purchase_tester app
  - [ ] Update configuration examples
- [ ] Update documentation
  - [ ] Update README.md with new setup instructions
  - [ ] Update API documentation
  - [ ] Add migration guide for existing users

## Low Priority

### Future Improvements
- [ ] Implement better error handling for invalid server URLs
- [ ] Add URL validation in configuration
- [ ] Consider adding configuration validation methods
- [ ] Add telemetry for tracking configuration usage
- [ ] Consider implementing configuration caching

### Technical Debt
- [ ] Remove deprecated methods and classes
- [ ] Clean up unused imports
- [ ] Update minimum SDK versions if needed
- [ ] Review and update dependencies

## Notes
- All platform-specific logic will be consolidated into a single configuration flow
- Breaking changes should be clearly documented in the migration guide
- Consider backward compatibility for existing implementations
- Test across all supported platforms (iOS, Android, macOS)