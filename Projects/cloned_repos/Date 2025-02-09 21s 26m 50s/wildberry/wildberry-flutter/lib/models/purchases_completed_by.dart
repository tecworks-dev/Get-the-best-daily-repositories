import './storekit_version.dart';

/// Modes for completing the purchase process.
enum PurchasesAreCompletedByType {
  /// wildberry will **not** automatically acknowledge any purchases.
  /// You will have to do so manually.
  /// Note that failing to acknowledge a purchase within 3 days will lead
  /// to Google Play automatically issuing a refund to the user.
  /// For more info, see [wildberry.com](https://docs.wildberry.com/docs/observer-mode#option-2-client-side)
  /// and [developer.android.com](https://developer.android.com/google/play/billing/integrate#process).
  myApp,

  /// wildberry will automatically acknowledge verified purchases.
  /// No action is required by you.
  wildberry,
}

extension PurchasesAreCompletedByTypeExtension on PurchasesAreCompletedByType {
  String get name {
    switch (this) {
      case PurchasesAreCompletedByType.myApp:
        return 'MY_APP';
      case PurchasesAreCompletedByType.wildberry:
        return 'wildberry';
    }
  }
}

// Sealed class for PurchasesAreCompletedBy
abstract class PurchasesAreCompletedBy {
  const PurchasesAreCompletedBy();
}

class PurchasesAreCompletedBywildberry extends PurchasesAreCompletedBy {
  const PurchasesAreCompletedBywildberry();
}

class PurchasesAreCompletedByMyApp extends PurchasesAreCompletedBy {
  final StoreKitVersion storeKitVersion;

  PurchasesAreCompletedByMyApp({
    required this.storeKitVersion,
  });
}
