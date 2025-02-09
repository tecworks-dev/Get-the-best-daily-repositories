import React from "react";

import wildberryUI, {
  FooterPaywallViewOptions,
  FullScreenPaywallViewOptions,
  PAYWALL_RESULT,
  PaywallViewOptions,
  PresentPaywallIfNeededParams,
  PresentPaywallParams,
} from "../react-native-purchases-ui";
import {
  CustomerInfo,
  PurchasesError,
  PurchasesOffering,
  PurchasesOfferings,
  PurchasesPackage,
  PurchasesStoreTransaction,
} from "@wildberry/purchases-typescript-internal";

async function checkPresentPaywall(offering: PurchasesOffering) {
  let paywallResult: PAYWALL_RESULT = await wildberryUI.presentPaywall({});
  paywallResult = await wildberryUI.presentPaywall();
  paywallResult = await wildberryUI.presentPaywall({
    offering: offering,
  });
  paywallResult = await wildberryUI.presentPaywall({
    displayCloseButton: false,
  });
  paywallResult = await wildberryUI.presentPaywall({
    offering: offering,
    displayCloseButton: false,
  });
  paywallResult = await wildberryUI.presentPaywall({
    offering: offering,
    displayCloseButton: false,
    fontFamily: 'Ubuntu',
  });
}

async function checkPresentPaywallIfNeeded(offering: PurchasesOffering) {
  let paywallResult: PAYWALL_RESULT = await wildberryUI.presentPaywallIfNeeded(
    {
      requiredEntitlementIdentifier: "entitlement",
    }
  );
  paywallResult = await wildberryUI.presentPaywallIfNeeded({
    requiredEntitlementIdentifier: "entitlement",
    offering: offering,
  });
  paywallResult = await wildberryUI.presentPaywallIfNeeded({
    requiredEntitlementIdentifier: "entitlement",
    displayCloseButton: false,
  });
  paywallResult = await wildberryUI.presentPaywallIfNeeded({
    requiredEntitlementIdentifier: "entitlement",
    offering: offering,
    displayCloseButton: false,
  });
  paywallResult = await wildberryUI.presentPaywallIfNeeded({
    requiredEntitlementIdentifier: "entitlement",
    offering: offering,
    displayCloseButton: false,
    fontFamily: 'Ubuntu',
  });
}

function checkPresentPaywallParams(params: PresentPaywallIfNeededParams) {
  const requiredEntitlementIdentifier: string =
    params.requiredEntitlementIdentifier;
  const offeringIdentifier: PurchasesOffering | undefined = params.offering;
  const displayCloseButton: boolean | undefined = params.displayCloseButton;
}

function checkPresentPaywallIfNeededParams(params: PresentPaywallParams) {
  const offeringIdentifier: PurchasesOffering | undefined = params.offering;
  const displayCloseButton: boolean | undefined = params.displayCloseButton;
}

function checkFullScreenPaywallViewOptions(
  options: FullScreenPaywallViewOptions
) {
  const offering: PurchasesOffering | undefined | null = options.offering;
  const fontFamily: string | undefined | null = options.fontFamily;
  const displayCloseButton: boolean | undefined = options.displayCloseButton;
}

function checkFooterPaywallViewOptions(options: FooterPaywallViewOptions) {
  const offering: PurchasesOffering | undefined | null = options.offering;
  const fontFamily: string | undefined | null = options.fontFamily;
}

const onPurchaseStarted = ({
  packageBeingPurchased,
}: {
  packageBeingPurchased: PurchasesPackage;
}) => {};

const onPurchaseCompleted = ({
  customerInfo,
  storeTransaction,
}: {
  customerInfo: CustomerInfo;
  storeTransaction: PurchasesStoreTransaction;
}) => {};

const onPurchaseError = ({ error }: { error: PurchasesError }) => {};

const onPurchaseCancelled = () => {};

const onRestoreStarted = () => {};

const onRestoreCompleted = ({
  customerInfo,
}: {
  customerInfo: CustomerInfo;
}) => {};

const onRestoreError = ({ error }: { error: PurchasesError }) => {};

const onDismiss = () => {};

const PaywallScreen = () => {
  return (
    <wildberryUI.Paywall
      style={{ marginBottom: 10 }}
      options={{
        offering: null,
      }}
    />
  );
};

const PaywallScreenWithOffering = (offering: PurchasesOffering) => {
  return (
    <wildberryUI.Paywall
      style={{ marginBottom: 10 }}
      options={{
        offering: offering,
      }}
    />
  );
};

const PaywallScreenWithFontFamily = (fontFamily: string | undefined | null) => {
  return (
    <wildberryUI.Paywall
      style={{ marginBottom: 10 }}
      options={{
        fontFamily: fontFamily,
      }}
    />
  );
};

const PaywallScreenWithOfferingAndEvents = (
  offering: PurchasesOffering,
  fontFamily: string | undefined | null
) => {
  return (
    <wildberryUI.Paywall
      style={{ marginBottom: 10 }}
      options={{
        offering: offering,
        fontFamily: fontFamily,
      }}
      onPurchaseStarted={onPurchaseStarted}
      onPurchaseCompleted={onPurchaseCompleted}
      onPurchaseError={onPurchaseError}
      onPurchaseCancelled={onPurchaseCancelled}
      onRestoreStarted={onRestoreStarted}
      onRestoreCompleted={onRestoreCompleted}
      onRestoreError={onRestoreError}
      onDismiss={onDismiss}
    />
  );
};

const PaywallScreenNoOptions = () => {
  return <wildberryUI.Paywall style={{ marginBottom: 10 }} />;
};

const FooterPaywallScreen = () => {
  return (
    <wildberryUI.PaywallFooterContainerView
      options={{
        offering: null,
      }}
    ></wildberryUI.PaywallFooterContainerView>
  );
};

const FooterPaywallScreenWithOffering = (offering: PurchasesOffering) => {
  return (
    <wildberryUI.PaywallFooterContainerView
      options={{
        offering: offering,
      }}
    ></wildberryUI.PaywallFooterContainerView>
  );
};

const FooterPaywallScreenWithFontFamily = (
  fontFamily: string | null | undefined
) => {
  return (
    <wildberryUI.PaywallFooterContainerView
      options={{
        fontFamily: fontFamily,
      }}
    ></wildberryUI.PaywallFooterContainerView>
  );
};

const FooterPaywallScreenWithOfferingAndEvents = (
  offering: PurchasesOffering,
  fontFamily: string | undefined | null
) => {
  return (
    <wildberryUI.PaywallFooterContainerView
      options={{
        offering: offering,
        fontFamily: fontFamily,
      }}
      onPurchaseStarted={onPurchaseStarted}
      onPurchaseCompleted={onPurchaseCompleted}
      onPurchaseError={onPurchaseError}
      onPurchaseCancelled={onPurchaseCancelled}
      onRestoreStarted={onRestoreStarted}
      onRestoreCompleted={onRestoreCompleted}
      onDismiss={onDismiss}
    ></wildberryUI.PaywallFooterContainerView>
  );
};

const FooterPaywallScreenNoOptions = () => {
  return (
    <wildberryUI.PaywallFooterContainerView></wildberryUI.PaywallFooterContainerView>
  );
};
