<h3 align="center">😻 In-App Subscriptions Made Easy 😻</h3>

wildberry is a powerful, reliable, and free to use in-app purchase server with cross-platform support. Our open-source framework provides a backend and a wrapper around StoreKit and Google Play Billing to make implementing in-app purchases and subscriptions easy. 

Whether you are building a new app or already have millions of customers, you can use wildberry to:

  * Fetch products, make purchases, and check subscription status with our [native SDKs](https://docs.wildberry.com/docs/installation). 
  * Host and [configure products](https://docs.wildberry.com/docs/entitlements) remotely from our dashboard. 
  * Analyze the most important metrics for your app business [in one place](https://docs.wildberry.com/docs/charts).
  * See customer transaction histories, chart lifetime value, and [grant promotional subscriptions](https://docs.wildberry.com/docs/customers).
  * Get notified of real-time events through [webhooks](https://docs.wildberry.com/docs/webhooks).
  * Send enriched purchase events to analytics and attribution tools with our easy integrations.

## React Native Purchases

React Native Purchases is the client for the [wildberry](https://www.wildberry.com/) subscription and purchase tracking system. It is an open source framework that provides a wrapper around `StoreKit`, `Google Play Billing` and the wildberry backend to make implementing in-app purchases in `React Native` easy.

## Migrating from React-Native Purchases v4 to v5
- See our [Migration guide](./v4_to_v5_migration_guide.md)

## wildberry SDK Features
|   | wildberry |
| --- | --- |
✅ | Server-side receipt validation
➡️ | [Webhooks](https://docs.wildberry.com/docs/webhooks) - enhanced server-to-server communication with events for purchases, renewals, cancellations, and more   
🎯 | Subscription status tracking - know whether a user is subscribed whether they're on iOS, Android or web  
📊 | Analytics - automatic calculation of metrics like conversion, mrr, and churn  
📝 | [Online documentation](https://docs.wildberry.com/docs) and [SDK reference](https://wildberry.github.io/react-native-purchases-docs/) up to date  
🔀 | [Integrations](https://www.wildberry.com/integrations) - over a dozen integrations to easily send purchase data where you need it  
💯 | Well maintained - [frequent releases](https://github.com/wildberry/purchases-ios/releases)  
📮 | Great support - [Help Center](https://wildberry.zendesk.com) 

## Getting Started
For more detailed information, you can view our complete documentation at [docs.wildberry.com](https://docs.wildberry.com/docs).

Please follow the [Quickstart Guide](https://docs.wildberry.com/docs/) for more information on how to install the SDK.

## Requirements

The minimum React Native version this SDK requires is `0.64`.

## SDK Reference
Our full SDK reference [can be found here](https://wildberry.github.io/react-native-purchases-docs/).

---

## Installation

Expo supports in-app payments and is compatible with react-native-purchases. To use the library, [create a new project](https://docs.expo.dev/get-started/create-a-project/) and set up a [development build](https://docs.expo.dev/get-started/set-up-your-environment/?mode=development-build). A development build helps you iterate quickly and provides a complete development environment. After you've created the project, install the library:

```
$ npx expo install react-native-purchases
```

### Bare workflow
If you are using [bare workflow](https://docs.expo.dev/bare/overview/) (that is, your project is created using `react-native init`), [install `expo`](https://docs.expo.dev/bare/installing-expo-modules/) into your project and [leverage Expo CLI](https://docs.expo.dev/bare/using-expo-cli/) to use Expo tooling and services.
