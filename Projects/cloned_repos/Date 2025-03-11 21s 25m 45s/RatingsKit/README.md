# RatingsKit

A Swift Package that provides an elegant and customizable ratings interface for your iOS and macOS applications.

<img src="https://github.com/user-attachments/assets/da8bdd08-5956-40aa-88b2-0d5698d8dac6" width="600">

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Swift Package Manager](#swift-package-manager)
- [Usage](#usage)
  - [Basic Implementation](#basic-implementation)
  - [Creating a Custom Rating Provider](#creating-a-custom-rating-provider)
  - [Using the Mock Provider](#using-the-mock-provider)
  - [Using with Supabase and App Store Connect API](#using-with-supabase-and-app-store-connect-api)
- [Customization](#customization)
- [Contributing](#contributing)

## Overview

RatingsKit makes it easy to add a ratings and review system to your application. It offers a complete UI for displaying app ratings and reviews, and provides a streamlined way for users to submit their ratings to the App Store.

## Features

- **Rating Request Screen**: A polished UI that displays app ratings, reviews, and encourages users to rate your app
- **Customizable Provider**: Easy integration with any data source through the `AppRatingProviding` protocol
- **Mock Implementation**: Ready-to-use mock data provider for development and preview purposes
- **Error Handling**: Built-in error states with customizable error handling
- **SwiftUI Design**: Modern UI built with SwiftUI for iOS 17+ and macOS 14+

## Requirements

- iOS 17.0+ / macOS 14.0+
- Swift 6.0+
- Xcode 15+

## Installation

### Swift Package Manager

To integrate RatingsKit into your Xcode project using Swift Package Manager, add it to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/Sedlacek-Solutions/RatingsKit.git", from: "1.0.0")
]
```

Or add it directly via Xcode's Package Manager integration.

## Usage

### Basic Implementation

```swift
import SwiftUI
import RatingsKit

struct RatingsView: View {
    var body: some View {
        RatingRequestScreen(
            appId: "YOUR_APP_ID",
            appRatingProvider: YourAppRatingProvider(),
            requestedRatingAction: {
                // Handle when user has requested to leave a rating
                print("User tapped to leave a rating")
            },
            maybeLaterAction: {
                // Handle when user decides to rate later
                print("User will rate later")
            },
            onError: { error in
                // Handle any errors that occur
                print("Error occurred: \(error)")
            }
        )
    }
}
```

### Creating a Custom Rating Provider

Implement the `AppRatingProviding` protocol to fetch ratings and reviews from your backend service:

```swift
struct YourAppRatingProvider: AppRatingProviding {
    func fetch() async throws -> AppRatingResponse {
        // Implement your logic to fetch ratings from your backend
        // For example:
        let (data, _) = try await URLSession.shared.data(from: yourBackendURL)
        return try JSONDecoder().decode(AppRatingResponse.self, from: data)
    }
}
```

### Using the Mock Provider

RatingsKit comes with a `MockAppRatingProvider` for testing and UI development:

```swift
RatingRequestScreen(
    appId: "YOUR_APP_ID",
    appRatingProvider: MockAppRatingProvider.withMockReviews
)
```

Available mock configurations:
- `.withMockReviews`: Provides a set of sample reviews
- `.noReviews`: Shows ratings without reviews
- `.noRatingsOrReviews`: Empty state with no ratings or reviews
- `.throwsError`: Simulates an error state

### Using with Supabase and App Store Connect API

For detailed instructions on how to create a Supabase Edge Function that fetches data from the App Store Connect API for use with RatingsKit, see our [Supabase Implementation Guide](SUPABASE-GUIDE.md).

## Customization

RatingsKit's UI is built with SwiftUI and follows system styling, making it blend seamlessly with your app. The interface automatically adapts to light and dark mode and respects system accessibility settings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
