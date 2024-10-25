// Copyright © 2024 Apple Inc. All Rights Reserved.

// APPLE INC.
// PRIVATE CLOUD COMPUTE SOURCE CODE INTERNAL USE LICENSE AGREEMENT
// PLEASE READ THE FOLLOWING PRIVATE CLOUD COMPUTE SOURCE CODE INTERNAL USE LICENSE AGREEMENT (“AGREEMENT”) CAREFULLY BEFORE DOWNLOADING OR USING THE APPLE SOFTWARE ACCOMPANYING THIS AGREEMENT(AS DEFINED BELOW). BY DOWNLOADING OR USING THE APPLE SOFTWARE, YOU ARE AGREEING TO BE BOUND BY THE TERMS OF THIS AGREEMENT. IF YOU DO NOT AGREE TO THE TERMS OF THIS AGREEMENT, DO NOT DOWNLOAD OR USE THE APPLE SOFTWARE. THESE TERMS AND CONDITIONS CONSTITUTE A LEGAL AGREEMENT BETWEEN YOU AND APPLE.
// IMPORTANT NOTE: BY DOWNLOADING OR USING THE APPLE SOFTWARE, YOU ARE AGREEING ON YOUR OWN BEHALF AND/OR ON BEHALF OF YOUR COMPANY OR ORGANIZATION TO THE TERMS OF THIS AGREEMENT.
// 1. As used in this Agreement, the term “Apple Software” collectively means and includes all of the Apple Private Cloud Compute materials provided by Apple here, including but not limited to the Apple Private Cloud Compute software, tools, data, files, frameworks, libraries, documentation, logs and other Apple-created materials. In consideration for your agreement to abide by the following terms, conditioned upon your compliance with these terms and subject to these terms, Apple grants you, for a period of ninety (90) days from the date you download the Apple Software, a limited, non-exclusive, non-sublicensable license under Apple’s copyrights in the Apple Software to download, install, compile and run the Apple Software internally within your organization only on a single Apple-branded computer you own or control, for the sole purpose of verifying the security and privacy characteristics of Apple Private Cloud Compute. This Agreement does not allow the Apple Software to exist on more than one Apple-branded computer at a time, and you may not distribute or make the Apple Software available over a network where it could be used by multiple devices at the same time. You may not, directly or indirectly, redistribute the Apple Software or any portions thereof. The Apple Software is only licensed and intended for use as expressly stated above and may not be used for other purposes or in other contexts without Apple's prior written permission. Except as expressly stated in this notice, no other rights or licenses, express or implied, are granted by Apple herein.
// 2. The Apple Software is provided by Apple on an "AS IS" basis. APPLE MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS, SYSTEMS, OR SERVICES. APPLE DOES NOT WARRANT THAT THE APPLE SOFTWARE WILL MEET YOUR REQUIREMENTS, THAT THE OPERATION OF THE APPLE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, THAT DEFECTS IN THE APPLE SOFTWARE WILL BE CORRECTED, OR THAT THE APPLE SOFTWARE WILL BE COMPATIBLE WITH FUTURE APPLE PRODUCTS, SOFTWARE OR SERVICES. NO ORAL OR WRITTEN INFORMATION OR ADVICE GIVEN BY APPLE OR AN APPLE AUTHORIZED REPRESENTATIVE WILL CREATE A WARRANTY.
// 3. IN NO EVENT SHALL APPLE BE LIABLE FOR ANY DIRECT, SPECIAL, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, COMPILATION OR OPERATION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 4. This Agreement is effective until terminated. Your rights under this Agreement will terminate automatically without notice from Apple if you fail to comply with any term(s) of this Agreement. Upon termination, you agree to cease all use of the Apple Software and destroy all copies, full or partial, of the Apple Software. This Agreement constitutes the entire understanding of the parties with respect to the subject matter contained herein, and supersedes all prior negotiations, representations, or understandings, written or oral. This Agreement will be governed and construed in accordance with the laws of the State of California, without regard to its choice of law rules.
// You may report security issues about Apple products to product-security@apple.com, as described here: https://www.apple.com/support/security/. Non-security bugs and enhancement requests can be made via https://bugreport.apple.com as described here: https://developer.apple.com/bug-reporting/
// EA1937
// 10/02/2024

//  Copyright © 2023 Apple Inc. All rights reserved.

import Foundation
import os
internal import CloudBoardConfigurationDAPI
import CFPreferenceCoder
internal import CloudBoardLogging

let preferenceUpdatesLogger: os.Logger = .init(
    subsystem: "com.apple.cloudos.cloudboard",
    category: "PreferencesUpdates"
)

/// An async sequence of updates to the preferences in the provided domain coming from the preferences service.
///
/// Use this type to subscribe to preferences updates that your app can change without a restart.
///
/// ## Fallback to local preferences
/// When the preferences service is unavailable, the system loads the preferences from the matching local domain.
///
/// ## Decoding your preferences
///
/// The ``PreferencesUpdates`` async sequence decodes the preferences into a preferences type that you provide.
///
/// Your preferences type must conform to the `Decodable`, `Sendable`, and `Equatable` protocols, so that the sequence
/// can transparently decode your preferences from the underlying serialization format and deduplicate identical
/// updates.
///
/// For example, for `ExampleApp`, the domain could be called `com.apple.cloudos.hotproperties.ExampleApp`, and
/// the preferences type would be defined as follows:
///
/// ```swift
/// struct ExampleAppPreferences: Decodable, Sendable, Hashable {
///     var greetingTemplateString: String?
///     var maxSizeInBytes: Double?
/// }
/// ```
///
/// > Note: It is recommended that you define preferences properties as optional, so that running your app does
/// not require either a fully defined preferences in the upstream preferences service, or in local preferences.
/// However, if your app does not have a reasonable default for a preferences property, define the property
/// as required and ensure that your local preferences domain has a fallback value set, and that you also set
/// that value in the upstream preferences service before deploying your code.
///
/// ## Subscribing for updates
///
/// Create a ``PreferencesUpdates`` async sequence for your hot properties preferences domain.
///
/// ```swift
/// let preferencesUpdates = PreferencesUpdates(
///     preferencesDomain: "com.apple.cloudos.hotproperties.ExampleApp",
///     maximumUpdateDuration: .seconds(15),
///     forType: ExampleAppPreferences.self
/// )
/// ```
///
/// Then start observing updates on the async sequence. Note that the first update might take over a minute to arrive,
/// in cases when the system reaching out to the preferences service hits delays or retries.
///
/// ```swift
/// for try await update in preferencesUpdates {
///     await update.applyingPreferences { newPreferences in
///         print("Received an update: \(newPreferences)")
///         // Update ExampleApp's internal state to use the new preferences.
///     }
/// }
/// ```
///
/// ## Confirming an update within the timeout
///
/// When you receive an update value from the async sequence, you must acknowledge that your app has
/// applied the new preferences by calling ``PreferencesUpdate/successfullyApplied()``, when your app successfully
/// applied the new preferences, or ``PreferencesUpdate/failedToApply(error:)``, when your app failed to update
/// itself.
///
/// You must provide this reply within the time duration you provided in `maximumUpdateDuration`, so set that value to
/// the upper bound of how long your app, when functioning correctly, can take to apply the updated preferences.
///
/// For convenience, you can use the ``PreferencesUpdate/applyingPreferences(_:)`` method, which takes a closure and
/// ensures that one of the confirmation methods is called on all code paths, interpreting thrown errors as a failure to
/// apply.
///
/// > Warning: If you fail to provide a response within `maximumUpdateDuration`, the async sequence will throw an error
/// and will tear itself down, and you will not receive further updates.
///
/// ## Treating the first value specially
///
/// If your app requires waiting for the first value on startup, since the ``PreferencesUpdates`` type is an async
/// sequence, you can create the iterator explicitly, wait for the first value, use that for the initial bootstrapping,
/// and later subscribe for subsequent updates.
///
/// ```swift
/// // Create the iterator explicitly
/// var iterator = preferencesUpdates.makeAsyncIterator()
///
/// // Use the first value.
/// let initialValue = try await iterator.next()!
/// await initialValue.applyingPreferences { value in
///    print("Initial value: \(value)")
/// }
///
/// // Handle subsequent preferences updates.
/// while let update = try await iterator.next() {
///    await initialValue.applyingPreferences { value in
///        print("Received an update: \(value)")
///    }
/// }
/// ```
///
/// > Important: The `maximumUpdateDuration` timeout is measured from the moment that the update is received by the
/// system, not from the moment it is received by your code from the iterator, so make sure to to subscribe for
/// subsequent updates without needless delay, or extend `maximumUpdateDuration`.
public struct PreferencesUpdates<PreferencesType: Decodable & Equatable & Sendable>: Sendable {
    private let _makeAsyncIterator: @Sendable () -> AsyncIterator

    /// Creates a new preferences updates async sequence.
    /// - Parameters:
    ///   - preferencesDomain: The preferences domain to watch, for example
    /// `com.apple.cloudos.hotproperties.ExampleApp`.
    ///   - maximumUpdateDuration: The maximum amount of time that the system should wait for a reply from your app
    ///     to a preferences update before timing out.
    ///   - preferencesType: The type of your preferences.
    public init(
        preferencesDomain: String,
        maximumUpdateDuration: Duration,
        forType _: PreferencesType.Type = PreferencesType.self
    ) {
        preferenceUpdatesLogger
            .info(
                "Subscription request for \(preferencesDomain, privacy: .public) with timeout \(maximumUpdateDuration.components.seconds, privacy: .public) seconds received."
            )
        self.init(makeIterator: {
            UpdateStreamIterator(
                of: PreferencesType.self,
                inDomain: preferencesDomain,
                preferences: CFPreferences(domain: preferencesDomain),
                maximumUpdateDuration: maximumUpdateDuration,
                makeUnreliableConnection: ConfigurationAPIXPCClient.localConnection
            )
        })
    }

    internal init(
        makeIterator: @escaping () -> UpdateStreamIterator<
            PreferencesType,
            AsyncThrowingStream<Subscription.Event, Error>
        >
    ) {
        self._makeAsyncIterator = {
            var iterator = makeIterator()
            preferenceUpdatesLogger
                .debug(
                    "Created a new iterator for \(iterator.domain, privacy: .public)."
                )
            return AsyncIterator(
                wrapping: {
                    do {
                        let value = try await iterator.next()
                        if let value {
                            preferenceUpdatesLogger
                                .debug(
                                    "Subscription for \(iterator.domain, privacy: .public) is emitting the value of: \(String(describing: value), privacy: .public)."
                                )
                        } else {
                            preferenceUpdatesLogger
                                .debug(
                                    "Subscription for \(iterator.domain, privacy: .public) completed by returning nil."
                                )
                        }
                        return value
                    } catch let error as CancellationError {
                        preferenceUpdatesLogger
                            .warning("Subscription for \(iterator.domain, privacy: .public) got canceled.")
                        throw error
                    } catch {
                        preferenceUpdatesLogger.warning(
                            "Subscription for \(iterator.domain, privacy: .public) threw an error: \(String(reportable: error), privacy: .public) (\(error))"
                        )
                        throw error
                    }
                },
                start: iterator.onFirstNextCall
            )
        }
    }
}

extension PreferencesUpdates: AsyncSequence {
    /// The element type of the async sequence.
    public typealias Element = PreferencesUpdate<PreferencesType>

    /// Creates an async iterator that emits elements of this async sequence.
    public func makeAsyncIterator() -> AsyncIterator {
        return self._makeAsyncIterator()
    }

    /// The async iterator for this async sequence.
    public struct AsyncIterator: AsyncIteratorProtocol {
        private var iterator: () async throws -> Element?
        private var start: (() async -> Void)?

        init(
            wrapping iterator: @escaping () async throws -> Element?,
            start: @escaping () async -> Void
        ) {
            self.iterator = iterator
            self.start = start
        }

        /// Advance the iterator to the next element.
        ///
        /// - Returns: The next element or `nil` if the iterator is exhausted.
        /// - Throws: If the iterator encounters an error or `maximumUpdateDuration` was reached.
        public mutating func next() async throws -> Element? {
            if let start {
                await start()
                self.start = nil
            }
            guard let element = try await self.iterator() else {
                return nil
            }
            return element
        }
    }
}

@available(*, unavailable)
extension PreferencesUpdates.AsyncIterator: Sendable {}

/// A single update to the preferences.
///
/// Either ``PreferencesUpdate/successfullyApplied()`` or ``PreferencesUpdate/failedToApply(error:)`` must be invoked
/// within `maximumUpdateDuration`, otherwise the preferences sequence is invalidated and throws an error.
///
/// As a convenience, you can use ``PreferencesUpdate/applyingPreferences(_:)`` that ensures exactly one reply is called
/// on all codepaths. A thrown error inside the sequence is interpreted as a failure to update to the new preferences.
public struct PreferencesUpdate<Preferences>: Sendable where Preferences: Decodable & Equatable & Sendable {
    private let reply: (revisionIdentifier: String, callback: @Sendable (String, Result<Void, Error>) async -> Void)?

    internal init(
        newValue: Preferences,
        reply: (String, @Sendable (String, Result<Void, Error>) async -> Void)?
    ) {
        self.newValue = newValue
        self.reply = reply
    }

    /// The new value of the preferences.
    public let newValue: Preferences

    /// A convenience method that allows a scoped application of preferences.
    ///
    /// This can be used to ensure that the preference update always has a result reported.
    ///
    /// When using this method, do not call ``PreferencesUpdate/applyingPreferences(_:)`` or
    /// ``PreferencesUpdate/applyingPreferences(_:)``, it will be called for you once your closure returns or throws.
    ///
    /// > Important: The provided closure must return (or throw) within `maximumUpdateDuration`.
    public func applyingPreferences<ReturnType>(
        _ body: (Preferences) async throws -> ReturnType
    ) async rethrows -> ReturnType {
        do {
            let returnValue = try await body(newValue)
            await successfullyApplied()
            return returnValue
        } catch {
            await self.failedToApply(error: error)
            throw error
        }
    }

    /// Called to mark the preferences as successfully applied.
    ///
    /// Prefer to use ``PreferencesUpdate/applyingPreferences(_:)`` instead.
    ///
    /// > Important: The method must be called within `maximumUpdateDuration`.
    public func successfullyApplied() async {
        guard let reply else {
            return
        }
        await reply.callback(reply.revisionIdentifier, .success(()))
    }

    /// Called to mark the preferences as failed to apply.
    ///
    /// Prefer to use ``PreferencesUpdate/applyingPreferences(_:)`` instead.
    ///
    /// > Important: The method must be called within `maximumUpdateDuration`.
    public func failedToApply(error: any Error) async {
        guard let reply else {
            return
        }
        await reply.callback(reply.revisionIdentifier, .failure(error))
    }
}

/// Information about the current preferences version.
public struct PreferencesVersionInfo: Sendable {
    /// An enumeration that represents a version of a project.
    public enum Version: Sendable {
        /// Represents a concrete revision.
        case revision(String)

        /// Represents the fallback.
        case fallback
    }

    /// The last successfully applied version.
    public var appliedVersion: Version?

    /// Creates a new version info value.
    /// - Parameters:
    ///   - appliedVersion: The last successfully applied version.
    public init(
        appliedVersion: Version?
    ) {
        self.appliedVersion = appliedVersion
    }
}

extension PreferencesVersionInfo.Version: CustomStringConvertible {
    public var description: String {
        switch self {
        case .revision(let revision):
            return revision
        case .fallback:
            return "fallback"
        }
    }
}

extension PreferencesVersionInfo {
    /// Provides the current version information from the preferences daemon.
    public static var current: Self {
        get async {
            await withLogging(
                operation: "currentConfigurationVersionInfo",
                logger: preferenceUpdatesLogger
            ) {
                for attempt in 1 ... 3 {
                    let connection = await ConfigurationAPIXPCClient.localConnection()
                    await connection.connect()
                    do {
                        let info = try await connection.currentConfigurationVersionInfo()
                        let outInfo: PreferencesVersionInfo = .init(info)
                        await connection.disconnect()
                        return outInfo
                    } catch {
                        preferenceUpdatesLogger
                            .warning(
                                "[Attempt: \(attempt, privacy: .public)] Failed to fetch current configuration version info. Error: \(String(describing: error), privacy: .public)"
                            )
                        await connection.disconnect()
                    }
                }
                return .init(appliedVersion: nil)
            }
        }
    }
}

extension PreferencesVersionInfo {
    fileprivate init(_ info: ConfigurationVersionInfo) {
        self.init(appliedVersion: info.appliedVersion.flatMap { .init($0) })
    }
}

extension PreferencesVersionInfo.Version {
    fileprivate init(_ version: ConfigurationVersionInfo.Version) {
        switch version {
        case .revision(let revision):
            self = .revision(revision)
        case .fallback:
            self = .fallback
        }
    }
}
