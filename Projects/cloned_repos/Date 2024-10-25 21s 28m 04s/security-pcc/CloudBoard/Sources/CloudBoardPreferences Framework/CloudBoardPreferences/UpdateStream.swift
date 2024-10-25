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

//  Copyright © 2024 Apple Inc. All rights reserved.

import CFPreferenceCoder
import Foundation
import os
internal import CloudBoardConfigurationDAPI

/// A delegate of the update stream.
protocol UpdateStreamDelegate: AnyObject, Sendable {
    /// Notifies the delegate that the subscriber successfully applied the provided configuration.
    func successfullyAppliedConfiguration(revisionIdentifier: String) async

    /// Notifies the delegate that the subscriber failed to apply the provided configuration.
    ///
    /// This includes timeouts.
    func failedToApplyConfiguration(revisionIdentifier: String) async
}

private let logger: os.Logger = .init(
    subsystem: "com.apple.cloudos.cloudboard",
    category: "UpdateStream"
)

/// An iterator of preference updates coming from the daemon.
struct UpdateStreamIterator<
    PreferencesType: Decodable & Equatable & Sendable,
    EventStreamType: AsyncSequence
> where EventStreamType.Element == Subscription.Event {
    /// An error thrown when the subscriber doesn't reply in time.
    struct TimeoutError: Error, LocalizedError, CustomStringConvertible {
        /// The duration the update stream waited for.
        var duration: Duration

        /// The preferences domain that was watched.
        var domain: String

        var description: String {
            "Timed out after not receiving a reply about configuration \(self.domain) for \(self.duration.components.seconds) seconds."
        }
    }

    /// The delegate of the stream.
    private let delegate: UpdateStreamDelegate

    /// The preferences domain to watch.
    let domain: String

    /// The preferences storage to read from.
    private let preferences: CFPreferences

    /// The maximum allowed time to wait for a reply from the subscriber after emitting an update.
    private let maximumUpdateDuration: Duration

    /// A closure called on the first `next` call, to start the subscription.
    private let _onFirstNextCall: () async -> Void

    /// The upstream stream iterator.
    private var upstreamIterator: EventStreamType.AsyncIterator

    /// The previous emitted element, used to avoid subsequent duplicate updates.
    private var previousElement: SubscriptionUpdate<PreferencesType>?

    /// Creates a new update stream iterator with an underlying subscription.
    /// - Parameters:
    ///   - type: The type of the preferences object.
    ///   - domain: The preferences domain to watch.
    ///   - preferences: The preferences storage to read from.
    ///   - maximumUpdateDuration: The maximum allowed time to wait for a reply from the subscriber after emitting
    ///     an update.
    ///   - makeUnreliableConnection: The factory closure to create a new unreliable connection to the daemon.
    init(
        of type: PreferencesType.Type = PreferencesType.self,
        inDomain domain: String,
        preferences: CFPreferences,
        maximumUpdateDuration: Duration,
        makeUnreliableConnection: @escaping () async -> ConfigurationAPIXPCClient
    ) where EventStreamType == AsyncThrowingStream<Subscription.Event, Error> {
        let subscription = Subscription(
            domain: domain,
            makeUnreliableConnection: makeUnreliableConnection
        )
        self.init(
            of: type,
            inDomain: domain,
            preferences: preferences,
            maximumUpdateDuration: maximumUpdateDuration,
            delegate: subscription,
            eventStream: subscription.eventStream,
            onFirstNextCall: {
                await subscription.start()
            }
        )
    }

    /// Creates a new update stream iterator.
    /// - Parameters:
    ///   - type: The type of the preferences object.
    ///   - domain: The preferences domain to watch.
    ///   - preferences: The preferences storage to read from.
    ///   - maximumUpdateDuration: The maximum allowed time to wait for a reply from the subscriber after emitting
    ///     an update.
    ///   - delegate: The update stream delegate.
    ///   - eventStream: The stream to watch for events from the daemon.
    ///   - onFirstNextCall: A closure called on the first `next` call.
    init(
        of _: PreferencesType.Type = PreferencesType.self,
        inDomain domain: String,
        preferences: CFPreferences,
        maximumUpdateDuration: Duration,
        delegate: UpdateStreamDelegate,
        eventStream: EventStreamType,
        onFirstNextCall: @Sendable @escaping () async -> Void
    ) where EventStreamType.Element == Subscription.Event {
        self.domain = domain
        self.preferences = preferences
        self.maximumUpdateDuration = maximumUpdateDuration
        self.delegate = delegate
        self._onFirstNextCall = onFirstNextCall
        self.upstreamIterator = eventStream.makeAsyncIterator()
    }

    /// Produces the next update.
    mutating func next() async throws -> PreferencesUpdate<PreferencesType>? {
        try await self.nextPublicUpdate()
    }

    /// Calls the first `next` call closure, generally used to start the underlying connection.
    func onFirstNextCall() async {
        await self._onFirstNextCall()
    }

    /// Produces the next subscription update.
    private mutating func nextSubscriptionUpdate() async throws -> SubscriptionUpdate<PreferencesType>? {
        guard let event = try await upstreamIterator.next() else {
            return nil
        }
        return try self.updateFromEvent(event)
    }

    /// Produces the next deduplicated update.
    private mutating func nextDeduplicatedUpdate() async throws -> SubscriptionUpdate<PreferencesType>? {
        while let upstreamElement = try await nextSubscriptionUpdate() {
            guard self.previousElement?.preferences == upstreamElement.preferences else {
                self.previousElement = upstreamElement
                return upstreamElement
            }
            await self.handleDiscardedElement(update: upstreamElement)
        }
        return nil
    }

    /// Produces the next public update.
    private mutating func nextPublicUpdate() async throws -> PreferencesUpdate<PreferencesType>? {
        guard let update = try await nextDeduplicatedUpdate() else {
            return nil
        }
        return PreferencesUpdate(
            newValue: update.preferences,
            reply: update.replyInfo.map { [self] reply in
                (reply.revisionIdentifier, { revisionIdentifier, result in
                    reply.cancellationTask.cancel()
                    await self.handleReply(revisionIdentifier: revisionIdentifier, result: result)
                })
            }
        )
    }

    /// Handles a reply result from the subscriber.
    /// - Parameters:
    ///   - revisionIdentifier: The revision identifier of the configuration.
    ///   - result: The result representing the reply.
    private func handleReply(
        revisionIdentifier: String,
        result: Result<Void, Error>
    ) async {
        logger.info(
            "Subscription for \(revisionIdentifier, privacy: .public) received a reply: \(String(describing: result), privacy: .public)."
        )
        switch result {
        case .success:
            logger.info("Subscriber successfully applied config \(revisionIdentifier, privacy: .public).")
            await self.delegate.successfullyAppliedConfiguration(revisionIdentifier: revisionIdentifier)
        case .failure(let error):
            logger.error(
                "Subscriber failed to apply config \(revisionIdentifier, privacy: .public) with error: \(String(reportable: error), privacy: .public) (\(error))"
            )
            await self.delegate.failedToApplyConfiguration(revisionIdentifier: revisionIdentifier)
        }
    }

    /// Handles a timeout, emitted when the subscriber failed to reply within the timeout.
    /// - Parameters:
    ///   - revisionIdentifier: The revision identifier of the configuration.
    ///   - maximumUpdateDuration: The timeout the iterator waited for.
    ///   - domain: The preferences domain watched.
    private func handleTimeout(
        revisionIdentifier: String,
        maximumUpdateDuration: Duration,
        domain: String
    ) async {
        let error = TimeoutError(duration: maximumUpdateDuration, domain: domain)
        await handleReply(revisionIdentifier: revisionIdentifier, result: .failure(error))
    }

    /// Handles when an update is identical to the previous one, thus skipped and not delivered to the subscriber.
    /// - Parameter update: The update value.
    private func handleDiscardedElement(
        update: SubscriptionUpdate<PreferencesType>
    ) async {
        logger.debug("Discarding a duplicate update: \(String(describing: update), privacy: .public)")
        guard let replyInfo = update.replyInfo else {
            return
        }
        replyInfo.cancellationTask.cancel()
        await self.handleReply(
            revisionIdentifier: replyInfo.revisionIdentifier,
            result: .success(())
        )
    }

    /// Produces a subscription update for the provided event.
    /// - Parameter event: An event from the stream.
    /// - Returns: A matching subscription update.
    private func updateFromEvent(_ event: Subscription.Event) throws -> SubscriptionUpdate<PreferencesType> {
        switch event {
        case .fallback:
            return try .init(
                preferences: Self.loadPreferencesFromCFPrefs(
                    domain: self.domain,
                    preferences: self.preferences
                ),
                replyInfo: nil
            )
        case .configuration(let config, shouldReply: let shouldReply):
            let revisionIdentifier = config.revisionIdentifier
            let replyInfo: SubscriptionUpdate<PreferencesType>.ReplyInfo? = if shouldReply {
                .init(
                    revisionIdentifier: revisionIdentifier,
                    cancellationTask: Task.detached { [self] in
                        do {
                            try await Task.sleep(for: self.maximumUpdateDuration)
                            await self.handleTimeout(
                                revisionIdentifier: revisionIdentifier,
                                maximumUpdateDuration: self.maximumUpdateDuration,
                                domain: self.domain
                            )
                        } catch {}
                    }
                )
            } else {
                nil
            }
            return try .init(
                preferences: Self.decodePreferencesFromJSON(data: config.contentsJSON),
                replyInfo: replyInfo
            )
        }
    }

    /// Loads preferences from a local store.
    /// - Parameter domain: The preferences domain.
    /// - Returns: The decoded preferences value.
    private static func loadPreferencesFromCFPrefs<Preferences>(
        domain _: String,
        preferences: CFPreferences
    ) throws -> Preferences where Preferences: Decodable & Equatable & Sendable {
        let value = try CFPreferenceDecoder().decode(Preferences.self, from: preferences)
        return value
    }

    /// Parses preferences from the provided JSON data.
    /// - Parameter data: The JSON data.
    /// - Returns: The decoded preferences value.
    private static func decodePreferencesFromJSON<Preferences>(
        data: Data
    ) throws -> Preferences where Preferences: Decodable & Equatable & Sendable {
        try JSONDecoder().decode(Preferences.self, from: data)
    }
}

/// An update to the preferences value.
struct SubscriptionUpdate<Preferences>: Equatable where Preferences: Decodable & Equatable & Sendable {
    /// The preferences value.
    let preferences: Preferences

    /// Metadata about a reply from the subscriber.
    struct ReplyInfo: Hashable {
        /// The revision identifier of the configuration.
        var revisionIdentifier: String

        /// The task that enforces cancellation within the timeout.
        var cancellationTask: Task<Void, Never>
    }

    /// Reply metadata, only provided for updates that require a reply.
    let replyInfo: ReplyInfo?
}
