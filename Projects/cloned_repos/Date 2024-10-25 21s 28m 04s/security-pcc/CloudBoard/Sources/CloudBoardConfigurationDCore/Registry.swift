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

// Copyright © 2024 Apple. All rights reserved.

import CloudBoardAsyncXPC
import CloudBoardCommon
import CloudBoardConfigurationDAPI
import CloudBoardLogging
import CloudBoardMetrics
import Foundation
import os

/// An actor that keeps track of client connections and informs them of configuration changes.
actor Registry {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "Registry"
    )

    /// The state of the configuration for a node.
    enum NodeConfigurationState: Hashable {
        /// No configuration applied yet.
        case none

        /// The fallback configuration was applied.
        case fallback

        /// A configuration package was applied.
        case package(NodeConfigurationPackage)
    }

    /// The state of the configuration for a domain.
    enum DomainConfigurationState: Hashable {
        /// No configuration applied yet.
        case none

        /// The fallback configuration was applied.
        case fallback

        /// A configuration package was applied.
        case package(DomainConfigurationPackage)
    }

    /// An update to the configuration state.
    enum DomainConfigurationUpdate: Hashable {
        /// The configuration is up-to-date.
        case upToDate

        /// The fallback configuration should be applied.
        case applyFallback

        /// The provided configuration should be applied.
        case applyConfiguration(DomainConfigurationPackage)
    }

    /// The metadata about an active connection.
    struct Connection: Sendable {
        /// The underlying client.
        var connection: ConfigurationAPIServerToClientConnection

        /// The identifier of the connection.
        var id: ConnectionID {
            self.connection.id
        }

        /// The domain for which the connection is registered.
        var domain: String

        /// The configuration present on the client, to the best of our knowledge.
        var currentConfiguration: ConfigurationInfo
    }

    /// A set of connections.
    typealias Connections = [ConnectionID: Connection]

    /// The state machine that drives the connection registry.
    private(set) var stateMachine: RegistryStateMachine {
        willSet {
            let oldState = self.stateMachine.state
            let newState = newValue.state
            guard oldState != newState else {
                return
            }
            Self.logger
                .debug(
                    "State machine transitionining from: \(oldState, privacy: .public) to: \(newState, privacy: .public)."
                )
            self.updateMetrics()
        }
    }

    /// The metrics to use.
    private let metrics: MetricsSystem

    /// Update metrics with the current state.
    private func updateMetrics() {
        self.metrics.emit(Metrics.Registry.ConnectionsGauge(value: Double(self.stateMachine.connectionCount)))
    }

    /// Creates a new registry.
    /// - Parameter metrics: The metrics to use.
    init(metrics: MetricsSystem) {
        self.stateMachine = .init()
        self.metrics = metrics
    }
}

extension Registry: FetcherDelegate {
    func applyConfiguration(_ configuration: NodeConfigurationPackage) async throws {
        try await withLogging(
            operation: "applyConfiguration \(configuration.revisionIdentifier)",
            sensitiveError: false,
            logger: Self.logger
        ) {
            let startInstant = ContinuousClock.now
            let promise = Promise<Void, RegistryStateMachine.StringError>()
            switch self.stateMachine.applyConfiguration(
                configuration: configuration,
                completion: promise
            ) {
            case .invokeCompletionWithSuccess(let promise, let upToDateConnectionCount):
                Self.logger
                    .debug(
                        "invokeCompletionWithSuccess (while \(upToDateConnectionCount, privacy: .public) connections are already up-to-date)"
                    )
                promise.succeed()
            case .send(
                let revisionIdentifier,
                let connectionsToUpdate,
                let upToDateConnectionCount
            ):
                let countConnectionsToUpdate = connectionsToUpdate.count
                Self.logger
                    .debug(
                        "Send config \(revisionIdentifier, privacy: .public) to \(countConnectionsToUpdate, privacy: .public) connections, while \(upToDateConnectionCount, privacy: .public) connections are already up-to-date."
                    )
                self.metrics.emit(
                    Metrics.Registry.ApplyConfigToConnectionCounter(
                        action: .increment(by: countConnectionsToUpdate)
                    )
                )
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for (domainConfiguration, connection) in connectionsToUpdate {
                        group.addTask {
                            let unappliedConfiguration = try UnappliedConfiguration(domainConfiguration)
                            try await connection.applyConfiguration(unappliedConfiguration)
                        }
                        try await group.waitForAll()
                    }
                }
            }
            let future = Future(promise)
            do {
                try await future.valueWithCancellation
                self.metrics.emit(Metrics.Registry.ApplyingSuccessHistogram(durationSinceStart: startInstant))
            } catch {
                self.metrics.emit(Metrics.Registry.ApplyingFailureHistogram(durationSinceStart: startInstant))
                throw error
            }
        }
    }

    func applyFallback() async throws {
        try await withLogging(operation: "applyFallback", sensitiveError: false, logger: Self.logger) {
            switch self.stateMachine.applyFallback() {
            case .send(let connections):
                let count = connections.count
                Self.logger.debug("send fallback to \(count, privacy: .public) connections")
                self.metrics.emit(Metrics.Registry.ApplyFallbackToConnectionCounter(action: .increment(by: count)))
                try await withThrowingTaskGroup(of: Void.self) { group in
                    for connection in connections {
                        group.addTask {
                            try await connection.applyFallback(.init())
                        }
                        try await group.waitForAll()
                    }
                }
            }
        }
    }
}

extension Registry: ConfigurationServerDelegate {
    func register(
        connection: ConfigurationAPIServerToClientConnection,
        domain: String,
        currentConfiguration: ConfigurationInfo
    ) async throws -> ConfigurationUpdate {
        try await withLogging(
            operation: "register \(domain) on connection \(connection.id) with \(currentConfiguration)",
            sensitiveError: false,
            logger: Self.logger
        ) {
            self.metrics.emit(Metrics.Registry.NewConnectionCounter(action: .increment))
            switch self.stateMachine.register(
                connection: connection,
                domain: domain,
                currentConfiguration: currentConfiguration
            ) {
            case .reply(let update):
                Self.logger
                    .info(
                        "register \(domain, privacy: .public) with \(currentConfiguration, privacy: .public) replied with reply \(update, privacy: .public)"
                    )
                switch update {
                case .upToDate:
                    return .upToDate
                case .applyFallback:
                    return .applyFallback
                case .applyConfiguration(let domainConfiguration):
                    let unappliedConfiguration = try UnappliedConfiguration(domainConfiguration)
                    return .applyConfiguration(unappliedConfiguration)
                }
            }
        }
    }

    func disconnected(connectionID: ConnectionID) async {
        await withLogging(operation: "disconnected \(connectionID)", sensitiveError: false, logger: Self.logger) {
            self.metrics.emit(Metrics.Registry.DisconnectCounter(action: .increment))
            switch self.stateMachine.disconnected(connectionID: connectionID) {
            case .reportDisconnectOneConnection(id: let id):
                Self.logger.debug("reportDisconnectOneConnection \(id, privacy: .public)")
            case .reportDisconnectedUnknownConnection(id: let id):
                Self.logger.debug("reportDisconnectedUnknownConnection \(id, privacy: .public)")
            case .invokeCompletionWithFailure(let promise, let stringError):
                Self.logger.error("invokeCompletionWithFailure \(stringError, privacy: .public)")
                promise.fail(with: stringError)
            }
        }
    }

    func succeededApplyingConfiguration(
        id: ConnectionID,
        revision: String
    ) async {
        await withLogging(
            operation: "succeededApplyingConfiguration \(id) to \(revision)",
            sensitiveError: false,
            logger: Self.logger
        ) {
            self.metrics.emit(Metrics.Registry.ApplyingSuccessCounter(action: .increment))
            switch self.stateMachine.succeededApplyingConfiguration(id: id, revision: revision) {
            case .reportError(let stringError):
                Self.logger.error(
                    "Failed applying configuration for id \(id, privacy: .public), revision \(revision, privacy: .public), error: \(String(reportable: stringError), privacy: .public), error (\(stringError, privacy: .public))."
                )
            case .reportSuccessOneConnection(let id, let revision):
                Self.logger.debug(
                    "Successfully applied configuration for id \(id, privacy: .public), revision \(revision, privacy: .public)."
                )
            case .invokeCompletionWithSuccess(let promise, let id, let revision):
                Self.logger.debug(
                    "Successfully applied configuration for id \(id, privacy: .public), revision \(revision, privacy: .public)."
                )
                Self.logger.notice(
                    "Successfully finished applying configuration revision \(revision, privacy: .public)."
                )
                promise.succeed()
            case .invokeCompletionWithFailure(
                let promise,
                error: let error,
                lastConnectionId: let id,
                lastConnectionRevision: let revision
            ):
                Self.logger.info(
                    "Successfully applied configuration for id \(id, privacy: .public), revision \(revision, privacy: .public)."
                )
                Self.logger.error(
                    "Failed applying configuration revision \(revision, privacy: .public), error: \(String(unredacted: error), privacy: .public)."
                )
                promise.fail(with: error)
            }
        }
    }

    func failedApplyingConfiguration(
        id: ConnectionID,
        revision: String,
        error: RegistryStateMachine.StringError
    ) async {
        await withLogging(
            operation: "failedApplyingConfiguration \(id) to \(revision)",
            sensitiveError: false,
            logger: Self.logger
        ) {
            self.metrics.emit(Metrics.Registry.ApplyingFailureCounter(action: .increment))
            switch self.stateMachine.failedApplyingConfiguration(id: id, revision: revision, error: error) {
            case .reportError(let stringError):
                Self.logger.error("State machine error: \(stringError, privacy: .public)")
            case .reportErrorOneConnection(let stringError, let id, let revision):
                Self.logger.error(
                    "Failed applying configuration for id \(id, privacy: .public), revision \(revision, privacy: .public), error: \(stringError, privacy: .public)."
                )
            case .invokeCompletionWithFailure(let promise, let error, let id, let revision):
                Self.logger.error(
                    "Failed applying configuration for id \(id, privacy: .public), revision \(revision, privacy: .public), error: \(String(unredacted: error), privacy: .public)."
                )
                Self.logger.error(
                    "Failed applying configuration revision \(revision, privacy: .public), error: \(String(unredacted: error), privacy: .public)."
                )
                promise.fail(with: error)
            }
        }
    }
}

extension Registry.NodeConfigurationState {
    /// Extracts the configuration just for the provided domain.
    ///
    /// If no entry is found, returns an empty configuration.
    /// - Parameter domain: The domain for which to extract the configuration values.
    /// - Returns: The configuration just for the domain.
    func configuration(forDomain domain: String) -> Registry.DomainConfigurationState {
        switch self {
        case .none:
            return .none
        case .fallback:
            return .fallback
        case .package(let package):
            return .package(package.configuration(forDomain: domain))
        }
    }
}

extension Registry.DomainConfigurationState: CustomStringConvertible {
    var description: String {
        switch self {
        case .none:
            return "none"
        case .fallback:
            return "fallback"
        case .package(let domainConfigurationPackage):
            return "package(\(domainConfigurationPackage.description))"
        }
    }
}

extension Registry.DomainConfigurationUpdate: CustomStringConvertible {
    var description: String {
        switch self {
        case .upToDate:
            return "upToDate"
        case .applyFallback:
            return "applyFallback"
        case .applyConfiguration(let domainConfigurationPackage):
            return "applyConfiguration(\(domainConfigurationPackage.description))"
        }
    }
}

extension Registry.Connection: CustomStringConvertible {
    var description: String {
        "\(id): \(domain) (\(currentConfiguration))"
    }
}

extension Registry.Connection: Equatable {
    static func == (lhs: Registry.Connection, rhs: Registry.Connection) -> Bool {
        lhs.id == rhs.id && lhs.domain == rhs.domain && lhs.currentConfiguration == rhs.currentConfiguration
    }
}

extension Registry.Connection: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(domain)
        hasher.combine(currentConfiguration)
    }
}
