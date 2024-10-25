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
import Foundation

/// A state machine that tracks the state of the connection registry.
///
/// It controls the state transitions and both keeps track of the current
/// state, such as the current configuration package, and also verifies
/// that state transitions are accompanied with valid side effects.
///
/// For a diagram, check out `Docs/registry-state-machine.md`.
struct RegistryStateMachine {
    /// The current state of the state machine.
    enum State {
        /// The idle state.
        case idle(
            configuration: Registry.NodeConfigurationState,
            connections: Registry.Connections
        )

        /// The provided config is being applied to the unapplied connections.
        case applyingConfig(
            configuration: NodeConfigurationPackage,
            completion: Promise<Void, StringError>,
            unappliedConnections: Registry.Connections,
            appliedConnections: Registry.Connections,
            failedToApplyConnections: Registry.Connections,
            disconnectedDuringApplyingConnections: Registry.Connections
        )

        /// A state used to avoid CoW copies.
        case mutating
    }

    /// The current state of the state machine.
    private(set) var state: State = .idle(configuration: .none, connections: [:])

    /// A stringified error.
    struct StringError: Error, LocalizedError, CustomStringConvertible, Hashable {
        /// The message of the error.
        let message: String

        var description: String {
            self.message
        }

        var errorDescription: String? { self.description }
    }
}

extension RegistryStateMachine {
    /// The action returned by the `register` method.
    enum RegisterAction: Hashable {
        /// Reply with the provided update.
        case reply(update: Registry.DomainConfigurationUpdate)
    }

    /// Registers the provided connection.
    /// - Parameters:
    ///   - connection: The underlying connection.
    ///   - domain: The configuration domain to register for.
    ///   - currentConfiguration: The configuration currently present on the peer.
    /// - Returns: The action to perform.
    mutating func register(
        connection: ConfigurationAPIServerToClientConnection,
        domain: String,
        currentConfiguration: ConfigurationInfo
    ) -> RegisterAction {
        let id = connection.id
        switch self.state {
        case .idle(configuration: let configuration, connections: var connections):
            self.state = .mutating
            guard connections[id] == nil else {
                fatalError("Connection \(id) is already registered.")
            }
            let candidateConfiguration = configuration.configuration(forDomain: domain)
            let (update, newConfiguration) = Self.computeUpdate(
                from: currentConfiguration,
                to: candidateConfiguration
            )
            let activeConnection = Registry.Connection(
                connection: connection,
                domain: domain,
                currentConfiguration: newConfiguration
            )
            connections[id] = activeConnection
            self.state = .idle(configuration: configuration, connections: connections)
            return .reply(update: update)
        case .applyingConfig(
            configuration: let configuration,
            completion: let completion,
            unappliedConnections: let unappliedConnections,
            appliedConnections: var appliedConnections,
            failedToApplyConnections: let failedToApplyConnections,
            disconnectedDuringApplyingConnections: let disconnectedDuringApplyingConnections
        ):
            self.state = .mutating
            guard unappliedConnections[id] == nil, appliedConnections[id] == nil else {
                fatalError("Connection \(id) is already registered.")
            }
            let candidateConfiguration: Registry.DomainConfigurationState = .package(
                configuration.configuration(forDomain: domain)
            )
            let (update, newConfiguration) = Self.computeUpdate(
                from: currentConfiguration,
                to: candidateConfiguration
            )
            let activeConnection = Registry.Connection(
                connection: connection,
                domain: domain,
                currentConfiguration: newConfiguration
            )
            // Connections registered during an ongoing application are considered
            // already applied.
            appliedConnections[id] = activeConnection
            self.state = .applyingConfig(
                configuration: configuration,
                completion: completion,
                unappliedConnections: unappliedConnections,
                appliedConnections: appliedConnections,
                failedToApplyConnections: failedToApplyConnections,
                disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
            )
            return .reply(update: update)
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }

    /// The action returned by the `disconnected` method.
    enum DisconnectedAction {
        /// Report the connection as disconnected.
        case reportDisconnectOneConnection(id: ConnectionID)

        /// Report the connection as not found.
        case reportDisconnectedUnknownConnection(id: ConnectionID)

        /// Invoke the provided completion with the provided error.
        case invokeCompletionWithFailure(Promise<Void, StringError>, StringError)
    }

    /// Handles the disconnected connection.
    /// - Parameter id: The identifier of the connection.
    /// - Returns: The action to perform.
    mutating func disconnected(
        connectionID id: ConnectionID
    ) -> DisconnectedAction {
        switch self.state {
        case .idle(configuration: let configuration, connections: var connections):
            self.state = .mutating
            guard connections.removeValue(forKey: id) != nil else {
                self.state = .idle(configuration: configuration, connections: connections)
                return .reportDisconnectedUnknownConnection(id: id)
            }
            self.state = .idle(configuration: configuration, connections: connections)
            return .reportDisconnectOneConnection(id: id)
        case .applyingConfig(
            configuration: let configuration,
            completion: let completion,
            unappliedConnections: var unappliedConnections,
            appliedConnections: var appliedConnections,
            failedToApplyConnections: var failedToApplyConnections,
            disconnectedDuringApplyingConnections: var disconnectedDuringApplyingConnections
        ):
            self.state = .mutating
            if let foundConnection = unappliedConnections.removeValue(forKey: id) {
                // A connection that we're waiting for a confirmation of disconnected.
                // Consider this a failure during applying the configuration.
                disconnectedDuringApplyingConnections[id] = foundConnection
            } else if appliedConnections.removeValue(forKey: id) != nil {
                // A connection that successfully applied disconnected, that's okay, just remove it.
            } else if let foundConnection = failedToApplyConnections.removeValue(forKey: id) {
                // A connection that first failed applying then disconnected, just move it to disconnected.
                disconnectedDuringApplyingConnections[id] = foundConnection
            } else if disconnectedDuringApplyingConnections.removeValue(forKey: id) != nil {
                fatalError("Received two disconnect calls for connection with id \(id).")
            } else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportDisconnectedUnknownConnection(id: id)
            }
            if unappliedConnections.isEmpty {
                let connections = appliedConnections.merging(failedToApplyConnections) { connection, _ in
                    fatalError(
                        "The connection \(connection.id) was present in both applied and failedToApply connection lists."
                    )
                }
                self.state = .idle(
                    configuration: .package(configuration),
                    connections: connections
                )
                let error =
                    StringError(
                        message: "Some connections disconnected before successfully applying: \(Self.connectionsDescription(disconnectedDuringApplyingConnections))"
                    )
                return .invokeCompletionWithFailure(completion, error)
            } else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportDisconnectOneConnection(id: id)
            }
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }

    /// The action returned by the `applyConfiguration` method.
    enum ApplyConfigurationAction {
        /// Invoke the provided completion with success.
        case invokeCompletionWithSuccess(
            Promise<Void, StringError>,
            upToDateConnectionCount: Int
        )

        /// Send the provided configuration packages to their respective connections, and report how many
        /// connections are already up-to-date and don't need updating.
        case send(
            revisionIdentifier: String,
            connectionsToUpdate: [(DomainConfigurationPackage, ConfigurationAPIServerToClientConnection)],
            upToDateConnectionCount: Int
        )
    }

    /// Applies the provided configuration.
    /// - Parameters:
    ///   - configuration: The configuration to apply.
    ///   - completion: The completion to invoke when the configuration is applied.
    /// - Returns: The action to perform.
    mutating func applyConfiguration(
        configuration: NodeConfigurationPackage,
        completion: Promise<Void, StringError>
    ) -> ApplyConfigurationAction {
        switch self.state {
        case .idle(_, let connections):
            var connectionsToUpdate: [ConnectionID: Registry.Connection] = [:]
            var upToDateConnections: [ConnectionID: Registry.Connection] = [:]
            for (id, connection) in connections {
                if
                    case .revision(let revisionIdentifier) = connection.currentConfiguration,
                    revisionIdentifier == configuration.revisionIdentifier {
                    upToDateConnections[id] = connection
                } else {
                    connectionsToUpdate[id] = connection
                }
            }
            let alreadyAppliedConnections = upToDateConnections.count
            if connectionsToUpdate.isEmpty {
                self.state = .idle(
                    configuration: .package(configuration),
                    connections: upToDateConnections
                )
                return .invokeCompletionWithSuccess(completion, upToDateConnectionCount: alreadyAppliedConnections)
            } else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: connectionsToUpdate,
                    appliedConnections: upToDateConnections,
                    failedToApplyConnections: [:],
                    disconnectedDuringApplyingConnections: [:]
                )
                let unappliedConnections = connectionsToUpdate
                    .values
                    .map { connection in
                        (
                            configuration.configuration(forDomain: connection.domain),
                            connection.connection
                        )
                    }
                return .send(
                    revisionIdentifier: configuration.revisionIdentifier,
                    connectionsToUpdate: unappliedConnections,
                    upToDateConnectionCount: alreadyAppliedConnections
                )
            }
        case .applyingConfig:
            fatalError("Invalid transition: tried to apply a new config before a previous one was completed.")
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }

    /// The action returned by the `applyFallback` method.
    enum ApplyFallbackAction {
        /// Send the request to apply fallback to the provided connections.
        case send([ConfigurationAPIServerToClientConnection])
    }

    /// Applies the fallback configuration.
    /// - Returns: The action to perform.
    mutating func applyFallback() -> ApplyFallbackAction {
        switch self.state {
        case .idle(let existingConfiguration, let connections):
            guard case .none = existingConfiguration else {
                fatalError("Invalid transition: must be from 'none' to 'fallback'.")
            }
            self.state = .mutating
            let updatedConnections: [ConnectionID: (Registry.Connection, Bool)] = connections.mapValues { value in
                let (update, new) = Self.computeUpdate(
                    from: value.currentConfiguration,
                    to: .fallback
                )
                var value = value
                value.currentConfiguration = new
                return (value, update != .upToDate)
            }
            self.state = .idle(
                configuration: .fallback,
                connections: updatedConnections.mapValues(\.0)
            )
            let connectionsToFallBack = updatedConnections.filter { $0.value.1 }.map(\.value.0.connection)
            return .send(connectionsToFallBack)
        case .applyingConfig:
            fatalError("Invalid transition: tried to apply a new config before a previous one was completed.")
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }

    /// The action returned by the `succeededApplyingConfiguration` method.
    enum SucceededApplyingConfigurationAction {
        /// Report the provided error.
        case reportError(StringError)

        /// Report that the provided connection successfully applied the provided configuration revision.
        case reportSuccessOneConnection(
            id: ConnectionID,
            revision: String
        )

        /// Invoke the provided completion with success, while also reporting that the last connection succeeded.
        case invokeCompletionWithSuccess(
            Promise<Void, StringError>,
            lastConnectionId: ConnectionID,
            lastConnectionRevision: String
        )

        /// Invoke the provided completion with the provided error, while also reporting the last connection
        /// that completed.
        case invokeCompletionWithFailure(
            Promise<Void, StringError>,
            error: StringError,
            lastConnectionId: ConnectionID,
            lastConnectionRevision: String
        )
    }

    /// Handle that a connection successfully applied the provided configuration revision.
    ///
    /// - Parameters:
    ///   - id: The identifier of the connection.
    ///   - revision: The revision that was applied.
    /// - Returns: The action to perform.
    mutating func succeededApplyingConfiguration(
        id: ConnectionID,
        revision: String
    ) -> SucceededApplyingConfigurationAction {
        switch self.state {
        case .idle:
            return .reportError(StringError(message: "Received a succeededApplyingConfig in idle state."))
        case .applyingConfig(
            configuration: let configuration,
            completion: let completion,
            unappliedConnections: var unappliedConnections,
            appliedConnections: var appliedConnections,
            failedToApplyConnections: let failedToApplyConnections,
            disconnectedDuringApplyingConnections: let disconnectedDuringApplyingConnections
        ):
            self.state = .mutating
            guard
                var connection = unappliedConnections.removeValue(forKey: id),
                configuration.revisionIdentifier == revision
            else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportError(
                    StringError(
                        message: "Received a succeededApplyingConfig for an unrecognized apply request (\(id), \(revision))."
                    )
                )
            }
            connection.currentConfiguration = .revision(revision)
            appliedConnections[id] = connection
            if unappliedConnections.isEmpty {
                defer {
                    let connections = appliedConnections.merging(failedToApplyConnections) { connection, _ in
                        fatalError(
                            "The connection \(connection.id) was present in both applied and failedToApply connection lists."
                        )
                    }
                    self.state = .idle(
                        configuration: .package(configuration),
                        connections: connections
                    )
                }
                let failedConnections = disconnectedDuringApplyingConnections
                    .merging(failedToApplyConnections) { connection, _ in
                        fatalError(
                            "The connection \(connection.id) was present in both disconnectedDuringApplying and failedToApply connection lists."
                        )
                    }
                if failedConnections.isEmpty {
                    return .invokeCompletionWithSuccess(
                        completion,
                        lastConnectionId: id,
                        lastConnectionRevision: revision
                    )
                } else {
                    let error =
                        StringError(
                            message: "Some connections failed to apply: \(Self.connectionsDescription(failedConnections))"
                        )
                    return .invokeCompletionWithFailure(
                        completion,
                        error: error,
                        lastConnectionId: id,
                        lastConnectionRevision: revision
                    )
                }
            } else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportSuccessOneConnection(id: id, revision: revision)
            }
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }

    /// The action returned by the `failedApplyingConfiguration` method.
    enum FailedApplyingConfigurationAction {
        /// Report the error.
        case reportError(StringError)

        /// Report that the provided connection failed to apply the provided configuration revision.
        case reportErrorOneConnection(
            StringError,
            id: ConnectionID,
            revision: String
        )

        /// Invoke the provided completion with the provided error, while also reporting the last connection
        /// that completed.
        case invokeCompletionWithFailure(
            Promise<Void, StringError>,
            error: StringError,
            lastConnectionId: ConnectionID,
            lastConnectionRevision: String
        )
    }

    /// Handle that a connection failed to apply a configuration revision.
    ///
    /// - Parameters:
    ///   - id: The identifier of the connection.
    ///   - revision: The revision of the configuration that failed to apply.
    ///   - error: The error that caused the failure.
    /// - Returns: The action to perform.
    mutating func failedApplyingConfiguration(
        id: ConnectionID,
        revision: String,
        error: StringError
    ) -> FailedApplyingConfigurationAction {
        switch self.state {
        case .idle:
            return .reportError(StringError(message: "Received a failedApplyingConfig in idle state."))
        case .applyingConfig(
            configuration: let configuration,
            completion: let completion,
            unappliedConnections: var unappliedConnections,
            appliedConnections: let appliedConnections,
            failedToApplyConnections: var failedToApplyConnections,
            disconnectedDuringApplyingConnections: let disconnectedDuringApplyingConnections
        ):
            self.state = .mutating
            guard
                let connection = unappliedConnections.removeValue(forKey: id),
                configuration.revisionIdentifier == revision
            else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportError(
                    StringError(
                        message: "Received a failedApplyingConfig for an unrecognized apply request (\(id), \(revision))."
                    )
                )
            }
            failedToApplyConnections[id] = connection
            if unappliedConnections.isEmpty {
                let connections = appliedConnections.merging(failedToApplyConnections) { connection, _ in
                    fatalError(
                        "The connection \(connection.id) was present in both applied and failedToApply connection lists."
                    )
                }
                self.state = .idle(
                    configuration: .package(configuration),
                    connections: connections
                )
                let error =
                    StringError(
                        message: "Some connections failed to apply: \(Self.connectionsDescription(failedToApplyConnections))"
                    )
                return .invokeCompletionWithFailure(
                    completion,
                    error: error,
                    lastConnectionId: id,
                    lastConnectionRevision: revision
                )
            } else {
                self.state = .applyingConfig(
                    configuration: configuration,
                    completion: completion,
                    unappliedConnections: unappliedConnections,
                    appliedConnections: appliedConnections,
                    failedToApplyConnections: failedToApplyConnections,
                    disconnectedDuringApplyingConnections: disconnectedDuringApplyingConnections
                )
                return .reportErrorOneConnection(
                    error,
                    id: id,
                    revision: revision
                )
            }
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }
}

extension RegistryStateMachine {
    /// Computes the update from the provided configuration info to the candidate configuration state.
    /// - Parameters:
    ///   - from: The current configuration info.
    ///   - to: The candidate configuration state.
    /// - Returns: The update to apply, and the new configuration info after that update.
    static func computeUpdate(
        from: ConfigurationInfo,
        to: Registry.DomainConfigurationState
    ) -> (update: Registry.DomainConfigurationUpdate, new: ConfigurationInfo) {
        switch (from, to) {
        case (.revision(let existingRevision), .package(let newConfiguration)):
            let newRevision = newConfiguration.revisionIdentifier
            if existingRevision == newRevision {
                // Already up-to-date.
                return (.upToDate, .revision(newRevision))
            } else {
                // Needs updating.
                return (.applyConfiguration(newConfiguration), .revision(newRevision))
            }
        case (.none, .fallback):
            // Go from none -> fallback.
            return (.applyFallback, .fallback)
        case (_, .package(let newConfiguration)):
            // Move to a config.
            let newRevision = newConfiguration.revisionIdentifier
            return (.applyConfiguration(newConfiguration), .revision(newRevision))
        case (_, .none), (_, .fallback):
            // No update, either because we're up-to-date or because we never regress
            // from a useful value (fallback, package) to an unusable one (none), only
            // the other way around.
            return (.upToDate, from)
        }
    }

    private static func connectionsDescription(_ connections: Registry.Connections) -> String {
        connections.values.sorted { $0.id < $1.id }.map(\.description).joined(separator: ", ")
    }
}

extension RegistryStateMachine: CustomStringConvertible {
    var description: String {
        self.state.description
    }
}

extension RegistryStateMachine {
    /// The number of connections registered in the state machine.
    var connectionCount: Int {
        switch self.state {
        case .idle(_, connections: let connections):
            return connections.count
        case .applyingConfig(
            _,
            _,
            unappliedConnections: let unappliedConnections,
            appliedConnections: let appliedConnections,
            failedToApplyConnections: let failedToApplyConnections,
            _
        ):
            return unappliedConnections.count + appliedConnections.count + failedToApplyConnections.count
        case .mutating:
            fatalError("Invalid state: mutating")
        }
    }
}
