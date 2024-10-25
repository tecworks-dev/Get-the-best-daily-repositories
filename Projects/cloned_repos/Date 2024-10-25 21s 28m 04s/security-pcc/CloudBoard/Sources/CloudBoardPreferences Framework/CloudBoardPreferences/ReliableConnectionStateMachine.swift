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

internal import CloudBoardConfigurationDAPI
import Foundation

/// The state machine of `ReliableConnection`.
struct ReliableConnectionStateMachine {
    /// A wrapper of a connection to the daemon.
    final class DaemonConnection {
        /// The underlying connection.
        let underlyingConnection: ConfigurationAPIXPCClientProtocol

        /// Creates a new connection.
        /// - Parameter connection: The underlying connection.
        init(underlyingConnection: ConfigurationAPIXPCClientProtocol) {
            self.underlyingConnection = underlyingConnection
        }
    }

    /// The current state.
    enum State: Hashable {
        /// The initial state.
        ///
        /// No attempt to connect has happened yet, or the connection has been instructed to disconnect.
        case disconnected

        /// Values associated with the `attemptingToConnect` state.
        struct AttemptingToConnectState: Hashable {
            /// The current configuration.
            var currentConfiguration: ConfigurationInfo
        }

        /// Trying to connect, or reconnect after an interruption.
        case attemptingToConnect(AttemptingToConnectState)

        /// Values associated with the `connected` state.
        struct ConnectedState: Hashable {
            /// The underlying connection to the daemon.
            var connection: DaemonConnection

            /// The current configuration.
            var currentConfiguration: ConfigurationInfo

            /// Whether a reply is expected from the framework side.
            var pendingReply: Bool
        }

        /// Connected.
        case connected(ConnectedState)

        /// Terminal state, either disconnected, canceled, or performed an invalid state transition.
        case terminal(TerminationCause)
    }

    /// The cause of a warning.
    enum WarningCause: Hashable {
        /// The connection was interrupted while disconnected.
        case interruptedDisconnectedConnection

        /// The state is already terminal.
        case alreadyTerminal(TerminationCause)

        /// Trying to apply configuration while in the `attemptingToConnect` state.
        case willApplyConfigurationInAttemptingToConnect
    }

    /// The cause of a terminal error.
    enum TerminationCause: Hashable {
        /// The connection was disconnected gracefully.
        case gracefulDisconnect

        /// Performed an invalid state transition.
        case invalidStateTransition(state: String, transition: String)

        /// Tried to connect a connection that was already terminal.
        case cannotReconnectTerminalConnection

        /// Tried to connect a connection that was already connected.
        case cannotReconnectConnectedConnection

        /// An error was thrown that should be propagated to the subscriber.
        case userPropagatedError(String)

        /// The connection was canceled.
        case canceled

        /// The client does not have the required entitlement.
        case notEntitled(String)
    }

    /// The current state of the state machine.
    private(set) var state: State = .disconnected

    /// The action returned by the `connect` method.
    enum ConnectAction: Hashable {
        /// Connect and register with the provided configuration info.
        case connectAndRegister(ConfigurationInfo)

        /// Disconnect and report a terminal error.
        case disconnectAndReportTerminalError(DaemonConnection, TerminationCause)

        /// Report a terminal error.
        case reportTerminalError(TerminationCause)
    }

    /// Call this method when the connection is asked to connect.
    mutating func connect() -> ConnectAction {
        switch self.state {
        case .disconnected:
            self.state = .attemptingToConnect(.init(currentConfiguration: .none))
            return .connectAndRegister(.none)
        case .attemptingToConnect:
            let terminationCause: TerminationCause = .invalidStateTransition(
                state: "attemptingToConnect",
                transition: "connect"
            )
            self.state = .terminal(terminationCause)
            return .reportTerminalError(terminationCause)
        case .connected(let state):
            let terminationCause: TerminationCause = .cannotReconnectConnectedConnection
            self.state = .terminal(terminationCause)
            return .disconnectAndReportTerminalError(state.connection, terminationCause)
        case .terminal:
            let terminationCause: TerminationCause = .cannotReconnectTerminalConnection
            self.state = .terminal(terminationCause)
            return .reportTerminalError(terminationCause)
        }
    }

    /// The action returned by the `didAttemptToConnectAndRegister` method.
    enum DidAttemptToConnectAndRegisterAction: Hashable {
        /// Report a terminal error.
        case reportTerminalError(TerminationCause)

        /// Tear down the connection and report a terminal error.
        case teardownConnectionAndReportTerminalError(DaemonConnection, TerminationCause)

        /// The first event emitted after an established connection.
        enum ConnectedFirstEvent: Hashable {
            /// No event, wait for further updates.
            case none

            /// Fall back to static configuration.
            case fallback

            /// Apply the provided configuration.
            case configuration(UnappliedConfiguration)
        }

        /// Report that a connection was established and emit the provided event.
        case reportConnected(ConnectedFirstEvent)

        /// Report that connection establishment failed and schedule a reconnect for later.
        case reportFailedToConnectAndScheduleReconnect(ReliableConnection.ConnectionError)
    }

    /// Call this method with the result of a connection and registration attempt.
    mutating func didAttemptToConnectAndRegister(
        _ result: Result<(DaemonConnection, ConfigurationUpdate), Error>
    ) -> DidAttemptToConnectAndRegisterAction {
        switch self.state {
        case .disconnected:
            let terminationCause: TerminationCause = .invalidStateTransition(
                state: "disconnected",
                transition: "didAttemptToConnectAndRegister"
            )
            self.state = .terminal(terminationCause)
            return .reportTerminalError(terminationCause)
        case .attemptingToConnect(let state):
            switch result {
            case .success(let success):
                let (connection, registrationResult) = success
                let firstEvent: DidAttemptToConnectAndRegisterAction.ConnectedFirstEvent
                let newConfigurationInfo: ConfigurationInfo
                switch registrationResult {
                case .upToDate:
                    firstEvent = .none
                    newConfigurationInfo = state.currentConfiguration
                case .applyFallback:
                    switch self.shouldApplyFallback() {
                    case .apply:
                        firstEvent = .fallback
                    case .dontApply:
                        firstEvent = .none
                    }
                    newConfigurationInfo = .fallback
                case .applyConfiguration(let unappliedConfiguration):
                    firstEvent = .configuration(unappliedConfiguration)
                    newConfigurationInfo = .revision(unappliedConfiguration.revisionIdentifier)
                }
                self.state = .connected(.init(
                    connection: connection,
                    currentConfiguration: newConfigurationInfo,
                    pendingReply: false
                ))
                return .reportConnected(firstEvent)
            case .failure(let failure):
                if
                    let apiError = failure as? ConfigurationAPIError,
                    case .userPropagatedError(let message) = apiError {
                    let terminationCause: TerminationCause = .userPropagatedError(message)
                    self.state = .terminal(terminationCause)
                    return .reportTerminalError(terminationCause)
                }
                if failure is CancellationError {
                    let terminationCause: TerminationCause = .canceled
                    self.state = .terminal(terminationCause)
                    return .reportTerminalError(terminationCause)
                }
                if case .notEntitled(let entitlement) = failure as? ConfigurationAPIError {
                    let terminationCause: TerminationCause = .notEntitled(entitlement)
                    self.state = .terminal(terminationCause)
                    return .reportTerminalError(terminationCause)
                }
                return .reportFailedToConnectAndScheduleReconnect(
                    .unknownError(String(describing: failure))
                )
            }
        case .connected(let connectedState):
            let terminationCause: TerminationCause = .invalidStateTransition(
                state: "connected",
                transition: "didAttemptToConnectAndRegister"
            )
            self.state = .terminal(terminationCause)
            return .teardownConnectionAndReportTerminalError(connectedState.connection, terminationCause)
        case .terminal(let terminationCause):
            return .reportTerminalError(terminationCause)
        }
    }

    /// The action returned by the `disconnect` method.
    enum DisconnectAction: Hashable {
        /// Disconnect the provided connection.
        case disconnect(DaemonConnection)

        /// Do nothing.
        case noop
    }

    /// Call this method when the connection is asked to disconnect.
    mutating func disconnect() -> DisconnectAction {
        switch self.state {
        case .disconnected:
            return .noop
        case .attemptingToConnect:
            self.state = .terminal(.gracefulDisconnect)
            return .noop
        case .connected(let state):
            self.state = .terminal(.gracefulDisconnect)
            return .disconnect(state.connection)
        case .terminal:
            return .noop
        }
    }

    /// The action returned by the `connectionInterrupted` method.
    enum ConnectionInterruptedAction: Hashable {
        /// Report a warning.
        case reportWarning(WarningCause)

        /// Schedule a reconnect for later.
        case scheduleReconnect(ConfigurationInfo)

        /// Tear down the provided connection and schedule a reconnect for later.
        case teardownConnectionAndScheduleReconnect(DaemonConnection, ConfigurationInfo)

        /// Do nothing.
        case noop
    }

    /// Call this method when the connection notifies you that it was interrupted.
    mutating func connectionInterrupted() -> ConnectionInterruptedAction {
        switch self.state {
        case .disconnected:
            return .reportWarning(.interruptedDisconnectedConnection)
        case .attemptingToConnect(let state):
            return .scheduleReconnect(state.currentConfiguration)
        case .connected(let state):
            self.state = .attemptingToConnect(.init(currentConfiguration: state.currentConfiguration))
            return .teardownConnectionAndScheduleReconnect(state.connection, state.currentConfiguration)
        case .terminal(let terminationCause):
            return .reportWarning(.alreadyTerminal(terminationCause))
        }
    }

    /// The action returned by the `willApplyConfiguration` method.
    enum WillApplyConfigurationAction: Hashable {
        /// Just apply.
        case apply(UnappliedConfiguration)

        /// Report a warning and still apply.
        case reportWarningAndApply(WarningCause, UnappliedConfiguration)

        /// Report a terminal error and don't apply.
        case reportTerminalErrorDontApply(TerminationCause)
    }

    /// Call this method before the connection is asked to apply a configuration.
    mutating func willApplyConfiguration(_ configuration: UnappliedConfiguration) -> WillApplyConfigurationAction {
        switch self.state {
        case .disconnected:
            let terminationCause: TerminationCause = .invalidStateTransition(
                state: "disconnected",
                transition: "willApplyConfiguration"
            )
            self.state = .terminal(terminationCause)
            return .reportTerminalErrorDontApply(terminationCause)
        case .attemptingToConnect:
            return .reportWarningAndApply(.willApplyConfigurationInAttemptingToConnect, configuration)
        case .connected(var connectedState):
            connectedState.currentConfiguration = .revision(configuration.revisionIdentifier)
            connectedState.pendingReply = true
            self.state = .connected(connectedState)
            return .apply(configuration)
        case .terminal(let terminationCause):
            return .reportTerminalErrorDontApply(terminationCause)
        }
    }

    /// The action returned by the `willReply` method.
    enum WillReplyAction: Hashable {
        /// Reply to the provided connection.
        case reply(DaemonConnection)

        /// Discard the reply to the connection.
        case dontReply
    }

    /// Call this method before forwarding a reply from the subscriber to the daemon.
    mutating func willReply() -> WillReplyAction {
        switch self.state {
        case .disconnected, .attemptingToConnect, .terminal:
            return .dontReply
        case .connected(var connectedState):
            guard connectedState.pendingReply else {
                return .dontReply
            }
            connectedState.pendingReply = false
            self.state = .connected(connectedState)
            return .reply(connectedState.connection)
        }
    }

    /// The action returned by the `shouldApplyFallback` method.
    enum ShouldApplyFallbackAction: Hashable {
        /// Apply fallback, as instructed.
        case apply

        /// Discard applying fallback.
        case dontApply
    }

    /// Call this method to check whether to go ahead with applying fallback.
    mutating func shouldApplyFallback() -> ShouldApplyFallbackAction {
        switch self.state {
        case .disconnected, .terminal:
            return .dontApply
        case .attemptingToConnect(let state):
            guard case .none = state.currentConfiguration else {
                return .dontApply
            }
            return .apply
        case .connected(let state):
            guard case .none = state.currentConfiguration else {
                return .dontApply
            }
            return .apply
        }
    }

    /// The action returned by the `applyFallback` method.
    enum ApplyFallbackAction: Hashable {
        /// Apply fallback, as instructed.
        case apply

        /// Discard applying fallback.
        case dontApply
    }

    /// Call this method before applying fallback.
    mutating func applyFallback() -> ApplyFallbackAction {
        switch (self.state, self.shouldApplyFallback()) {
        case (.disconnected, _), (.terminal, _):
            return .dontApply
        case (.attemptingToConnect(var state), .apply):
            state.currentConfiguration = .fallback
            self.state = .attemptingToConnect(state)
            return .apply
        case (.connected(var state), .apply):
            state.currentConfiguration = .fallback
            self.state = .connected(state)
            return .apply
        case (_, .dontApply):
            return .dontApply
        }
    }
}

extension ReliableConnectionStateMachine.State: CustomStringConvertible {
    var description: String {
        switch self {
        case .disconnected:
            return "disconnected"
        case .attemptingToConnect(let state):
            return "attemptingToConnect (\(state.currentConfiguration))"
        case .connected(let state):
            return "connected (\(state.currentConfiguration), pendingReply: \(state.pendingReply))"
        case .terminal(let terminationCause):
            return "terminal (\(terminationCause))"
        }
    }
}

extension ReliableConnectionStateMachine.DaemonConnection: Equatable {
    static func == (
        lhs: ReliableConnectionStateMachine.DaemonConnection,
        rhs: ReliableConnectionStateMachine.DaemonConnection
    ) -> Bool {
        ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
}

extension ReliableConnectionStateMachine.DaemonConnection: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

extension ReliableConnectionStateMachine.DaemonConnection {
    func disconnect() async {
        await underlyingConnection.disconnect()
    }
}

extension ReliableConnectionStateMachine.DaemonConnection: ConfigurationAPIClientToServerProtocol {
    func register(_ registration: Registration) async throws -> ConfigurationUpdate {
        try await underlyingConnection.register(registration)
    }

    func successfullyAppliedConfiguration(_ success: ConfigurationApplyingSuccess) async throws {
        try await underlyingConnection.successfullyAppliedConfiguration(success)
    }

    func failedToApplyConfiguration(_ failure: ConfigurationApplyingFailure) async throws {
        try await underlyingConnection.failedToApplyConfiguration(failure)
    }

    func currentConfigurationVersionInfo() async throws -> ConfigurationVersionInfo {
        try await underlyingConnection.currentConfigurationVersionInfo()
    }
}
