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
import os
internal import CloudBoardLogging

/// A delegate of a reliable connection.
protocol ReliableConnectionDelegate: AnyObject, Sendable {
    /// Apply fallback.
    func applyFallback() async

    /// Apply a new configuration.
    /// - Parameters:
    ///   - configuration: The configuration to apply.
    ///   - shouldReply: Whether the delegate must ensure a reply is sent back (to `successfullyAppliedConfiguration`
    ///     or `failedToApplyConfiguration`).
    func applyConfiguration(_ configuration: UnappliedConfiguration, shouldReply: Bool) async

    /// A terminal error was encountered and the reliable connection is shut down and will not emit any more events.
    /// - Parameter error: The encountered error.
    func reportTerminalError(_ error: Error) async
}

/// A connection to the daemon that automatically reconnects after interruptions.
actor ReliableConnection {
    private static let logger: os.Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "ReliableConnection"
    )

    /// The preferences domain being watched.
    private let domain: String

    /// A closure that creates a new connection to the daemon.
    private let makeUnreliableConnection: () async -> ConfigurationAPIXPCClientProtocol

    /// The amount of time to delay reconnecting for.
    private let reconnectDelay: Duration

    /// The delegate.
    private weak var delegate: ReliableConnectionDelegate?

    /// The underlying state machine.
    private(set) var stateMachine: ReliableConnectionStateMachine {
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
        }
    }

    /// An error emitted by a reliable connection.
    enum ConnectionError: Error, LocalizedError, CustomStringConvertible, Hashable {
        /// The delegate is nil.
        case delegateNotSet

        /// An error that should be surfaced to the subscriber.
        case userPropagatedError(String)

        /// The state machine encountered an error.
        case stateMachineError(ReliableConnectionStateMachine.TerminationCause)

        /// Unknown error.
        case unknownError(String)

        var description: String {
            switch self {
            case .delegateNotSet:
                return "ReliableConnection delegate not set."
            case .userPropagatedError(let error):
                return error
            case .stateMachineError(let error):
                return "ReliableConnection state error: \(error)."
            case .unknownError(let string):
                return "Unknown error: \(string)."
            }
        }

        var errorDescription: String? {
            self.description
        }
    }

    /// Creates a new reliable connection.
    /// - Parameters:
    ///   - domain: The domain to watch.
    ///   - makeUnreliableConnection: The factory closure to create new connections to the daemon.
    ///   - reconnectDelay: The delay to wait between reconnect attempts.
    init(
        domain: String,
        makeUnreliableConnection: @escaping () async -> ConfigurationAPIXPCClientProtocol,
        reconnectDelay: Duration
    ) {
        self.domain = domain
        self.makeUnreliableConnection = makeUnreliableConnection
        self.reconnectDelay = reconnectDelay
        self.stateMachine = .init()
    }

    /// Sets the delegate.
    /// - Parameter delegate: The delegate.
    func set(delegate: ReliableConnectionDelegate) async {
        self.delegate = delegate
    }

    /// Enables the reliable connection to establish, and retain, a valid underlying connection.
    func connect() async {
        Self.logger.debug("ReliableConnection connect called.")
        switch self.stateMachine.connect() {
        case .connectAndRegister(let configurationInfo):
            await self.connectAndRegister(currentConfiguration: configurationInfo)
        case .disconnectAndReportTerminalError(let connection, let error):
            Self.logger.error("Connect failed with error: \(String(describing: error), privacy: .public)")
            await connection.disconnect()
            await self.delegate?.reportTerminalError(ConnectionError.stateMachineError(error))
        case .reportTerminalError(let error):
            Self.logger.error("Connect failed with error: \(String(describing: error), privacy: .public)")
            await self.delegate?.reportTerminalError(ConnectionError.stateMachineError(error))
        }
    }

    /// Disables the reliable connection.
    func disconnect() async {
        Self.logger.debug("ReliableConnection disconnect called.")
        switch self.stateMachine.disconnect() {
        case .disconnect(let connection):
            await self.teardownConnection(connection)
        case .noop:
            break
        }
        self.cleanup()
    }

    /// Cleans up the underlying connection.
    private func cleanup() {
        self.delegate = nil
    }

    /// Tears down the underlying connection.
    private func teardownConnection(_ connection: ReliableConnectionStateMachine.DaemonConnection) async {
        await connection.disconnect()
    }

    /// Reports a terminal error and cleans up.
    private func reportTerminalErrorAndCleanup(_ error: Error) async {
        Self.logger.error(
            "ReliableConnection terminal error: \(String(reportable: error), privacy: .public) (\(error))"
        )
        await self.delegate?.reportTerminalError(error)
        self.cleanup()
    }

    /// Reports a warning.
    private func reportWarning(_ warning: ReliableConnectionStateMachine.WarningCause) {
        Self.logger.warning("ReliableConnection warning: \(String(describing: warning), privacy: .public)")
    }

    /// Connects and registers an underlying connection.
    private func connectAndRegister(currentConfiguration: ConfigurationInfo) async {
        let result = await attemptToConnectAndRegister(currentConfiguration: currentConfiguration)
        switch self.stateMachine.didAttemptToConnectAndRegister(result) {
        case .reportTerminalError(let stateError):
            await self.reportTerminalErrorAndCleanup(ConnectionError.stateMachineError(stateError))
        case .teardownConnectionAndReportTerminalError(let connection, let stateError):
            await connection.disconnect()
            await self.reportTerminalErrorAndCleanup(ConnectionError.stateMachineError(stateError))
        case .reportConnected(let firstEvent):
            Self.logger
                .info(
                    "ReliableConnection is successfully connected, first event: \(String(describing: firstEvent), privacy: .public)"
                )
            switch firstEvent {
            case .none:
                break
            case .fallback:
                await self.delegate?.applyFallback()
            case .configuration(let configuration):
                await self.delegate?.applyConfiguration(configuration, shouldReply: false)
            }
        case .reportFailedToConnectAndScheduleReconnect(let error):
            Self.logger.error(
                "ReliableConnection failed to connect with error: \(String(reportable: error), privacy: .public) (\(error)), will retry."
            )
            await self.reconnectAfterDelay(currentConfiguration: currentConfiguration)
        }
    }

    /// Attempts to connect and register an underlying connection.
    private func attemptToConnectAndRegister(
        currentConfiguration: ConfigurationInfo
    ) async -> Result<(ReliableConnectionStateMachine.DaemonConnection, ConfigurationUpdate), Error> {
        await withLogging(operation: "attemptToConnectAndRegister", logger: Self.logger) {
            guard !Task.isCancelled else {
                return .failure(CancellationError())
            }
            let connection = await self.makeUnreliableConnection()
            await connection.set(delegate: self)
            await connection.connect()
            do {
                let registrationResult = try await connection.register(.init(
                    domainName: self.domain,
                    currentConfiguration: currentConfiguration
                ))
                Self.logger
                    .debug(
                        "ReliableConnection got registration result: \(String(describing: registrationResult), privacy: .public)."
                    )
                return .success((.init(underlyingConnection: connection), registrationResult))
            } catch {
                return .failure(error)
            }
        }
    }

    /// Reconnects a new underlying connection after a delay.
    private func reconnectAfterDelay(currentConfiguration: ConfigurationInfo) async {
        do {
            try await Task.sleep(for: self.reconnectDelay)
            await self.connectAndRegister(currentConfiguration: currentConfiguration)
        } catch {}
    }
}

extension ReliableConnection {
    /// Notifies the daemon that the subscriber successfully applied the provided configuration.
    func successfullyAppliedConfiguration(_ success: ConfigurationApplyingSuccess) async {
        Self.logger
            .debug(
                "ReliableConnection got successfullyAppliedConfiguration \(success.revisionIdentifier, privacy: .public)."
            )
        switch self.stateMachine.willReply() {
        case .reply(let connection):
            do {
                try await connection.successfullyAppliedConfiguration(success)
            } catch {
                Self.logger.error(
                    "ReliableConnection failed successfullyAppliedConfiguration \(success.revisionIdentifier, privacy: .public) after error: \(String(reportable: error), privacy: .public) (\(error))."
                )
                // If this happened, the daemon considers us failed anyway, so don't retry.
                // The connection will reconnect automatically.
            }
        case .dontReply:
            Self.logger
                .warning(
                    "ReliableConnection skipping a reply out of successfullyAppliedConfiguration \(success.revisionIdentifier, privacy: .public) because the connection got interrupted."
                )
        }
    }

    /// Notifies the daemon that the subscriber failed to apply the provided configuration.
    func failedToApplyConfiguration(_ failure: ConfigurationApplyingFailure) async {
        Self.logger
            .debug("ReliableConnection got failedToApplyConfiguration \(failure.revisionIdentifier, privacy: .public).")
        switch self.stateMachine.willReply() {
        case .reply(let connection):
            do {
                try await connection.failedToApplyConfiguration(failure)
            } catch {
                Self.logger.error(
                    "ReliableConnection failed failedToApplyConfiguration \(failure.revisionIdentifier, privacy: .public) after error: \(String(reportable: error), privacy: .public) (\(error))."
                )
                // If this happened, the daemon considers us failed anyway, so don't retry.
                // The connection will reconnect automatically.
            }
        case .dontReply:
            Self.logger
                .warning(
                    "ReliableConnection skipping a reply out of successfullyAppliedConfiguration \(failure.revisionIdentifier, privacy: .public) because the connection got interrupted."
                )
        }
    }
}

extension ReliableConnection: ConfigurationAPIClientDelegateProtocol {
    func surpriseDisconnect() async {
        Self.logger
            .error("ReliableConnection got a surpriseDisconnect (the daemon probably crashed). Will try to reconnect.")
        switch self.stateMachine.connectionInterrupted() {
        case .reportWarning(let warning):
            self.reportWarning(warning)
        case .scheduleReconnect(let configurationInfo):
            await self.reconnectAfterDelay(currentConfiguration: configurationInfo)
        case .teardownConnectionAndScheduleReconnect(let connection, let configurationInfo):
            await self.teardownConnection(connection)
            await self.reconnectAfterDelay(currentConfiguration: configurationInfo)
        case .noop:
            break
        }
    }

    func applyFallback(_: FallbackToStaticConfiguration) async throws {
        guard let delegate else {
            throw ConnectionError.delegateNotSet
        }
        Self.logger.debug("ReliableConnection got applyFeedback.")
        switch self.stateMachine.applyFallback() {
        case .apply:
            await delegate.applyFallback()
        case .dontApply:
            Self.logger.warning(
                "ReliableConnection is dropping a call to applyFeedback to avoid thrashing."
            )
        }
    }

    func applyConfiguration(_ configuration: UnappliedConfiguration) async throws {
        guard let delegate else {
            throw ConnectionError.delegateNotSet
        }
        Self.logger
            .debug("ReliableConnection got applyConfiguration \(configuration.revisionIdentifier, privacy: .public).")
        switch self.stateMachine.willApplyConfiguration(configuration) {
        case .apply(let configuration):
            await delegate.applyConfiguration(configuration, shouldReply: true)
        case .reportWarningAndApply(let warning, let configuration):
            self.reportWarning(warning)
            await delegate.applyConfiguration(configuration, shouldReply: true)
        case .reportTerminalErrorDontApply(let error):
            await self.reportTerminalErrorAndCleanup(ConnectionError.stateMachineError(error))
        }
    }
}
