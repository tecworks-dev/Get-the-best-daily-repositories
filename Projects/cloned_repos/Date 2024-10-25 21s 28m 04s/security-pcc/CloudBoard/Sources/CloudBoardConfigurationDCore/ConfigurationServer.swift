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

import CloudBoardConfigurationDAPI
import CloudBoardLogging
import Foundation
import os

/// A delegate of the server, which is informed of events from the server.
///
/// The delegate is informed of the following events:
/// - A new connection is established.
/// - A connection is disconnected.
/// - A connection has successfully applied a configuration.
/// - A connection has failed to apply a configuration.
protocol ConfigurationServerDelegate: AnyObject, Sendable {
    /// Registers a new connection.
    ///
    /// - Parameters:
    ///   - connection: The underlying connection.
    ///   - domain: The requested configuration domain.
    ///   - currentConfiguration: The current configuration of the peer.
    /// - Returns: A configuration update that the peer should apply.
    func register(
        connection: ConfigurationAPIServerToClientConnection,
        domain: String,
        currentConfiguration: ConfigurationInfo
    ) async throws -> ConfigurationUpdate

    /// Informs the delegate that a connection was disconnected.
    /// - Parameter connectionID: The identifier of the connection.
    func disconnected(connectionID: ConnectionID) async

    /// Informs the delegate that a peer successfully applied a configuration.
    /// - Parameters:
    ///   - id: The identifier of the connection.
    ///   - revision: The configuration revision that was applied.
    func succeededApplyingConfiguration(
        id: ConnectionID,
        revision: String
    ) async

    /// Informs the delegate that a peer failed to apply a configuration.
    /// - Parameters:
    ///   - id: The identifier of the connection.
    ///   - revision: The configuration revision that failed to apply.
    ///   - error: The underlying error that caused the failure.
    func failedApplyingConfiguration(
        id: ConnectionID,
        revision: String,
        error: RegistryStateMachine.StringError
    ) async
}

/// A delegate of the server, which provides information about the current state.
protocol ConfigurationServerInfoDelegate: AnyObject, Sendable {
    /// Returns the current configuration version info.
    func currentConfigurationVersionInfo(
        connectionID: ConnectionID
    ) async -> ConfigurationVersionInfo
}

/// An actor that represents the server that handles connections from peers.
actor ConfigurationServer {
    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "ConfigurationServerXPCHandler"
    )

    /// The underlying XPC server.
    private let xpcServer: ConfigurationAPIXPCServer

    /// The delegate that gets informed of events.
    private weak var delegate: ConfigurationServerDelegate?

    /// The delegate that provides info.
    private weak var infoDelegate: ConfigurationServerInfoDelegate?

    /// Creates a new server.
    /// - Parameter xpcServer: The underlying XPC server.
    init(xpcServer: ConfigurationAPIXPCServer) {
        self.xpcServer = xpcServer
    }

    /// Sets the delegate that gets informed of events.
    ///
    /// Note that this method should be called before `run()` is called.
    /// - Parameter delegate: The delegate that gets informed of events.
    func set(delegate: ConfigurationServerDelegate?) async {
        self.delegate = delegate
    }

    /// Sets the delegate that provides info.
    ///
    /// Note that this method should be called before `run()` is called.
    /// - Parameter delegate: The delegate that provides info.
    func set(infoDelegate: ConfigurationServerInfoDelegate?) async {
        self.infoDelegate = infoDelegate
    }

    /// Runs the server.
    func run() async throws {
        try await withLogging(operation: "run", sensitiveError: false, logger: Self.logger) {
            // React to incoming requests in ConfigurationServerXPCHandler.
            await self.xpcServer.set(delegate: self)
            await self.xpcServer.connect()
            Self.logger.notice("XPC listener is connected.")
            // Sleep forever, cancellation will result in a thrown error here.
            while true {
                try await Task.sleep(for: .seconds(60))
            }
        }
    }

    private enum ServerError: Swift.Error, LocalizedError, CustomStringConvertible {
        case noDelegate

        var description: String {
            switch self {
            case .noDelegate:
                return "No delegate"
            }
        }

        var errorDescription: String? {
            self.description
        }
    }
}

extension ConfigurationServer: ConfigurationAPIServerDelegateProtocol {
    func register(
        _ registration: Registration,
        connection: ConfigurationAPIServerToClientConnection
    ) async throws -> ConfigurationUpdate {
        guard let delegate = self.delegate else {
            throw ServerError.noDelegate
        }
        return try await delegate.register(
            connection: connection,
            domain: registration.domainName,
            currentConfiguration: registration.currentConfiguration
        )
    }

    func disconnected(_ connectionID: CloudBoardConfigurationDAPI.ConnectionID) async {
        guard let delegate = self.delegate else {
            Self.logger.error("No delegate in \(#function, privacy: .public).")
            return
        }
        await delegate.disconnected(connectionID: connectionID)
    }

    func successfullyAppliedConfiguration(
        _ success: ConfigurationApplyingSuccess,
        connectionID: ConnectionID
    ) async throws {
        guard let delegate = self.delegate else {
            throw ServerError.noDelegate
        }
        await delegate.succeededApplyingConfiguration(
            id: connectionID,
            revision: success.revisionIdentifier
        )
    }

    func failedToApplyConfiguration(
        _ failure: ConfigurationApplyingFailure,
        connectionID: ConnectionID,
        error: Error
    ) async throws {
        guard let delegate = self.delegate else {
            throw ServerError.noDelegate
        }
        await delegate.failedApplyingConfiguration(
            id: connectionID,
            revision: failure.revisionIdentifier,
            error: .init(message: "\(error)")
        )
    }

    func currentConfigurationVersionInfo(
        connectionID: ConnectionID
    ) async throws -> ConfigurationVersionInfo {
        guard let infoDelegate = self.infoDelegate else {
            throw ServerError.noDelegate
        }
        return await infoDelegate.currentConfigurationVersionInfo(
            connectionID: connectionID
        )
    }
}
