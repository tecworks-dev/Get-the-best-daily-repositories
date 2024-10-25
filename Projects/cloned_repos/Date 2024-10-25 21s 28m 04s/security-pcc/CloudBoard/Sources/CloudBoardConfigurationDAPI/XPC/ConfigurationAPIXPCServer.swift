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

import CloudBoardAsyncXPC
import os

/// An XPC connection to the client.
public final class XPCClientConnection: ConfigurationAPIServerToClientConnection {
    private let connection: CloudBoardAsyncXPCConnection

    public let id: ConnectionID
    public var name: String {
        return self.connection.name
    }

    init(connection: CloudBoardAsyncXPCConnection) {
        self.connection = connection
        self.id = .init(xpcConnection: connection)
    }
}

extension XPCClientConnection: ConfigurationAPIServerToClientProtocol {
    public func applyFallback(_ fallback: FallbackToStaticConfiguration) async throws {
        return try await self.connection.send(
            ConfigurationAPIXPCServerToClientMessages.ApplyFallback(fallback: fallback)
        )
    }

    public func applyConfiguration(_ configuration: UnappliedConfiguration) async throws {
        return try await self.connection.send(
            ConfigurationAPIXPCServerToClientMessages.ApplyConfiguration(configuration: configuration)
        )
    }

    public func entitledDomains() async -> [String] {
        return await self.connection.entitlementValue(
            for: kConfigurationAPIXPCLocalServiceEntitlement
        )
    }
}

public actor ConfigurationAPIXPCServer {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "ConfigurationAPIXPCServer"
    )

    private let listener: CloudBoardAsyncXPCListener
    private weak var delegate: ConfigurationAPIServerDelegateProtocol?

    internal var endpoint: CloudBoardAsyncXPCEndpoint? {
        get async { await self.listener.endpoint }
    }

    public init(
        listener: CloudBoardAsyncXPCListener,
        delegate: ConfigurationAPIServerDelegateProtocol? = nil
    ) {
        self.listener = listener
        self.delegate = delegate
    }
}

extension ConfigurationAPIXPCServer: ConfigurationAPIServerProtocol {
    public func set(delegate: ConfigurationAPIServerDelegateProtocol?) async {
        self.delegate = delegate
    }

    public func connect() async {
        await self.listener.set(delegate: self)
        await self.listener.listen(buildMessageHandlerStore: self.configureHandlers)
    }

    internal func configureHandlers(_ handlers: inout CloudBoardAsyncXPCConnection.MessageHandlerStore) {
        handlers.register(ConfigurationAPIXPCClientToServerMessages.Register.self) { message in
            guard let delegate = await self.delegate else {
                throw ConfigurationAPIError.noDelegateSet
            }
            guard let xpcConnection = CloudBoardAsyncXPCContext.context?.peerConnection else {
                throw ConfigurationAPIError.noXPCConnectionInTaskLocal
            }
            let connection = XPCClientConnection(connection: xpcConnection)
            let domain = message.registration.domainName
            let entitledDomains = await connection.entitledDomains()
            guard entitledDomains.contains(
                message.registration.domainName
            ) else {
                Self.logger.error("""
                Connection \(connection.name, privacy: .public) \
                not entitled for domain \(domain, privacy: .public)
                """)
                throw ConfigurationAPIError.notEntitled(domain)
            }
            do {
                return try await delegate.register(
                    message.registration,
                    connection: connection
                )
            } catch {
                throw ConfigurationAPIError.userPropagatedError("\(error)")
            }
        }
        handlers
            .register(ConfigurationAPIXPCClientToServerMessages.SuccessfullyAppliedConfig.self) { message in
                guard let delegate = await self.delegate else {
                    throw ConfigurationAPIError.noDelegateSet
                }
                guard let connection = CloudBoardAsyncXPCContext.context?.peerConnection else {
                    throw ConfigurationAPIError.noXPCConnectionInTaskLocal
                }
                try await delegate.successfullyAppliedConfiguration(
                    message.success,
                    connectionID: .init(xpcConnection: connection)
                )
                return ExplicitSuccess()
            }
        handlers.register(ConfigurationAPIXPCClientToServerMessages.FailedToApplyConfig.self) { message in
            guard let delegate = await self.delegate else {
                throw ConfigurationAPIError.noDelegateSet
            }
            guard let connection = CloudBoardAsyncXPCContext.context?.peerConnection else {
                throw ConfigurationAPIError.noXPCConnectionInTaskLocal
            }
            struct NoErrorProvidedError: Error {}
            try await delegate.failedToApplyConfiguration(
                message.failure,
                connectionID: .init(xpcConnection: connection),
                error: NoErrorProvidedError()
            )
            return ExplicitSuccess()
        }
        handlers.register(ConfigurationAPIXPCClientToServerMessages.CurrentConfigurationVersionInfo.self) { _ in
            guard let delegate = await self.delegate else {
                throw ConfigurationAPIError.noDelegateSet
            }
            guard let connection = CloudBoardAsyncXPCContext.context?.peerConnection else {
                throw ConfigurationAPIError.noXPCConnectionInTaskLocal
            }
            struct NoErrorProvidedError: Error {}
            return try await delegate.currentConfigurationVersionInfo(
                connectionID: .init(xpcConnection: connection)
            )
        }
    }
}

extension ConfigurationAPIXPCServer: CloudBoardAsyncXPCListenerDelegate {
    public func invalidatedConnection(_ connection: CloudBoardAsyncXPCConnection) async {
        guard let delegate = self.delegate else {
            Self.logger.error("No delegate set.")
            return
        }
        await delegate.disconnected(.init(xpcConnection: connection))
    }
}
