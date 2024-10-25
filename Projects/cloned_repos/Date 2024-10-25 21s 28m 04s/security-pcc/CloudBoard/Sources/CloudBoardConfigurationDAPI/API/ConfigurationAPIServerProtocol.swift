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
import Foundation
import Security

public protocol ConfigurationAPIServerToClientProtocol: AnyObject, Sendable {
    /// Applies the fallback configuration on the client.
    ///
    /// > Important: Applying fallback does not require confirmation back to the server, so do not call
    /// the `successfullyAppliedConfiguration` or `failedToApplyConfiguration` methods in response to receiving
    /// this method.
    /// - Parameter fallback: Information about the fallback.
    func applyFallback(_ fallback: FallbackToStaticConfiguration) async throws

    /// Starts the applying of the configuration on the client.
    ///
    /// > Important: Applying configuration requires confirmation back to the server. You must call
    /// either `successfullyAppliedConfiguration` on success, or `failedToApplyConfiguration` on failure within
    /// a reasonable time window. It is the receiver's responsibility to enforce timeouts, so that a reply
    /// is always delivered. If the receiver crashes, the server will learn about it from the XPC listener and
    /// handle the disconnect appropriately.
    /// - Parameter configuration: Information about the configuration.
    func applyConfiguration(_ configuration: UnappliedConfiguration) async throws
}

public protocol ConfigurationAPIServerProtocol: AnyObject, Sendable {
    func set(delegate: ConfigurationAPIServerDelegateProtocol?) async
    func connect() async
}

/// An identifier of a connection.
public struct ConnectionID: Hashable, Sendable, CustomStringConvertible, ExpressibleByStringLiteral, Comparable {
    private enum Kind: Hashable {
        case xpcConnectionID(CloudBoardAsyncXPCConnection.ID, name: String)
        case string(String)

        static func == (lhs: Kind, rhs: Kind) -> Bool {
            switch (lhs, rhs) {
            case (.xpcConnectionID(let lhsId, _), .xpcConnectionID(let rhsId, _)):
                return lhsId == rhsId
            case (.string(let lhsString), .string(let rhsString)):
                return lhsString == rhsString
            default:
                return false
            }
        }

        func hash(into hasher: inout Hasher) {
            switch self {
            case .xpcConnectionID(let id, _):
                hasher.combine(0)
                hasher.combine(id)
            case .string(let string):
                hasher.combine(1)
                hasher.combine(string)
            }
        }
    }

    private let kind: Kind

    /// Creates a new identifier with the provided XPC connection identifier.
    public init(xpcConnection: CloudBoardAsyncXPCConnection) {
        self.kind = .xpcConnectionID(xpcConnection.id, name: xpcConnection.name)
    }

    /// Creates a new identifier with the provided string.
    public init(string: String) {
        self.kind = .string(string)
    }

    public var description: String {
        switch self.kind {
        case .xpcConnectionID(let xpcConnectionID, let name):
            return "\(xpcConnectionID) (\(name))"
        case .string(let string):
            return string
        }
    }

    public init(stringLiteral value: String) {
        self.init(string: value)
    }

    public static func < (lhs: ConnectionID, rhs: ConnectionID) -> Bool {
        lhs.description < rhs.description
    }
}

/// A connection from the server to the client with a unique identifier.
public protocol ConfigurationAPIServerToClientConnection: ConfigurationAPIServerToClientProtocol {
    var id: ConnectionID { get }
}

/// A delegate of the server that handles events for each connection.
public protocol ConfigurationAPIServerDelegateProtocol: AnyObject, Sendable {
    /// Registers the connection for configuration updates to the provided domain.
    func register(
        _ registration: Registration,
        connection: ConfigurationAPIServerToClientConnection
    ) async throws -> ConfigurationUpdate

    /// Handles a connection disconnecting.
    func disconnected(
        _ connectionID: ConnectionID
    ) async

    /// Handles a connection reporting successful applying of the provided configuration revision.
    func successfullyAppliedConfiguration(
        _ success: ConfigurationApplyingSuccess,
        connectionID: ConnectionID
    ) async throws

    /// Handles a connection reporting failed applying of the provided configuration revision.
    func failedToApplyConfiguration(
        _ failure: ConfigurationApplyingFailure,
        connectionID: ConnectionID,
        error: Error
    ) async throws

    /// Returns the current configuration version info.
    func currentConfigurationVersionInfo(
        connectionID: ConnectionID
    ) async throws -> ConfigurationVersionInfo
}
