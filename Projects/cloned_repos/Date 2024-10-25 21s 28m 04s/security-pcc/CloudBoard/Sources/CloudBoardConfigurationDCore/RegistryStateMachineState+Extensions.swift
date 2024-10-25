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

import CloudBoardConfigurationDAPI
import Foundation

extension RegistryStateMachine.State: CustomStringConvertible {
    var description: String {
        switch self {
        case .idle(configuration: let configuration, connections: let connections):
            return "idle(configuration: \(configuration), connections: \(connections))"
        case .applyingConfig(
            configuration: let configuration,
            completion: _,
            unappliedConnections: let unappliedConnections,
            appliedConnections: let appliedConnections,
            failedToApplyConnections: let failedToApplyConnections,
            disconnectedDuringApplyingConnections: let disconnectedDuringApplyingConnections
        ):
            return "applyingConfig(configuration: \(configuration), unappliedConnections: \(unappliedConnections), appliedConnections: \(appliedConnections), failedToApplyConnections: \(failedToApplyConnections), disconnectedDuringApplyingConnections: \(disconnectedDuringApplyingConnections))"
        case .mutating:
            return "mutating"
        }
    }
}

extension RegistryStateMachine.State: Equatable {
    static func == (lhs: RegistryStateMachine.State, rhs: RegistryStateMachine.State) -> Bool {
        switch (lhs, rhs) {
        case (
            .idle(let lhsConfiguration, let lhsConnections),
            .idle(let rhsConfiguration, let rhsConnections)
        ):
            return lhsConfiguration == rhsConfiguration && lhsConnections == rhsConnections
        case (
            .applyingConfig(
                let lhsConfiguration,
                _,
                let lhsUnappliedConnections,
                let lhsAppliedConnections,
                let lhsFailedToApplyConnections,
                let lhsDisconnectedDuringApplyingConnections
            ),
            .applyingConfig(
                let rhsConfiguration,
                _,
                let rhsUnappliedConnections,
                let rhsAppliedConnections,
                let rhsFailedToApplyConnections,
                let rhsDisconnectedDuringApplyingConnections
            )
        ):
            return lhsConfiguration == rhsConfiguration
                && lhsUnappliedConnections == rhsUnappliedConnections
                && lhsAppliedConnections == rhsAppliedConnections
                && lhsFailedToApplyConnections == rhsFailedToApplyConnections
                && lhsDisconnectedDuringApplyingConnections == rhsDisconnectedDuringApplyingConnections
        case (
            .mutating,
            .mutating
        ):
            return true
        default:
            return false
        }
    }
}

extension RegistryStateMachine.State: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .idle(
            let configuration,
            let connections
        ):
            hasher.combine(0)
            hasher.combine(configuration)
            hasher.combine(connections)
        case .applyingConfig(
            let configuration,
            _,
            let unappliedConnections,
            let appliedConnections,
            let failedToApplyConnections,
            let disconnectedDuringApplyingConnections
        ):
            hasher.combine(1)
            hasher.combine(configuration)
            hasher.combine(unappliedConnections)
            hasher.combine(appliedConnections)
            hasher.combine(failedToApplyConnections)
            hasher.combine(disconnectedDuringApplyingConnections)
        case .mutating:
            hasher.combine(2)
        }
    }
}

extension RegistryStateMachine.DisconnectedAction: Equatable {
    static func == (
        lhs: RegistryStateMachine.DisconnectedAction,
        rhs: RegistryStateMachine.DisconnectedAction
    ) -> Bool {
        switch (lhs, rhs) {
        case (
            .reportDisconnectOneConnection(id: let lhsId),
            .reportDisconnectOneConnection(id: let rhsId)
        ):
            return lhsId == rhsId
        case (
            .reportDisconnectedUnknownConnection(id: let lhsId),
            .reportDisconnectedUnknownConnection(id: let rhsId)
        ):
            return lhsId == rhsId
        case (
            .invokeCompletionWithFailure(_, let lhsError),
            .invokeCompletionWithFailure(_, let rhsError)
        ):
            return lhsError == rhsError
        default:
            return false
        }
    }
}

extension RegistryStateMachine.DisconnectedAction: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .reportDisconnectOneConnection(id: let id):
            hasher.combine(0)
            hasher.combine(id)
        case .reportDisconnectedUnknownConnection(id: let id):
            hasher.combine(1)
            hasher.combine(id)
        case .invokeCompletionWithFailure(_, let error):
            hasher.combine(2)
            hasher.combine(error)
        }
    }
}

private func messageComparator(
    lhs: (DomainConfigurationPackage, any ConfigurationAPIServerToClientConnection),
    rhs: (DomainConfigurationPackage, any ConfigurationAPIServerToClientConnection)
) -> Bool {
    lhs.1.id < rhs.1.id
}

private func areMessagesEqual(
    _ lhsMessages: [(DomainConfigurationPackage, any ConfigurationAPIServerToClientConnection)],
    _ rhsMessages: [(DomainConfigurationPackage, any ConfigurationAPIServerToClientConnection)]
) -> Bool {
    guard lhsMessages.count == rhsMessages.count else {
        return false
    }
    for (lhsItem, rhsItem) in zip(
        lhsMessages.sorted(by: messageComparator),
        rhsMessages.sorted(by: messageComparator)
    ) {
        guard lhsItem.0 == rhsItem.0, lhsItem.1.id == rhsItem.1.id else {
            return false
        }
    }
    return true
}

private func hashMessages(
    _ messages: [(DomainConfigurationPackage, any ConfigurationAPIServerToClientConnection)],
    into hasher: inout Hasher
) {
    for message in messages.sorted(by: messageComparator) {
        hasher.combine(message.0)
        hasher.combine(message.1.id)
    }
}

extension RegistryStateMachine.ApplyConfigurationAction: Equatable {
    static func == (
        lhs: RegistryStateMachine.ApplyConfigurationAction,
        rhs: RegistryStateMachine.ApplyConfigurationAction
    ) -> Bool {
        switch (lhs, rhs) {
        case (.invokeCompletionWithSuccess, .invokeCompletionWithSuccess):
            return true
        case (
            .send(let lhsRevisionIdentifier, let lhsConnectionsToUpdate, let lhsConnectionsUpToDate),
            .send(let rhsRevisionIdentifier, let rhsConnectionsToUpdate, let rhsConnectionsUpToDate)
        ):
            return lhsRevisionIdentifier == rhsRevisionIdentifier && areMessagesEqual(
                lhsConnectionsToUpdate,
                rhsConnectionsToUpdate
            ) && lhsConnectionsUpToDate == rhsConnectionsUpToDate
        default:
            return false
        }
    }
}

extension RegistryStateMachine.ApplyConfigurationAction: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .invokeCompletionWithSuccess:
            hasher.combine(0)
        case .send(
            let revisionIdentifier,
            let messages,
            let connectionsUpToDate
        ):
            hasher.combine(1)
            hasher.combine(revisionIdentifier)
            hashMessages(messages, into: &hasher)
            hasher.combine(connectionsUpToDate)
        }
    }
}

private func messageComparator(
    lhs: any ConfigurationAPIServerToClientConnection,
    rhs: any ConfigurationAPIServerToClientConnection
) -> Bool {
    lhs.id < rhs.id
}

private func areMessagesEqual(
    _ lhsMessages: [any ConfigurationAPIServerToClientConnection],
    _ rhsMessages: [any ConfigurationAPIServerToClientConnection]
) -> Bool {
    guard lhsMessages.count == rhsMessages.count else {
        return false
    }
    for (lhsItem, rhsItem) in zip(
        lhsMessages.sorted(by: messageComparator),
        rhsMessages.sorted(by: messageComparator)
    ) {
        guard lhsItem.id == rhsItem.id else {
            return false
        }
    }
    return true
}

private func hashMessages(
    _ messages: [any ConfigurationAPIServerToClientConnection],
    into hasher: inout Hasher
) {
    for message in messages.sorted(by: messageComparator) {
        hasher.combine(message.id)
    }
}

extension RegistryStateMachine.ApplyFallbackAction: Equatable {
    static func == (
        lhs: RegistryStateMachine.ApplyFallbackAction,
        rhs: RegistryStateMachine.ApplyFallbackAction
    ) -> Bool {
        switch (lhs, rhs) {
        case (.send(let lhsMessages), .send(let rhsMessages)):
            return areMessagesEqual(lhsMessages, rhsMessages)
        }
    }
}

extension RegistryStateMachine.ApplyFallbackAction: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .send(let messages):
            hashMessages(messages, into: &hasher)
        }
    }
}

extension RegistryStateMachine.SucceededApplyingConfigurationAction: Equatable {
    static func == (
        lhs: RegistryStateMachine.SucceededApplyingConfigurationAction,
        rhs: RegistryStateMachine.SucceededApplyingConfigurationAction
    ) -> Bool {
        switch (lhs, rhs) {
        case (.reportError(let lhsError), .reportError(let rhsError)):
            return lhsError == rhsError
        case (
            .reportSuccessOneConnection(let lhsId, let lhsRevision),
            .reportSuccessOneConnection(let rhsId, let rhsRevision)
        ):
            return lhsId == rhsId && lhsRevision == rhsRevision
        case (
            .invokeCompletionWithSuccess(_, let lhsId, let lhsRevision),
            .invokeCompletionWithSuccess(_, let rhsId, let rhsRevision)
        ):
            return lhsId == rhsId && lhsRevision == rhsRevision
        case (
            .invokeCompletionWithFailure(_, let lhsError, let lhsId, let lhsRevision),
            .invokeCompletionWithFailure(_, let rhsError, let rhsId, let rhsRevision)
        ):
            return lhsError == rhsError && lhsId == rhsId && lhsRevision == rhsRevision
        default:
            return false
        }
    }
}

extension RegistryStateMachine.SucceededApplyingConfigurationAction: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .reportError(let error):
            hasher.combine(0)
            hasher.combine(error)
        case .reportSuccessOneConnection(let id, let revision):
            hasher.combine(1)
            hasher.combine(id)
            hasher.combine(revision)
        case .invokeCompletionWithSuccess(_, let id, let revision):
            hasher.combine(2)
            hasher.combine(id)
            hasher.combine(revision)
        case .invokeCompletionWithFailure(
            _,
            error: let error,
            lastConnectionId: let id,
            lastConnectionRevision: let revision
        ):
            hasher.combine(3)
            hasher.combine(error)
            hasher.combine(id)
            hasher.combine(revision)
        }
    }
}

extension RegistryStateMachine.FailedApplyingConfigurationAction: Equatable {
    static func == (
        lhs: RegistryStateMachine.FailedApplyingConfigurationAction,
        rhs: RegistryStateMachine.FailedApplyingConfigurationAction
    ) -> Bool {
        switch (lhs, rhs) {
        case (.reportError(let lhsError), .reportError(let rhsError)):
            return lhsError == rhsError
        case (
            .reportErrorOneConnection(let lhsError, let lhsId, let lhsRevision),
            .reportErrorOneConnection(let rhsError, let rhsId, let rhsRevision)
        ):
            return lhsError == rhsError && lhsId == rhsId && lhsRevision == rhsRevision
        case (
            .invokeCompletionWithFailure(_, let lhsError, let lhsId, let lhsRevision),
            .invokeCompletionWithFailure(_, let rhsError, let rhsId, let rhsRevision)
        ):
            return lhsError == rhsError && lhsId == rhsId && lhsRevision == rhsRevision
        default:
            return false
        }
    }
}

extension RegistryStateMachine.FailedApplyingConfigurationAction: Hashable {
    func hash(into hasher: inout Hasher) {
        switch self {
        case .reportError(let error):
            hasher.combine(0)
            hasher.combine(error)
        case .reportErrorOneConnection(let error, let id, let revision):
            hasher.combine(1)
            hasher.combine(error)
            hasher.combine(id)
            hasher.combine(revision)
        case .invokeCompletionWithFailure(_, let error, let id, let revision):
            hasher.combine(2)
            hasher.combine(error)
            hasher.combine(id)
            hasher.combine(revision)
        }
    }
}
