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

import CloudBoardCommon
import CloudBoardJobHelperAPI
import CloudBoardLogging
import InternalGRPC

enum GRPCTransformableError: GRPCStatusTransformable {
    case drainingRequests
    case maxConcurrentRequestsExceeded
    case maxCumulativeRequestBytesExceeded
    case workloadUnhealthy
    case invalidAEAD
    case ohttpDecapsulationFailure
    case ohttpEncapsulationFailure
    case unknownKeyID
    case expiredKey
    case idleTimeoutExceeded
    case protocolError
    case abortedCleanly
    case connectionCancelled

    func makeGRPCStatus() -> GRPCStatus {
        switch self {
        case .drainingRequests:
            return .init(code: .unavailable, message: "CloudBoard is temporarily draining traffic")
        case .maxConcurrentRequestsExceeded:
            return .init(code: .resourceExhausted, message: "Max concurrent requests exceeded")
        case .maxCumulativeRequestBytesExceeded:
            return .init(code: .resourceExhausted, message: "Max cumulative request size exceeded")
        case .workloadUnhealthy:
            return .init(code: .unavailable, message: "Workload is unhealthy")
        case .invalidAEAD:
            return .init(code: .internalError, message: "CloudBoard error: Invalid AEAD")
        case .ohttpDecapsulationFailure:
            return .init(code: .internalError, message: "CloudBoard error: OHTTP decapsulation failure")
        case .ohttpEncapsulationFailure:
            return .init(code: .internalError, message: "CloudBoard error: OHTTP encapsulation failure")
        case .unknownKeyID:
            return .init(code: .internalError, message: "CloudBoard error: unknown key ID")
        case .expiredKey:
            return .init(code: .internalError, message: "CloudBoard error: expired key")
        case .idleTimeoutExceeded:
            return .init(code: .deadlineExceeded, message: "Idle timeout exceeded")
        case .protocolError:
            return .init(code: .failedPrecondition, message: "GRPC protocol misused")
        case .abortedCleanly:
            return .init(code: .cancelled, message: "RPC cleanly aborted by ROPES, no error")
        case .connectionCancelled:
            return .init(code: .cancelled, message: "Connection cancelled")
        }
    }
}

extension GRPCTransformableError: ReportableError {
    var publicDescription: String {
        switch self {
        case .drainingRequests:
            "GRPCTransformableError.drainingRequests"
        case .maxConcurrentRequestsExceeded:
            "GRPCTransformableError.maxConcurrentRequestsExceeded"
        case .maxCumulativeRequestBytesExceeded:
            "GRPCTransformableError.maxCumulativeRequestBytesExceeded"
        case .workloadUnhealthy:
            "GRPCTransformableError.workloadUnhealthy"
        case .invalidAEAD:
            "GRPCTransformableError.invalidAEAD"
        case .ohttpDecapsulationFailure:
            "GRPCTransformableError.ohttpDecapsulationFailure"
        case .ohttpEncapsulationFailure:
            "GRPCTransformableError.ohttpEncapsulationFailure"
        case .unknownKeyID:
            "GRPCTransformableError.unknownKeyID"
        case .expiredKey:
            "GRPCTransformableError.expiredKey"
        case .idleTimeoutExceeded:
            "GRPCTransformableError.idleTimeoutExceeded"
        case .protocolError:
            "GRPCTransformableError.protocolError"
        case .abortedCleanly:
            "GRPCTransformableError.abortedCleanly"
        case .connectionCancelled:
            "GRPCTransformableError.connectionCancelled"
        }
    }
}

extension GRPCTransformableError {
    init(failureReason: FailureReason) {
        switch failureReason {
        case .invalidAEAD:
            self = .invalidAEAD
        case .ohttpDecapsulationFailure:
            self = .ohttpDecapsulationFailure
        case .ohttpEncapsulationFailure:
            self = .ohttpEncapsulationFailure
        case .unknownKeyID:
            self = .unknownKeyID
        case .expiredKey:
            self = .expiredKey
        }
    }
}

extension GRPCTransformableError {
    init(idleTimeoutError: IdleTimeoutError) {
        switch idleTimeoutError {
        case .idleTimeoutExceeded:
            self = .idleTimeoutExceeded
        }
    }
}

extension GRPCTransformableError {
    init(_ invokeWorkloadError: InvokeWorkloadStreamState.Error) {
        switch invokeWorkloadError {
        case .receivedDuplicateSetup,
             .receivedParametersBeforeSetup,
             .receivedDuplicateParameters,
             .receivedChunkBeforeSetup,
             .receivedChunkBeforeParameters,
             .receivedChunkAfterFinal,
             .receivedChunkAfterTermination,
             .receivedDuplicateFinalChunk,
             .receivedSetupAfterTermination,
             .receivedParametersAfterTermination,
             .receivedFinalChunkAfterTermination,
             .receivedDuplicateTermination,
             .receivedTerminationBeforeSetup:
            self = .protocolError
        case .rpcAborted:
            self = .abortedCleanly
        case .unexpectedEndOfStream:
            self = .connectionCancelled
        }
    }
}

extension SessionError: GRPCStatusTransformable {
    func makeGRPCStatus() -> GRPCStatus {
        switch self {
        case .sessionReplayed:
            return .init(code: .permissionDenied, message: "Rejected because of suspected replay attack")
        case .unknownKeyID:
            return .init(
                code: .invalidArgument,
                message: "Rejected because the node/key ID is unknown or the corresponding key has expired"
            )
        }
    }
}
