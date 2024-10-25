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

import CloudBoardJobAPI
import CloudBoardLogging
import Foundation
import os

struct CloudBoardJobHelperRequestSummary: RequestSummary {
    let operationName: String = "cb_jobhelper response summary"
    let type: String = "RequestSummary"
    let serviceName: String = "cb_jobhelper"
    let namespace: String = "cloudboard"
    var requestPlaintextMetadata: ParametersData.PlaintextMetadata?
    var jobUUID: UUID
    var requestMessageCount: Int = 0
    var responseMessageCount: Int = 0
    var error: Error?
    var startTimeNanos: Int64?
    var endTimeNanos: Int64?

    init(jobUUID: UUID) {
        self.jobUUID = jobUUID
    }

    var requestID: String? {
        return self.requestPlaintextMetadata?.requestID
    }

    var automatedDeviceGroup: String? {
        return self.requestPlaintextMetadata?.automatedDeviceGroup
    }

    mutating func populateRequestMetadata(_ requestMetadata: ParametersData.PlaintextMetadata) {
        self.requestPlaintextMetadata = requestMetadata
    }

    /// NOTE: This value will be logged as public and therefore must not contain public information
    public func log(to logger: Logger) {
        logger.log("""
        ttl=\(self.type, privacy: .public)
        jobID=\(self.jobUUID.uuidString, privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        client.feature_id=\(self.requestPlaintextMetadata?.featureID ?? "", privacy: .public)
        client.bundle_id=\(self.requestPlaintextMetadata?.bundleID ?? "", privacy: .public)
        client.automated_device_group=\(self.automatedDeviceGroup ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        tracing.start_time_unix_nano=\(self.startTimeNanos ?? 0, privacy: .public)
        tracing.end_time_unix_nano=\(self.endTimeNanos ?? 0, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        durationMicros=\(self.durationMicros ?? 0, privacy: .public)
        tracing.status=\(self.status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        requestMessageCount=\(self.requestMessageCount, privacy: .public)
        responseMessageCount=\(self.responseMessageCount, privacy: .public)
        """)
    }
}

struct WorkloadJobManagerCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    let operationName: StaticString
    let serviceName: StaticString = "cb_jobhelper"
    let namespace: StaticString = "cloudboard"

    var logMetadata: CloudBoardJobHelperLogMetadata
    var requestMessageCount: Int
    var responseMessageCount: Int
    var message: StaticString
    var error: Error?

    public init(
        logMetadata: CloudBoardJobHelperLogMetadata,
        requestMessageCount: Int = 0,
        responseMessageCount: Int = 0,
        message: StaticString,
        operationName: StaticString = #function,
        error: Error? = nil
    ) {
        self.logMetadata = logMetadata
        self.requestMessageCount = requestMessageCount
        self.responseMessageCount = responseMessageCount
        self.message = message
        self.operationName = operationName
        if let error {
            self.error = error
        }
    }

    public func log(to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        message=\(self.message, privacy: .public)
        requestMessageCount=\(self.requestMessageCount, privacy: .public)
        responseMessageCount=\(self.responseMessageCount, privacy: .public)
        """)
    }
}

extension WorkloadJobManagerCheckpoint {
    public func logAppTermination(
        terminationMetadata: TerminationMetadata,
        to logger: Logger,
        level: OSLogType
    ) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        requestMessageCount=\(self.requestMessageCount, privacy: .public)
        responseMessageCount=\(self.responseMessageCount, privacy: .public)
        terminationStatusCode=\(terminationMetadata.statusCode.map { String(describing: $0) } ?? "", privacy: .public)
        """)
    }
}

extension PipelinePayload {
    /// This is a public description of the payload to be included in logs and metrics and must not include any
    /// sensitive data
    public var publicDescription: String {
        switch self {
        case .chunk: return "chunk"
        case .endOfInput: return "endOfInput"
        case .oneTimeToken: return "oneTimeToken"
        case .parameters: return "parameters"
        case .teardown: return "teardown"
        case .warmup: return "warmup"
        }
    }
}

extension PipelinePayload: CustomStringConvertible where T == Data {
    /// This is a non-public description and must not be logged publicly
    internal var description: String {
        switch self {
        case .chunk(let encodedRequestChunk): return "chunk bytes \(encodedRequestChunk.chunk.count)"
        case .endOfInput: return "endOfInput"
        case .oneTimeToken: return "oneTimeToken"
        case .parameters(let parameters): return "parameters \(parameters.plaintextMetadata)"
        case .teardown: return "teardown"
        case .warmup(let warmup): return "warmup \(warmup)"
        }
    }
}

extension WorkloadJobManagerCheckpoint {
    /// Log sanitised information about messages coming down the receive request pipeline
    public func logReceiveRequestPipelineMessage(
        pipelineMessage: PipelinePayload<Data>,
        to logger: Logger,
        level: OSLogType
    ) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        message=\(self.message, privacy: .public)
        requestMessageCount=\(self.requestMessageCount, privacy: .public)
        responseMessageCount=\(self.responseMessageCount, privacy: .public)
        pipelineMessage=\(pipelineMessage.publicDescription, privacy: .public)
        pipelineMessageDetailed=\(pipelineMessage)
        """)
    }
}

struct WorkloadJobStateMachineCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    var operationName: StaticString
    let serviceName: StaticString = "cb_jobhelper"
    let namespace: StaticString = "cloudboard"

    var error: Error?

    var logMetadata: CloudBoardJobHelperLogMetadata
    var state: WorkloadJobStateMachine.State
    var newState: WorkloadJobStateMachine.State?

    public init(
        logMetadata: CloudBoardJobHelperLogMetadata,
        state: WorkloadJobStateMachine.State,
        operation: StaticString
    ) {
        self.logMetadata = logMetadata
        self.state = state
        self.operationName = operation
    }

    public func loggingStateChange<Result>(
        to logger: Logger,
        level: OSLogType,
        _ body: () throws -> (Result, WorkloadJobStateMachine.State)
    ) rethrows -> Result {
        do {
            let (result, newState) = try body()
            var checkpoint = self
            checkpoint.newState = newState
            checkpoint.log(to: logger, level: level)
            return result
        } catch {
            var checkpoint = self
            checkpoint.error = error
            checkpoint.log(to: logger, level: level)
            throw error
        }
    }

    func log(to logger: Logger, level: OSLogType) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        state=\(self.state.publicDescription, privacy: .public)
        newState=\(self.newState?.publicDescription ?? "", privacy: .public)
        """)
    }
}

extension WorkloadJobStateMachine.State {
    public var publicDescription: StaticString {
        switch self {
        case .awaitingOneTimeToken: "awaitingOneTimeToken"
        case .awaitingTokenGrantingToken: "awaitingTokenGrantingToken"
        case .terminated: "terminated"
        case .validatedTokenGrantingToken: "validatedTokenGrantingToken"
        }
    }
}

struct CloudboardJobHelperCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    let operationName: StaticString
    let serviceName: StaticString = "cb_jobhelper"
    let namespace: StaticString = "cloudboard"

    var logMetadata: CloudBoardJobHelperLogMetadata
    var message: StaticString
    var error: Error?

    var durationMicros: Int64?

    public init(
        logMetadata: CloudBoardJobHelperLogMetadata,
        message: StaticString,
        operationName: StaticString = #function,
        error: Error? = nil
    ) {
        self.logMetadata = logMetadata
        self.message = message
        self.operationName = operationName
        if let error {
            self.error = error
        }
    }

    public func log(to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        message=\(self.message, privacy: .public)
        """)
    }
}
