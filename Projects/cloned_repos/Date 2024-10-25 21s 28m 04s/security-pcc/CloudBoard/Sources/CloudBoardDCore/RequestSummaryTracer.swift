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

import Atomics
import CloudBoardLogging
import CloudBoardMetrics
import Foundation
import NIOHPACK
import os
import ServiceContextModule
import Tracing

enum OperationNames {
    static let invokeWorkload = "workload.invocation"
    static let invokeWorkloadRequest = "workload.invocation.request"
    static let invokeWorkloadResponse = "workload.invocation.response"
    static let waitForWarmupComplete = "workload.waitForWarmupComplete"
    static let clientInvokeWorkloadRequest = "workload.client.invocation.request"
}

/// Tracer to collect and emit "Request Summary" information.
/// When using this tracer every request has a `RequestSummary` entry in the logs.
public final class RequestSummaryTracer: Tracing.Tracer, RequestIDInstrument, Sendable {
    fileprivate static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudBoardRequestSummary"
    )
    private let metrics: any MetricsSystem

    struct ActiveTrace {
        var invokeWorkloadSpan: Span?
        var invokeWorkloadRequestSpans: [Span] = []
        var invokeWorkloadResponseSpans: [Span] = []
        var clientInvokeWorkloadRequestSpans: [Span] = []
        var customSpans: [Span] = []

        init() {}
    }

    private let spanIDGenerator = ManagedAtomic<UInt64>(0)
    private let requestSpans: OSAllocatedUnfairLock<[String: ActiveTrace]> = .init(initialState: [:])

    public init(metrics: any MetricsSystem) {
        self.metrics = metrics
    }

    public func startSpan(
        _ operationName: String,
        context: @autoclosure () -> ServiceContext,
        ofKind kind: SpanKind,
        at instant: @autoclosure () -> some TracerInstant,
        function: String,
        file fileID: String,
        line: UInt
    ) -> Span {
        let context = context()
        let newSpanID = self.spanIDGenerator.loadThenWrappingIncrement(ordering: .relaxed)

        let parentID = context.spanID
        let rpcID = context.rpcID ?? "rpcID not passed through context"
        var nextContext = context
        nextContext.spanID = newSpanID

        if parentID == nil {
            // this is a new root span
            self.requestSpans.withLock {
                assert($0[rpcID] == nil)
                $0[rpcID] = .init()
            }
        }

        let span = Span(
            startTimeNanos: instant().nanosecondsSinceEpoch,
            operationName: operationName,
            kind: kind,
            rpcID: rpcID,
            requestID: context.requestID,
            parentID: parentID,
            spanID: newSpanID,
            context: nextContext,
            tracer: self,
            function: function,
            fileID: fileID,
            line: line,
            logger: Self.logger
        )

        return span
    }

    public func forceFlush() {
        // we can ignore this. we only flush to the logger
    }

    fileprivate func closeSpan(_ span: Span) {
        let rpcID = span.rpcID
        let isRoot = span.parentID == nil

        if isRoot {
            assert(OperationNames.invokeWorkload == span.operationName, "Unexpected span name: \(span.operationName)")
            let trace: ActiveTrace? = self.requestSpans.withLock { requestSpans in
                if var trace = requestSpans.removeValue(forKey: rpcID) {
                    trace.invokeWorkloadSpan = span
                    return trace
                }
                return nil
            }

            guard let trace else {
                Self.logger.error("Could not find any trace for rpcID: \(rpcID, privacy: .public)")
                return
            }

            // Only log invokeWorkload operations for now
            if span.operationName == OperationNames.invokeWorkload {
                CloudBoardDRequestSummary(from: trace).log(to: Self.logger)
                CloudBoardDRequestSummary(from: trace).measure(to: self.metrics)
            }
        } else if span.operationName == OperationNames.invokeWorkloadRequest {
            self.requestSpans.withLock {
                $0[rpcID]?.invokeWorkloadRequestSpans.append(span)
            }
        } else if span.operationName == OperationNames.invokeWorkloadResponse {
            self.requestSpans.withLock {
                $0[rpcID]?.invokeWorkloadResponseSpans.append(span)
            }
        } else if span.operationName == OperationNames.clientInvokeWorkloadRequest {
            self.requestSpans.withLock {
                $0[rpcID]?.clientInvokeWorkloadRequestSpans.append(span)
            }
        } else {
            self.requestSpans.withLock {
                assert($0[rpcID] != nil)
                $0[rpcID]?.customSpans.append(span)
            }
        }
    }
}

extension RequestSummaryTracer {
    public final class Span: Tracing.Span, Sendable {
        private struct Storage: Sendable {
            var recordingState: RecordingState
            var context: ServiceContext
            var events: [Tracing.SpanEvent]
            var status: Tracing.SpanStatus?
            var errors: [any Error]
            var attributes: Tracing.SpanAttributes
            var links: [Tracing.SpanLink]
            var operationName: String
        }

        enum RecordingState {
            case open(RequestSummaryTracer)
            case closed(endTimeNanos: UInt64)
        }

        let startTimeNanos: UInt64
        let kind: Tracing.SpanKind

        let rpcID: String
        let requestID: String?
        let spanID: UInt64
        let parentID: UInt64?

        let function: String
        let fileID: String
        let line: UInt

        var errors: [any Error] {
            self.storage.withLock { $0.errors }
        }

        var status: Tracing.SpanStatus? {
            self.storage.withLock { $0.status }
        }

        private let storage: OSAllocatedUnfairLock<Storage>

        private let logger: Logger?

        init(
            startTimeNanos: UInt64,
            operationName: String,
            kind: Tracing.SpanKind,
            rpcID: String,
            requestID: String?,
            parentID: UInt64?,
            spanID: UInt64,
            context: ServiceContext,
            tracer: RequestSummaryTracer,
            function: String,
            fileID: String,
            line: UInt,
            logger: Logger?
        ) {
            self.startTimeNanos = startTimeNanos
            self.kind = kind

            self.rpcID = rpcID
            self.requestID = requestID
            self.parentID = parentID
            self.spanID = spanID

            self.function = function
            self.fileID = fileID
            self.line = line
            self.storage = .init(
                initialState:
                .init(
                    recordingState: .open(tracer),
                    context: context,
                    events: [],
                    status: nil,
                    errors: [],
                    attributes: .init(),
                    links: [],
                    operationName: operationName
                )
            )
            self.logger = logger
        }

        deinit {
            self.storage.withLock { storage in
                switch storage.recordingState {
                case .closed:
                    break
                case .open:
                    let rpcID = storage.context.rpcID ?? ""
                    self.logger?
                        .fault(
                            "[rpcID: \(rpcID, privacy: .public), parentID: \(self.parentID ?? 0, privacy: .public), spanID: \(self.spanID, privacy: .public)] Reference to unended span dropped"
                        )
                }
            }
        }

        public var isRecording: Bool {
            self.storage.withLock { storage -> Bool in
                switch storage.recordingState {
                case .closed:
                    false
                case .open:
                    true
                }
            }
        }

        public var operationName: String {
            get { self.storage.withLock { $0.operationName } }
            set { self.storage.withLock { $0.operationName = newValue } }
        }

        public var context: ServiceContext {
            get { self.storage.withLock { $0.context } }
            set { self.storage.withLock { $0.context = newValue } }
        }

        public var attributes: Tracing.SpanAttributes {
            get { self.storage.withLock { $0.attributes } }
            set { self.storage.withLock { $0.attributes = newValue } }
        }

        public func setStatus(_ status: Tracing.SpanStatus) {
            self.storage.withLock { $0.status = status }
        }

        public func addEvent(_ event: Tracing.SpanEvent) {
            self.storage.withLock { $0.events.append(event) }
        }

        public func recordError(
            _ error: any Error,
            attributes _: Tracing.SpanAttributes,
            at _: @autoclosure () -> some Tracing.TracerInstant
        ) {
            self.storage.withLock { $0.errors.append(error) }
        }

        public func addLink(_ link: Tracing.SpanLink) {
            self.storage.withLock { $0.links.append(link) }
        }

        public func end(at instant: @autoclosure () -> some TracerInstant) {
            let tracer = self.storage.withLock { storage -> RequestSummaryTracer? in
                switch storage.recordingState {
                case .open(let tracer):
                    storage.recordingState = .closed(endTimeNanos: instant().nanosecondsSinceEpoch)
                    return tracer
                case .closed:
                    let rpcID = storage.context.rpcID ?? ""
                    self.logger?
                        .fault(
                            "[rpcID: \(rpcID, privacy: .public), parentID: \(self.parentID ?? 0, privacy: .public), spanID: \(self.spanID, privacy: .public)] Span already closed"
                        )
                    return nil
                }
            }
            if let tracer {
                tracer.closeSpan(self)
            }
        }

        fileprivate var endTimeNanos: UInt64 {
            self.storage.withLock { storage in
                guard case .closed(let stopTimeNanos) = storage.recordingState else {
                    let rpcID = storage.context.rpcID ?? ""
                    self.logger?
                        .fault(
                            "[rpcID: \(rpcID, privacy: .public), parentID: \(self.parentID ?? 0, privacy: .public), spanID: \(self.spanID, privacy: .public)] endTimeNanos queried for closed span"
                        )
                    return 0
                }
                return stopTimeNanos
            }
        }
    }
}

struct SpanIDKey: ServiceContextKey {
    typealias Value = UInt64

    static var nameOverride: String? { "spanID" }
}

extension ServiceContext {
    fileprivate var spanID: UInt64? {
        get { self[SpanIDKey.self] }
        set { self[SpanIDKey.self] = newValue }
    }
}

extension CloudBoardDRequestSummary {
    public init(from trace: RequestSummaryTracer.ActiveTrace) {
        var summary = CloudBoardDRequestSummary()
        summary.populate(invokeWorkloadSpan: trace.invokeWorkloadSpan)
        summary.populate(invokeWorkloadRequestSpans: trace.invokeWorkloadRequestSpans)
        summary.populate(invokeWorkloadResponseSpans: trace.invokeWorkloadResponseSpans)
        summary.populate(clientInvokeWorkloadRequestSpans: trace.clientInvokeWorkloadRequestSpans)
        self = summary
    }
}

struct CloudBoardDRequestSummary: RequestSummary {
    var operationName: String = OperationNames.invokeWorkload.description
    var type: String = "RequestSummary"
    var serviceName: String = "cloudboardd"
    var namespace: String = "cloudboard"
    var rpcID: String?
    var headers: HPACKHeaders?
    var startTimeNanos: Int64?
    var endTimeNanos: Int64?
    var error: Error?
    var requestID: String?
    var automatedDeviceGroup: String?
    var requestChunkCount: Int = 0
    var requestChunkTotalSize: Int = 0
    var requestFinalChunkSeen: Bool = false
    var requestWorkload: String?
    var requestBundleID: String?
    var requestFeatureID: String?
    var responseChunkCount: Int = 0
    var responseFinalChunkSeen: Bool = false
    var receivedSetup: Bool = false
    var receivedParameters: Bool = false
    var jobID: String?
    var jobHelperPID: Int?
    var connectionCancelled: Bool = false
    var ropesTerminationCode: Int?
    var ropesTerminationReason: String?

    private enum CodingKeys: String, CodingKey {
        case spanID
        case operationName = "tracing.name"
        case type = "tracing.type"
        case serviceName = "service.name"
        case namespace = "service.namespace"
        case rpcID
        case headers = "requestHeaders"
        case durationMicros
        case error
        case requestID
        case automatedDeviceGroup
        case requestChunkCount
        case requestFinalChunkSeen
        case responseChunkCount
        case responseFinalChunkSeen
        case receivedSetup
        case receivedParameters
        case jobID
        case jobHelperPID
        case status
        case connectionCancelled
        case ropesTerminationCode
        case ropesTerminationReason
    }

    public init() {}

    mutating func populate(invokeWorkloadSpan: RequestSummaryTracer.Span?) {
        if let span = invokeWorkloadSpan {
            self.operationName = span.operationName
            self.rpcID = span.rpcID
            if let workloadRequestHeaders = span.attributes.requestSummary.invocationAttributes
                .invocationRequestHeaders {
                self.headers = workloadRequestHeaders
            }
            self.connectionCancelled = span.attributes.requestSummary.invocationAttributes.connectionCancelled ?? false

            self.startTimeNanos = Int64(span.startTimeNanos)
            self.endTimeNanos = Int64(span.endTimeNanos)

            if let error = span.errors.first {
                self.populate(error: error)
            }
        }
    }

    mutating func populate(invokeWorkloadRequestSpans: [RequestSummaryTracer.Span]) {
        var invokeWorkloadRequestChunkSizes: [Int] = []
        var invokeWorkloadRequestFinalChunkSeen = false
        var receivedSetup = false
        var receivedParameters = false

        for span in invokeWorkloadRequestSpans {
            if let requestID = span.requestID, !requestID.isEmpty,
               self.requestID == nil {
                self.requestID = requestID
            }

            if let requestChunkSize = span.attributes.requestSummary.workloadRequestAttributes.chunkSize {
                invokeWorkloadRequestChunkSizes.append(requestChunkSize)
                self.requestChunkTotalSize += requestChunkSize
            }
            if span.attributes.requestSummary.workloadRequestAttributes.isFinal ?? false {
                invokeWorkloadRequestFinalChunkSeen = true
            }
            if span.attributes.requestSummary.workloadRequestAttributes.receivedSetup ?? false {
                receivedSetup = true
            }
            if span.attributes.requestSummary.workloadRequestAttributes.receivedParameters ?? false {
                receivedParameters = true
            }
            if let workload = span.attributes.requestSummary.workloadRequestAttributes.workload {
                self.requestWorkload = workload
            }
            if let featureID = span.attributes.requestSummary.workloadRequestAttributes.featureID {
                self.requestFeatureID = featureID
            }
            if let bundleID = span.attributes.requestSummary.workloadRequestAttributes.bundleID {
                self.requestBundleID = bundleID
            }
            if let automatedDeviceGroup = span.attributes.requestSummary.workloadRequestAttributes
                .automatedDeviceGroup {
                self.automatedDeviceGroup = automatedDeviceGroup
            }

            if let ropesTerminationCode = span.attributes.requestSummary.workloadRequestAttributes
                .ropesTerminationCode {
                self.ropesTerminationCode = ropesTerminationCode
            }
            if let ropesTerminationReason = span.attributes.requestSummary.workloadRequestAttributes
                .ropesTerminationReason {
                self.ropesTerminationReason = ropesTerminationReason
            }
        }
        self.requestChunkCount = invokeWorkloadRequestChunkSizes.count
        self.requestFinalChunkSeen = invokeWorkloadRequestFinalChunkSeen
        self.receivedSetup = receivedSetup
        self.receivedParameters = receivedParameters
    }

    mutating func populate(invokeWorkloadResponseSpans: [RequestSummaryTracer.Span]) {
        // There should be only one
        let span = invokeWorkloadResponseSpans.first
        self.responseChunkCount = span?.attributes.requestSummary.responseChunkAttributes.chunksCount ?? 0
        self.responseFinalChunkSeen = span?.attributes.requestSummary.responseChunkAttributes.isFinal ?? false
    }

    mutating func populate(clientInvokeWorkloadRequestSpans: [RequestSummaryTracer.Span]) {
        if let jobID = clientInvokeWorkloadRequestSpans
            .compactMap({ $0.attributes.requestSummary.clientRequestAttributes.jobID }).first {
            self.jobID = jobID
        }
        if let jobHelperPID = clientInvokeWorkloadRequestSpans
            .compactMap({ $0.attributes.requestSummary.clientRequestAttributes.jobHelperPID }).first {
            self.jobHelperPID = jobHelperPID
        }
    }

    func log(to logger: Logger) {
        logger.log("""
        ttl=RequestSummary
        request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.trace_id=\(self.requestID?.replacingOccurrences(of: "-", with: "").lowercased() ?? "", privacy: .public)
        tracing.start_time_unix_nano=\(self.startTimeNanos ?? 0, privacy: .public)
        tracing.end_time_unix_nano=\(self.endTimeNanos ?? 0, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        request.duration_ms=\(self.durationMicros.map { String($0 / 1000) } ?? "", privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        rpcId=\(self.rpcID ?? "", privacy: .public)
        requestChunkCount=\(self.requestChunkCount, privacy: .public)
        requestChunkTotalSize=\(self.requestChunkTotalSize, privacy: .public)
        requestFinalChunkSeen=\(self.requestFinalChunkSeen, privacy: .public)
        requestWorkload=\(self.requestWorkload ?? "", privacy: .public)
        client.feature_id=\(self.requestFeatureID ?? "", privacy: .public)
        client.bundle_id=\(self.requestBundleID ?? "", privacy: .public)
        client.automated_device_group=\(self.automatedDeviceGroup ?? "", privacy: .public)
        responseChunkCount=\(self.responseChunkCount, privacy: .public)
        responseFinalChunkSeen=\(self.responseFinalChunkSeen, privacy: .public)
        receivedSetup=\(self.receivedSetup, privacy: .public)
        receivedParameters=\(self.receivedParameters, privacy: .public)
        jobID=\(self.jobID ?? "", privacy: .public)
        jobHelperPID=\(self.jobHelperPID ?? 0, privacy: .public)
        connectionCancelled=\(self.connectionCancelled, privacy: .public)
        ropesTerminationCode=\(self.ropesTerminationCode.map { "\($0)" } ?? "", privacy: .public)
        ropesTerminationReason=\(self.ropesTerminationReason.map { "\($0)" } ?? "", privacy: .public)
        """)
    }

    func measure(to metrics: any MetricsSystem) {
        metrics.emit(Metrics.CloudBoardProvider.RequestCounter(
            action: .increment,
            automatedDeviceGroup: !(self.automatedDeviceGroup?.isEmpty ?? true)
        ))
        if let durationMicros {
            if let error {
                metrics.emit(Metrics.CloudBoardProvider.FailedRequestCounter(
                    action: .increment,
                    failureReason: error,
                    automatedDeviceGroup: !(self.automatedDeviceGroup?.isEmpty ?? true)
                ))
                metrics.emit(Metrics.CloudBoardProvider.RequestTimeHistogram(
                    duration: .microseconds(durationMicros),
                    featureId: self.requestFeatureID,
                    bundleId: self.requestBundleID,
                    automatedDeviceGroup: !(self.automatedDeviceGroup?.isEmpty ?? true),
                    onlySetupReceived: self.receivedSetup && !self.receivedParameters,
                    failureReason: error
                ))
            } else {
                metrics.emit(Metrics.CloudBoardProvider.RequestTimeHistogram(
                    duration: .microseconds(durationMicros),
                    featureId: self.requestFeatureID,
                    bundleId: self.requestBundleID,
                    automatedDeviceGroup: !(self.automatedDeviceGroup?.isEmpty ?? true),
                    onlySetupReceived: self.receivedSetup && !self.receivedParameters
                ))
            }
        } else {
            RequestSummaryTracer.logger.fault("""
            RequestSummary duration could not be determined. Top-level span likely not closed correctly.
            request.uuid=\(self.requestID ?? "UNKNOWN", privacy: .public)
            jobID=\(self.jobID ?? "", privacy: .public)
            """)
        }
    }
}

extension HPACKHeaders: @retroactive Encodable {
    struct HeaderJSONFormat: Encodable {
        var key: String
        var value: String
    }

    public func encode(to encoder: any Encoder) throws {
        let formatedHeaders = self.map { HeaderJSONFormat(key: $0.name, value: $0.value) }
        try formatedHeaders.encode(to: encoder)
    }
}
