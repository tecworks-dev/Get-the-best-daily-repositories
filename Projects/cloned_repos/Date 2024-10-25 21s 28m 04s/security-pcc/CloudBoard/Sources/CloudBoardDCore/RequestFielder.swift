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

import AppServerSupport.OSLaunchdJob
import CloudBoardCommon
import CloudBoardJobHelperAPI
import CloudBoardLogging
import CloudBoardMetrics
import Foundation
import os
import Tracing
import XPC

enum RequestFielderError: Error {
    case managedJobNotFound
    case tooManyManagedJobs
    case tornDownBeforeRunning
    case unexpectedTerminationError(Error)
    case illegalStateAfterClientIsRunning(String)
    case illegalStateAfterClientIsConnected(String)
    case illegalStateAfterClientTerminationFailed(String)
    case jobHelperUnavailable(String)
    case monitoringCompletedEarly(Error?)
    case monitoringCompletedFromConnected(Error?)
    case monitoringCompletedMoreThanOnce
    case jobNeverRan
    case setDelegateCalledOnTerminatingInstance
}

public actor RequestFielder: CloudBoardJobHelperInstanceProtocol, Equatable {
    public static func == (lhs: RequestFielder, rhs: RequestFielder) -> Bool {
        return lhs.job.uuid == rhs.job.uuid
    }

    fileprivate static let log: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard", category: "RequestFielder"
    )
    static let cloudBoardLaunchdManagerName = "com.apple.cloudos.cloudboardd"

    private var stateMachine: RequestFielderStateMachine

    public var used: Bool = false

    private let job: MonitoredLaunchdJobInstance

    public var id: UUID {
        return self.job.uuid
    }

    private let metrics: MetricsSystem
    private let tracer: any Tracer

    private var warmupCompletePromise: Promise<Void, Error>

    init(
        _ jobHelperLaunchdJob: ManagedLaunchdJob,
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol,
        metrics: MetricsSystem,
        tracer: any Tracer
    ) throws {
        self.job = try MonitoredLaunchdJobInstance(
            jobHelperLaunchdJob, metrics: metrics
        )
        self.tracer = tracer
        self.stateMachine = RequestFielderStateMachine(
            jobID: self.job.uuid,
            delegate: delegate,
            metrics: metrics,
            tracer: self.tracer
        )
        self.metrics = metrics
        self.warmupCompletePromise = Promise()
    }

    public func run() async throws {
        do {
            Self.log.info("Running job \(self.job.uuid, privacy: .public)")
            for try await state in self.job {
                Self.log.info("""
                Job \(self.job.uuid, privacy: .public) state changed to \
                '\(state, privacy: .public)'
                """)
                switch state {
                case .initialized, .created, .starting:
                    // Nothing to do
                    break
                case .running(let pid):
                    try await self.stateMachine.clientIsRunning(pid: pid)
                case .terminating:
                    self.used = true
                case .terminated(let terminationCondition):
                    self.used = true
                    terminationCondition.emitMetrics(
                        metricsSystem: self.metrics,
                        counterFactory: Metrics.CloudBoardDaemon
                            .CBJobHelperExitCounter.Factory()
                    )
                    await self.stateMachine.clientTerminated()
                case .neverRan:
                    await self.stateMachine.clientTerminated()
                }
            }
        } catch {
            Self.log.error("""
            Error while monitoring job \(self.job.uuid, privacy: .public), \
            no longer monitoring: \
            \(String(unredacted: error), privacy: .public)
            """)
            try await self.stateMachine.monitoringCompleted(error: error)
            throw error
        }
        try await self.stateMachine.monitoringCompleted()
    }

    public func markWarmupComplete(error: Error? = nil) {
        if let error {
            self.warmupCompletePromise.fail(with: error)
        } else {
            self.warmupCompletePromise.succeed()
        }
    }

    public func set(
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol
    ) async throws {
        try await self.stateMachine.set(delegate: delegate)
    }

    public func waitForExit() async throws {
        try await self.stateMachine.waitForTermination()
    }

    public func waitForWarmupComplete() async throws {
        let result: Result<Void, Error>
        do {
            result = try await Future(
                self.warmupCompletePromise
            ).resultWithCancellation
        } catch {
            // current task got cancelled
            Self.log.error(
                "waitForWarmupComplete() warmupCompletePromise task cancelled"
            )
            throw error
        }

        do {
            try result.get()
        } catch {
            Self.log.error("""
            waitForWarmupComplete() warmupCompletePromise returned error: \
            \(String(unredacted: error), privacy: .public)
            """)
            throw error
        }
    }

    public func invokeWorkloadRequest(
        _ request: InvokeWorkloadRequest
    ) async throws {
        switch request {
        case .warmup:
            ()
        case .parameters, .requestChunk:
            self.used = true
        }
        try await self.stateMachine.invokeWorkloadRequest(request)
    }

    public func teardown() async throws {
        self.used = true
        try await self.stateMachine.teardown()
    }
}

internal actor RequestFielderStateMachine {
    enum State: CustomStringConvertible {
        case awaitingJobHelperConnection([InvokeWorkloadRequest])
        case connecting(
            [InvokeWorkloadRequest],
            CloudBoardJobHelperAPIClientDelegateProtocol
        )
        case connected(CloudBoardJobHelperAPIXPCClient, [InvokeWorkloadRequest])
        case terminating
        case terminated
        case monitoringCompleted

        var description: String {
            switch self {
            case .awaitingJobHelperConnection: "awaitingJobHelperConnection"
            case .connecting: "connecting"
            case .connected: "connected"
            case .terminating: "terminating"
            case .terminated: "terminated"
            case .monitoringCompleted: "monitoringCompleted"
            }
        }
    }

    private let jobID: UUID
    internal var remotePID: Int?
    private var delegate: CloudBoardJobHelperAPIClientDelegateProtocol
    private var state: State = .awaitingJobHelperConnection([]) {
        didSet(oldState) {
            RequestFielder.log.trace("""
            RequestFielderStateMachine state changed: \
            \(oldState, privacy: .public) -> \(self.state, privacy: .public)
            """)
        }
    }

    private let metrics: MetricsSystem
    private let tracer: any Tracer

    private let terminationPromise = Promise<Void, Error>()
    private var terminationRequested: Bool = false

    init(
        jobID: UUID,
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol,
        metrics: any MetricsSystem,
        tracer: any Tracer
    ) {
        self.jobID = jobID
        self.delegate = delegate
        self.tracer = tracer
        self.metrics = metrics
    }

    func set(
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol
    ) async throws {
        self.delegate = delegate
        switch self.state {
        case .awaitingJobHelperConnection, .connecting:
            ()
        case .connected(let client, _):
            await client.set(delegate: self.delegate)
        case .terminating, .terminated, .monitoringCompleted:
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            Received request to update delegate on terminating RequestFielder
            """)
            throw RequestFielderError.setDelegateCalledOnTerminatingInstance
        }
    }

    deinit {
        switch state {
        case .awaitingJobHelperConnection, .connecting:
            terminationPromise.fail(
                with: RequestFielderError.tornDownBeforeRunning
            )
        case .connected:
            // This cannot happen if `monitoringCompleted` is called as promised
            fatalError("RequestFielderStateMachine dropped without cleaning up")
        case .terminated, .monitoringCompleted:
            () // terminationPromise should be completed
        case .terminating:
            // if we're hitting this case, neither `monitoringCompleted` has
            // been called (`RequestFielder.run()` invariant violated), nor
            // proper termination sequence has been allowed to complete.
            // In that case, terminationPromise will either leak, but if we
            // complete it here, it will race with ongoing termination.
            fatalError(
                "Must wait for RequestFielderStateMachine to finish terminating"
            )
        }
    }

    func invokeWorkloadRequest(_ request: InvokeWorkloadRequest) async throws {
        try await self.tracer.withSpan(
            OperationNames.clientInvokeWorkloadRequest
        ) { span in
            span.attributes.requestSummary.clientRequestAttributes.jobHelperPID
                = self.remotePID
            span.attributes.requestSummary.clientRequestAttributes.jobID
                = self.jobID.uuidString

            switch self.state {
            case .awaitingJobHelperConnection(let bufferedWorkloadRequests):
                RequestFielder.log.debug("""
                \(self.logMetadata(), privacy: .public) \
                Buffering workload request while waiting for connection \
                to cb_jobhelper"
                """)
                self.state = .awaitingJobHelperConnection(
                    bufferedWorkloadRequests + [request]
                )
            case .connecting(let bufferedWorkloadRequests, let setDelegate):
                RequestFielder.log.debug("""
                \(self.logMetadata(), privacy: .public) \
                Buffering workload request while waiting for connection \
                to cb_jobhelper
                """)
                self.state = .connecting(
                    bufferedWorkloadRequests + [request], setDelegate
                )
            case .connected(let client, let bufferedWorkloadRequests):
                if bufferedWorkloadRequests.isEmpty {
                    // No buffered requests, we can go ahead and forward
                    // the request directly
                    RequestFielder.log.debug("""
                    \(self.logMetadata(), privacy: .public) \
                    Sending workload request to cb_jobhelper
                    """)
                    try await client.invokeWorkloadRequest(request)
                } else {
                    RequestFielder.log.debug("""
                    \(self.logMetadata(), privacy: .public) \
                    Buffering workload request while there are previously \
                    buffered requests to be forwarded to connected \
                    cb_jobhelper
                    """)
                    self.state = .connected(
                        client, bufferedWorkloadRequests + [request]
                    )
                }
            case .terminating, .terminated, .monitoringCompleted:
                RequestFielder.log.error("""
                \(self.logMetadata(), privacy: .public) \
                Cannot forward workload request to cb_jobhelper \
                currently terminating
                """)
                throw RequestFielderError.jobHelperUnavailable(
                    "\(self.state)"
                )
            }
        }
    }

    func clientIsRunning(pid: Int?) async throws {
        self.remotePID = pid
        // Notice-/default-level log to ensure that we have the
        // cb_jobhelper associated with the current request is
        // visible in Splunk
        RequestFielder.log.notice("""
        \(self.logMetadata(), privacy: .public) \
        cb_jobhelper is running
        """)
        switch self.state {
        case .awaitingJobHelperConnection(let bufferedWorkloadRequests):
            let setDelegate = self.delegate
            self.state = .connecting(bufferedWorkloadRequests, setDelegate)
            // Pass the stashed delegate as it may change across
            // await calls
            let client = await self.connect(delegate: setDelegate)
            try await self.clientIsConnected(client: client)
        case .connecting, .connected,
             .terminating, .terminated,
             .monitoringCompleted:
            RequestFielder.log.fault("""
            \(self.logMetadata(), privacy: .public) \
            State machine in unexpected state \
            \(self.state, privacy: .public) after cb_jobhelper state \
            reported to be \"running\"
            """)
            throw RequestFielderError.illegalStateAfterClientIsRunning(
                "\(self.state)"
            )
        }
    }

    func connect(
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol
    ) async -> CloudBoardJobHelperAPIXPCClient {
        RequestFielder.log.debug("""
        \(self.logMetadata(), privacy: .public) \
        Connecting to cb_jobhelper
        """)
        let client = await CloudBoardJobHelperAPIXPCClient.localConnection(
            self.jobID
        )
        await client.set(delegate: delegate)
        await client.connect()
        RequestFielder.log.debug("""
        \(self.logMetadata(), privacy: .public) \
        Connected to cb_jobhelper
        """)
        return client
    }

    private func clientIsConnected(
        client: CloudBoardJobHelperAPIXPCClient
    ) async throws {
        // Our locally stashed delegate could have changed while we were
        // connecting. Make sure the one we have set on the XPC client is up
        // to date before moving forward.
        while case .connecting(let bufferedWorkloadRequests, let setDelegate)
            = self.state, !(setDelegate === delegate) {
            self.state = .connecting(bufferedWorkloadRequests, self.delegate)
            await client.set(delegate: self.delegate)
        }

        switch self.state {
        case .connecting(let bufferedWorkloadRequests, _):
            self.state = .connected(client, bufferedWorkloadRequests)
            if self.terminationRequested {
                // cb_jobhelper was requested to terminate
                await self.teardownConnectedClient(
                    client: client,
                    bufferedWorkloadRequests: bufferedWorkloadRequests
                )
            } else {
                do {
                    while case .connected(_, let bufferedWorkloadRequests)
                        = self.state, let nextBufferedWorkloadRequest
                        = bufferedWorkloadRequests.first {
                        // We can only remove the request from the state after
                        // we have executed the request as we otherwise risk a
                        // race where newly arriving request chunks are sent to
                        // cb_jobhelper ahead of buffered ones.
                        RequestFielder.log.debug("""
                        \(self.logMetadata(), privacy: .public) \
                        Forwarding buffered workload request to cb_jobhelper
                        """)
                        try await client.invokeWorkloadRequest(
                            nextBufferedWorkloadRequest
                        )
                        if case .connected(_, var bufferedWorkloadRequests)
                            = self.state {
                            // Since we have successfully forwarded the request
                            // we can now remove it from the list of buffered
                            // requests
                            bufferedWorkloadRequests.removeFirst()
                            self.state = .connected(
                                client, bufferedWorkloadRequests
                            )
                        }
                    }
                } catch {
                    let outstandingRequests: [InvokeWorkloadRequest]
                        = switch self.state {
                    case .awaitingJobHelperConnection(
                        let bufferedWorkloadRequests
                    ),
                    .connecting(let bufferedWorkloadRequests, _),
                    .connected(_, let bufferedWorkloadRequests):
                        bufferedWorkloadRequests
                    default:
                        []
                    }
                    RequestFielder.log.error("""
                    \(self.logMetadata(), privacy: .public) \
                    Failed to forward buffered workload \
                    request to cb_jobhelper: \
                    \(String(unredacted: error), privacy: .public), \
                    tearing down cloud app and cb_jobhelper
                    """)
                    await self.teardownConnectedClient(
                        client: client,
                        bufferedWorkloadRequests: outstandingRequests
                    )
                }
            }
        case .terminated:
            // We have terminated in the meantime, nothing we can do
            ()
        case .awaitingJobHelperConnection,
             .connected,
             .terminating,
             .monitoringCompleted:
            RequestFielder.log.fault("""
            \(self.logMetadata(), privacy: .public) \
            State machine in unexpected state \
            \(self.state, privacy: .public) after connecting to cb_jobhelper
            """)
            throw RequestFielderError.illegalStateAfterClientIsConnected(
                "\(self.state)"
            )
        }
    }

    func teardown() async throws {
        RequestFielder.log.log("""
        \(self.logMetadata(), privacy: .public) \
        Received request to teardown job
        """)
        if self.terminationRequested {
            RequestFielder.log.info("""
            \(self.logMetadata(), privacy: .public) \
            Job termination already requested, waiting for termination
            """)
            try await self.waitForTermination()
            return
        }

        self.terminationRequested = true

        defer {
            try? self.removeJobs()
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await self.teardownTask() }
            group.addTask { try await self.teardownTimeoutTask() }
            RequestFielder.log.info("""
            \(self.logMetadata(), privacy: .public) \
            Waiting for job termination to complete
            """)
            try await group.next()
            group.cancelAll()
            RequestFielder.log.info("""
            \(self.logMetadata(), privacy: .public) \
            Teardown complete
            """)
        }
    }

    private func teardownTask() async throws {
        switch self.state {
        case .awaitingJobHelperConnection, .connecting:
            RequestFielder.log.info("""
            \(self.logMetadata(), privacy: .public) \
            Received request to teardown cb_jobhelper while not yet connected, \
            waiting for connection
            """)
            try await self.waitForTermination()
        case .connected(let client, let bufferedWorkloadRequests):
            await self.teardownConnectedClient(
                client: client,
                bufferedWorkloadRequests: bufferedWorkloadRequests
            )
            try await self.waitForTermination()
        case .terminating:
            try await self.waitForTermination()
        case .terminated, .monitoringCompleted:
            RequestFielder.log.debug("""
            \(self.logMetadata(), privacy: .public) \
            Ignoring request to teardown cb_jobhelper \
            in state \(self.state, privacy: .public)
            """)
        }
    }

    private func teardownTimeoutTask() async throws {
        RequestFielder.log.info("""
        \(self.logMetadata(), privacy: .public) \
        Teardown timeout task started
        """)
        do {
            try await Task.sleep(for: .seconds(10))
        } catch is CancellationError {
            RequestFielder.log.info("Teardown timeout cancelled")
            throw CancellationError()
        }
        RequestFielder.log.error("""
        \(self.logMetadata(), privacy: .public) \
        Termination of job timed out
        """)
        try self.removeJobs()
    }

    private func removeJobs() throws {
        let (cbJobHelper, cloudApp) = LaunchdJobHelper.fetchManagedLaunchdJobs(
            withUUID: self.jobID, logger: RequestFielder.log
        )
        if let cloudApp {
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            Cloud app still running, removing
            """)
            self.metrics.emit(
                Metrics.RequestFielder.CloudAppTerminateFailureCounter(
                    action: .increment
                )
            )
            try LaunchdJobHelper.removeManagedLaunchdJob(
                job: cloudApp, logger: RequestFielder.log
            )
        }
        if let cbJobHelper {
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper still running, removing
            """)
            self.metrics.emit(
                Metrics.RequestFielder.CBJobHelperTerminateFailureCounter(
                    action: .increment
                )
            )
            try LaunchdJobHelper.removeManagedLaunchdJob(
                job: cbJobHelper, logger: RequestFielder.log
            )
        }
    }

    internal func waitForTermination() async throws {
        do {
            _ = try await Future(self.terminationPromise).resultWithCancellation
        } catch let error as CancellationError {
            throw error
        } catch {
            RequestFielder.log.fault("""
            \(self.logMetadata(), privacy: .public) \
            Unexpected error while waiting for \
            terminationPromise to be fulfiled: \
            \(String(unredacted: error), privacy: .public)
            """)
            throw RequestFielderError.unexpectedTerminationError(error)
        }
    }

    private func teardownConnectedClient(
        client: CloudBoardJobHelperAPIClientToServerProtocol,
        bufferedWorkloadRequests: [InvokeWorkloadRequest]? = nil
    ) async {
        if let bufferedWorkloadRequests, bufferedWorkloadRequests.count > 0 {
            RequestFielder.log.warning("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper requested to terminate with \
            \(bufferedWorkloadRequests.count, privacy: .public) \
            buffered workload requests
            """)
        }
        self.state = .terminating
        RequestFielder.log.info("Sending teardown request to cb_jobhelper")
        do {
            try await client.teardown()
        } catch {
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            client.teardown() returned error: \
            \(String(unredacted: error), privacy: .public)
            """)
        }
    }

    func clientTerminated() async {
        switch self.state {
        case .awaitingJobHelperConnection(let bufferedWorkloadRequests),
             .connecting(let bufferedWorkloadRequests, _):
            if bufferedWorkloadRequests.count > 0 {
                RequestFielder.log.warning("""
                \(self.logMetadata(), privacy: .public) \
                cb_jobhelper has terminated with \
                \(bufferedWorkloadRequests.count, privacy: .public) \
                buffered workload requests
                """)
            }
            self.terminationPromise.succeed()
        case .connected, .terminating:
            self.terminationPromise.succeed()
        case .terminated:
            RequestFielder.log.warning("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper reported to have terminated twice
            """)
        case .monitoringCompleted:
            RequestFielder.log.warning("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper reported to have terminated after monitoring stopped
            """)
        }

        self.state = .terminated
    }

    // This routine is guaranteed to be invoked once we make it to the point
    // of calling RequestFielder.run(). That ensures that the terminationPromise
    // is completed.
    func monitoringCompleted(error: Error? = nil) async throws {
        defer {
            self.state = .monitoringCompleted
        }

        switch self.state {
        case .awaitingJobHelperConnection, .connecting, .terminating:
            if let error {
                RequestFielder.log.error("""
                \(self.logMetadata(), privacy: .public) \
                cb_jobhelper monitoring stopped before receiving \
                termination notification with error: \
                \(String(unredacted: error), privacy: .public)
                """)
            } else {
                RequestFielder.log.error("""
                \(self.logMetadata(), privacy: .public) \
                cb_jobhelper monitoring stopped before receiving termination \
                notification
                """)
            }
            self.terminationPromise.fail(
                with: RequestFielderError.monitoringCompletedEarly(error)
            )
            throw RequestFielderError.monitoringCompletedEarly(error)
        case .connected:
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper monitoring stopped before receiving termination \
            notification (while connected)
            """)
            self.terminationPromise.fail(
                with: RequestFielderError.monitoringCompletedFromConnected(
                    error
                )
            )
            throw RequestFielderError.monitoringCompletedFromConnected(error)
        case .terminated:
            // terminationPromise fulfilled in clientTerminated()
            ()
        case .monitoringCompleted:
            RequestFielder.log.error("""
            \(self.logMetadata(), privacy: .public) \
            cb_jobhelper monitoring reported to have completed twice
            """)
            throw RequestFielderError.monitoringCompletedMoreThanOnce
        }
    }
}

extension RequestFielderStateMachine {
    private func logMetadata() -> CloudBoardDaemonLogMetadata {
        return CloudBoardDaemonLogMetadata(
            jobID: self.jobID,
            rpcID: CloudBoardDaemon.rpcID,
            requestTrackingID: CloudBoardDaemon.requestTrackingID,
            remotePID: self.remotePID
        )
    }
}
