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

//  Copyright © 2023 - 2024 Apple Inc. All rights reserved.

import AppServerSupport.OSLaunchdJob
import CloudBoardCommon
import CloudBoardJobAPI
import CloudBoardJobHelperAPI
import CloudBoardLogging
import CloudBoardMetrics
import Foundation
import InternalSwiftProtobuf
import os

enum CloudBoardAppWorkloadError: Error {
    case tornDownBeforeRunning
    case unexpectedTerminationError(Error)
    case illegalStateAfterClientIsRunning(String)
    case illegalStateAfterClientIsConnected(String)
    case illegalStateAfterClientTerminationFailed(String)
    case cloudAppUnavailable(String)
    case monitoringCompletedEarly(Error?)
    case monitoringCompletedMoreThanOnce
    case jobNeverRan
}

actor CloudBoardAppWorkload: CloudBoardJobHelperWorkload {
    private var stateMachine: CloudBoardAppStateMachine

    private let job: MonitoredLaunchdJobInstance
    private let machServiceName: String
    private var delegate: CloudBoardJobAPIClientDelegateProtocol

    private let log: Logger
    private(set) var remotePID: Int?
    private var requestID: String?
    private let metrics: MetricsSystem

    init(
        managedJob: ManagedLaunchdJob,
        machServiceName: String,
        log: Logger,
        delegate: CloudBoardJobAPIClientDelegateProtocol,
        metrics: any MetricsSystem,
        jobUUID: UUID
    ) throws {
        self.log = log

        // Create a new job instance
        self.job = try MonitoredLaunchdJobInstance(managedJob, uuid: jobUUID, metrics: metrics)
        self.machServiceName = machServiceName
        self.delegate = delegate
        self.stateMachine = CloudBoardAppStateMachine(
            jobID: self.job.uuid,
            machServiceName: machServiceName,
            log: log,
            delegate: delegate
        )
        self.metrics = metrics
    }

    func run() async throws {
        await self.stateMachine.start()
        CloudBoardAppWorkloadCheckpoint(
            jobID: self.job.uuid,
            requestID: self.requestID,
            remotePID: self.remotePID,
            message: "Running job",
            state: nil
        ).log(to: self.log, level: .default)
        var lastKnownJobState: MonitoredLaunchdJobInstance.AsyncIterator.State? = nil
        defer {
            CloudBoardAppWorkloadCheckpoint(
                jobID: self.job.uuid,
                requestID: self.requestID,
                remotePID: self.remotePID,
                message: "Job finished",
                state: lastKnownJobState
            ).log(to: self.log, level: .default)
        }
        do {
            for try await state in self.job {
                lastKnownJobState = state
                CloudBoardAppWorkloadCheckpoint(
                    jobID: self.job.uuid,
                    requestID: self.requestID,
                    remotePID: self.remotePID,
                    message: "Job state changed",
                    state: state
                ).log(to: self.log, level: .info)
                switch state {
                case .initialized, .created, .starting, .terminating:
                    // Nothing to do
                    ()
                case .running(let pid):
                    self.remotePID = pid
                    try await self.stateMachine.clientIsRunning(pid: pid)
                case .terminated(let terminationCondition):
                    terminationCondition.emitMetrics(
                        metricsSystem: self.metrics,
                        counterFactory: Metrics.Workload.CloudAppExitCounter.Factory()
                    )

                    let statusCode: Int? = if case .exited(let status) = terminationCondition {
                        switch status {
                        case .osStatus(_, let code):
                            code
                        case .wait4Status(let code):
                            code
                        case .unknown:
                            nil
                        @unknown default:
                            nil
                        }
                    } else {
                        nil
                    }
                    await self.stateMachine.clientTerminated(statusCode: statusCode)
                case .neverRan:
                    await self.stateMachine.clientTerminated(statusCode: nil)
                @unknown default:
                    // Nothing to do
                    ()
                }
            }
        } catch {
            CloudBoardAppWorkloadCheckpoint(
                jobID: self.job.uuid,
                requestID: self.requestID,
                remotePID: self.remotePID,
                message: "Error while monitoring CloudApp, no longer monitoring",
                state: lastKnownJobState,
                error: error
            ).log(to: self.log, level: .error)
            try await self.stateMachine.monitoringCompleted(error: error)
            throw error
        }
        try await self.stateMachine.monitoringCompleted()
    }

    public func provideInput(_ data: Data?, isFinal: Bool) async throws {
        try await self.stateMachine.provideInput(data, isFinal: isFinal)
    }

    public func endOfInput() async throws {
        try await self.stateMachine.endOfInput()
    }

    public func warmup(_ warmupData: WarmupData) async throws {
        try await self.stateMachine.warmup(warmupData)
    }

    public func parameters(_ parametersData: ParametersData) async throws {
        self.requestID = parametersData.plaintextMetadata.requestID
        try await self.stateMachine.parameters(parametersData)
    }

    public func teardown() async throws {
        try await self.stateMachine.teardown()
    }
}

public actor CloudBoardAppStateMachine {
    // NOTE: The description of this type is publicly logged and/or included in metric dimensions and therefore MUST not
    // contain sensitive data.
    enum State: CustomStringConvertible {
        case awaitingAppConnection([(data: Data?, isFinal: Bool)], WarmupData?, ParametersData?)
        case connecting([(data: Data?, isFinal: Bool)], WarmupData?, ParametersData?)
        case connected(CloudBoardJobAPIXPCClient)
        case terminating
        case terminated
        case monitoringCompleted

        // NOTE: This description is publicly logged and/or included in metric dimensions and therefore MUST not contain
        // sensitive data.
        var description: String {
            switch self {
            case .awaitingAppConnection: "awaitingAppConnection"
            case .connecting: "connecting"
            case .connected: "connected"
            case .terminating: "terminating"
            case .terminated: "terminated"
            case .monitoringCompleted: "monitoringCompleted"
            }
        }
    }

    private let machServiceName: String
    private let jobID: UUID
    private var remotePID: Int?
    private var requestID: String?
    private var delegate: CloudBoardJobAPIClientDelegateProtocol

    private let log: Logger
    private var state: State = .awaitingAppConnection([], nil, nil) {
        didSet(oldState) {
            self.log.trace("""
            jobID=\(self.jobID, privacy: .public)
            message=\("state changed", privacy: .public)
            oldState=\(oldState, privacy: .public)
            state=\(self.state, privacy: .public)
            """)
        }
    }

    private var started: Bool = false
    private let terminationPromise = Promise<Void, Error>()
    private var terminationRequested: Bool = false

    init(jobID: UUID, machServiceName: String, log: Logger, delegate: CloudBoardJobAPIClientDelegateProtocol) {
        self.jobID = jobID
        self.machServiceName = machServiceName
        self.log = log
        self.delegate = delegate
    }

    deinit {
        if !started {
            terminationPromise.fail(with: CloudBoardAppWorkloadError.tornDownBeforeRunning)
        }
    }

    func start() async {
        self.started = true
    }

    func provideInput(_ data: Data?, isFinal: Bool) async throws {
        switch self.state {
        case .awaitingAppConnection(
            let bufferedInputChunks,
            let warmupData,
            let parametersData
        ):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "buffering input data while waiting for connection to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            self.state = .awaitingAppConnection(
                bufferedInputChunks + [(data, isFinal)], warmupData, parametersData
            )
        case .connecting(let bufferedInputChunks, let warmupData, let parametersData):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "buffering input data while waiting for connection to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            self.state = .connecting(
                bufferedInputChunks + [(data, isFinal)],
                warmupData,
                parametersData
            )
        case .connected(let client):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "sending input data to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            try await client.provideInput(data, isFinal: isFinal)
        case .terminating, .terminated, .monitoringCompleted:
            if data != nil {
                let error = CloudBoardAppWorkloadError.cloudAppUnavailable("\(self.state)")
                CloudBoardAppStateMachineCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Cannot forward input to CloudApp currently terminating",
                    state: self.state,
                    error: error
                ).log(to: self.log, level: .error)
                throw error
            } else {
                CloudBoardAppStateMachineCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "CloudApp currently terminating, no need to pass empty chunk",
                    state: self.state
                ).log(to: self.log, level: .debug)
            }
        }
    }

    func endOfInput() async throws {
        switch self.state {
        case .awaitingAppConnection(
            let bufferedInputChunks,
            let warmupData,
            let parametersData
        ):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "buffering end of input notification while waiting for connection to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            self.state = .awaitingAppConnection(
                bufferedInputChunks + [(nil, true)], warmupData, parametersData
            )
        case .connecting(let bufferedInputChunks, let warmupData, let parametersData):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "buffering end of input notification while waiting for connection to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            self.state = .connecting(
                bufferedInputChunks + [(nil, true)],
                warmupData,
                parametersData
            )
        case .connected(let client):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "sending end of input notification to CloudApp",
                state: self.state
            ).log(to: self.log, level: .debug)
            try await client.provideInput(nil, isFinal: true)
        case .terminating, .terminated, .monitoringCompleted:
            let error = CloudBoardAppWorkloadError.cloudAppUnavailable("\(self.state)")
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Cannot forward end of input notification to CloudApp currently terminating",
                state: self.state,
                error: error
            ).log(to: self.log, level: .error)
            throw error
        }
    }

    func clientIsRunning(pid: Int?) async throws {
        self.remotePID = pid
        // Notice-/default-level log to ensure that we have the cb_jobhelper associated with the current request is
        // visible in Splunk
        CloudBoardAppStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            message: "cloud app is running",
            state: self.state
        ).log(to: self.log, level: .default)
        switch self.state {
        case .awaitingAppConnection(let bufferedInputData, let warmupData, let parametersData):
            self.state = .connecting(bufferedInputData, warmupData, parametersData)
            try await self.connect()
        case .connecting, .connected,
             .terminating, .terminated, .monitoringCompleted:
            let error = CloudBoardAppWorkloadError.illegalStateAfterClientIsRunning("\(self.state)")
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "state machine in unexpected state after CloudApp state reported to be \"running\"",
                state: self.state,
                error: error
            ).log(to: self.log, level: .fault)
            throw error
        }
    }

    private func connect() async throws {
        CloudBoardAppStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            message: "Connecting to CloudApp",
            state: self.state
        ).log(to: self.log, level: .info)
        let client = await CloudBoardJobAPIXPCClient.localConnectionWithUUID(
            machServiceName: self.machServiceName,
            uuid: self.jobID
        )
        await client.set(delegate: self.delegate)
        await client.connect()
        CloudBoardAppStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            message: "Connected to CloudApp",
            state: self.state
        ).log(to: self.log, level: .info)
        try await self.clientIsConnected(client: client)
    }

    func warmup(_ warmupData: WarmupData) async throws {
        switch self.state {
        case .awaitingAppConnection(let data, _, let parametersData):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received warmup message",
                state: self.state
            ).log(to: self.log, level: .default)
            self.state = .awaitingAppConnection(data, warmupData, parametersData)
        case .connecting(let data, _, let parametersData):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received warmup message",
                state: self.state
            ).log(to: self.log, level: .default)
            self.state = .connecting(data, warmupData, parametersData)
        case .terminated:
            // This is a bit of a race but it's basically fine: we received this message as
            // we were tearing down. Do nothing.
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received warmup message after cloud app terminated, message will be dropped",
                state: self.state
            ).log(to: self.log, level: .error)
        case .connected(let client):
            try await client.warmup(details: .init(warmupData))
        case .terminating, .monitoringCompleted:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received warmup data while CloudApp is terminating or monitoring completed",
                state: self.state
            ).log(to: self.log, level: .fault)
        }
    }

    func parameters(_ parametersData: ParametersData) async throws {
        self.requestID = parametersData.plaintextMetadata.requestID
        switch self.state {
        case .awaitingAppConnection(let data, let warmupData, _):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received parameters message",
                state: self.state
            ).log(to: self.log, level: .default)
            self.state = .awaitingAppConnection(data, warmupData, parametersData)
        case .connecting(let data, let warmupData, _):
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received parameters message",
                state: self.state
            ).log(to: self.log, level: .default)
            self.state = .connecting(data, warmupData, parametersData)
        case .terminated:
            // This is a bit of a race but it's basically fine: we received this message as
            // we were tearing down. Do nothing.
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received parameters message after cloud app terminated, message will be dropped",
                state: self.state
            ).log(to: self.log, level: .error)
        case .connected(let client):
            try await client.receiveParameters(parametersData: parametersData)
        case .terminating, .monitoringCompleted:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received parameters data while CloudApp is terminating or monitoring completed",
                state: self.state
            ).log(to: self.log, level: .fault)
        }
    }

    // Expected to be invoked from .connectingPendingWarmupData or .connecting states
    func clientIsConnected(client: CloudBoardJobAPIXPCClient) async throws {
        switch self.state {
        case .connecting:
            if let warmupData = state.warmupData {
                CloudBoardAppStateMachineCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Client is connected, forwarding previously received warmup data",
                    state: self.state
                ).log(to: self.log, level: .default)
                try await client.warmup(details: .init(warmupData))
            }
            if let parametersData = state.parametersData {
                try await client.receiveParameters(parametersData: parametersData)
                CloudBoardAppStateMachineCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Client is connected, forwarding previously received parameters data",
                    state: self.state
                ).log(to: self.log, level: .default)
            }
            if self.terminationRequested {
                // the CloudApp was requested to terminate
                // note the bufferedInputChunks here might be stale, but we don't
                // actually use the contained data so this is fine.
                try await self.teardownConnectedClient(
                    client: client,
                    bufferedInputChunks: self.state.bufferedInputChunks
                )
            } else {
                while let (data, isFinal) = state.nextBufferedChunk() {
                    CloudBoardAppStateMachineCheckpoint(
                        logMetadata: self.logMetadata(),
                        message: "forwarding buffered input chunk to CloudApp",
                        state: self.state
                    ).log(to: self.log, level: .debug)
                    try await client.provideInput(data, isFinal: isFinal)
                }

                switch self.state {
                case .connecting:
                    self.state = .connected(client)
                case .terminated, .terminating, .monitoringCompleted:
                    // We have terminated in the meantime, nothing we can do
                    ()
                case .awaitingAppConnection,
                     .connected:
                    let error = CloudBoardAppWorkloadError.illegalStateAfterClientIsConnected("\(self.state)")
                    CloudBoardAppStateMachineCheckpoint(
                        logMetadata: self.logMetadata(),
                        message: "State machine in unexpected state after connecting to CloudApp",
                        state: self.state,
                        error: error
                    ).log(to: self.log, level: .fault)
                    throw error
                }
            }
        case .terminated, .terminating, .monitoringCompleted:
            // We have terminated in the meantime, nothing we can do
            ()
        case .awaitingAppConnection, .connected:
            let error = CloudBoardAppWorkloadError.illegalStateAfterClientIsConnected("\(self.state)")
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "State machine in unexpected state after connecting to CloudApp",
                state: self.state,
                error: error
            ).log(to: self.log, level: .fault)
            throw error
        }
    }

    func teardown() async throws {
        if self.terminationRequested {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Job termination already requested, waiting for termination",
                state: self.state
            ).log(to: self.log, level: .default)
        } else {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received request to teardown job",
                state: self.state
            ).log(to: self.log, level: .default)
            self.terminationRequested = true
        }

        switch self.state {
        case .awaitingAppConnection, .connecting:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received request to teardown CloudApp while not yet connected, waiting for connection",
                state: self.state
            ).log(to: self.log, level: .default)
            try await self.waitForTermination()
        case .connected(let client):
            try await self.teardownConnectedClient(client: client)
            try await self.waitForTermination()
        case .terminating:
            try await self.waitForTermination()
        case .terminated, .monitoringCompleted:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Ignoring request to teardown CloudApp",
                state: self.state
            ).log(to: self.log, level: .default)
        }
    }

    func waitForTermination() async throws {
        do {
            _ = try await Future(self.terminationPromise).resultWithCancellation
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Unexpected error while waiting for terminationPromise to be fulfiled",
                state: self.state,
                error: error
            ).log(to: self.log, level: .fault)
            throw CloudBoardAppWorkloadError.unexpectedTerminationError(error)
        }
    }

    func teardownConnectedClient(
        client: CloudBoardJobAPIClientToServerProtocol,
        bufferedInputChunks: [(data: Data?, isFinal: Bool)] = []
    ) async throws {
        if bufferedInputChunks.count > 0 {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "CloudApp requested to terminate with buffered input chunks",
                state: self.state
            ).log(to: self.log, level: .error)
        }
        self.state = .terminating
        CloudBoardAppStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            message: "Sending teardown request to CloudApp",
            state: self.state
        ).log(to: self.log, level: .debug)
        do {
            try await client.teardown()
        } catch {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "client.teardown() returned error",
                state: self.state,
                error: error
            ).log(to: self.log, level: .error)
        }
    }

    func clientTerminated(statusCode: Int?) async {
        switch self.state {
        case .awaitingAppConnection(_, _, _),
             .connecting:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "CloudApp has terminated with buffered input data chunks",
                state: self.state
            ).log(to: self.log, level: .error)
            self.terminationPromise.succeed()
        case .connected, .terminating:
            self.terminationPromise.succeed()
        case .terminated:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "CloudApp reported to have terminated after monitoring stopped",
                state: self.state
            ).log(to: self.log, level: .error)
        case .monitoringCompleted:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "CloudApp reported to have terminated after monitoring stopped",
                state: self.state
            ).log(to: self.log, level: .default)
        }

        await self.delegate.cloudBoardJobAPIClientAppTerminated(statusCode: statusCode)

        self.state = .terminated
    }

    // This routine is guaranteed to be invoked once we make it to the point
    // of calling CloudBoardAppWorkload.run(). That ensures that the terminationPromise
    // is completed.
    func monitoringCompleted(error: Error? = nil) async throws {
        defer {
            self.state = .monitoringCompleted
        }

        // Signal that the workload has completed.
        do {
            try await self.delegate.endJob()
        } catch {
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Failed to invoke endJob",
                state: self.state,
                error: error
            ).log(to: self.log, level: .error)
        }

        switch self.state {
        case .awaitingAppConnection, .connecting, .terminating:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "CloudApp monitoring stopped before receiving termination notification",
                state: self.state,
                error: error
            ).log(to: self.log, level: .error)
            self.terminationPromise.fail(with: CloudBoardAppWorkloadError.monitoringCompletedEarly(error))
            throw CloudBoardAppWorkloadError.monitoringCompletedEarly(error)
        case .connected:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "cb_jobhelper monitoring stopped before receiving termination notification",
                state: self.state
            ).log(to: self.log, level: .error)
        case .terminated:
            // terminationPromise fulfilled in clientTerminated()
            ()
        case .monitoringCompleted:
            CloudBoardAppStateMachineCheckpoint(
                logMetadata: self.logMetadata(),
                message: "cb_jobhelper monitoring reported to have completed twice",
                state: self.state,
                error: CloudBoardAppWorkloadError.monitoringCompletedMoreThanOnce
            ).log(to: self.log, level: .error)
            throw CloudBoardAppWorkloadError.monitoringCompletedMoreThanOnce
        }
    }
}

extension WarmupDetails {
    init(_ data: WarmupData) {
        self = .init(
            initialMetrics: .init(
                setupMessageReceived: data.setupMessageReceived
            )
        )
    }
}

// Needed to safely de-queue chunks after async calls.
extension CloudBoardAppStateMachine.State {
    var bufferedInputChunks: [(data: Data?, isFinal: Bool)] {
        switch self {
        case .awaitingAppConnection(let chunks, _, _):
            return chunks
        case .connecting(let chunks, _, _):
            return chunks
        case .connected, .monitoringCompleted, .terminating, .terminated:
            return []
        }
    }

    var warmupData: WarmupData? {
        switch self {
        case .awaitingAppConnection(_, let data, _):
            return data
        case .connecting(_, let data, _):
            return data
        case .connected, .monitoringCompleted, .terminating, .terminated:
            return nil
        }
    }

    var parametersData: ParametersData? {
        switch self {
        case .awaitingAppConnection(_, _, let data):
            return data
        case .connecting(_, _, let data):
            return data
        case .connected, .monitoringCompleted, .terminating, .terminated:
            return nil
        }
    }

    mutating func nextBufferedChunk() -> (data: Data?, isFinal: Bool)? {
        switch self {
        case .connecting(var bufferedInputData, let warmupData, let payloadData):
            guard bufferedInputData.count > 0 else { return nil }
            let data = bufferedInputData.removeFirst()
            self = .connecting(bufferedInputData, warmupData, payloadData)
            return data
        case .awaitingAppConnection, .connected, .terminating, .terminated, .monitoringCompleted:
            return nil
        }
    }
}

extension CloudBoardJobAPI.ParametersData {
    init(_ data: CloudBoardJobHelperAPI.Parameters) {
        self.init(
            parametersReceived: data.parametersReceived,
            plaintextMetadata: .init(data.plaintextMetadata, requestID: data.requestID)
        )
    }
}

extension CloudBoardJobAPI.ParametersData.PlaintextMetadata {
    init(_ data: CloudBoardJobHelperAPI.Parameters.PlaintextMetadata, requestID: String) {
        self.init(
            bundleID: data.bundleID,
            bundleVersion: data.bundleVersion,
            featureID: data.featureID,
            clientInfo: data.clientInfo,
            workloadType: data.workloadType,
            workloadParameters: data.workloadParameters,
            requestID: requestID,
            automatedDeviceGroup: data.automatedDeviceGroup
        )
    }
}

extension CloudBoardAppStateMachine {
    private func logMetadata() -> CloudBoardJobHelperLogMetadata {
        return CloudBoardJobHelperLogMetadata(
            jobID: self.jobID,
            requestTrackingID: self.requestID,
            remotePID: self.remotePID
        )
    }
}

struct CloudBoardAppStateMachineCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    var operationName: StaticString

    var serviceName: StaticString = "cb_jobhelper"

    var namespace: StaticString = "cloudboard"

    var error: Error?

    var logMetadata: CloudBoardJobHelperLogMetadata

    var message: StaticString

    var state: CloudBoardAppStateMachine.State

    public init(
        logMetadata: CloudBoardJobHelperLogMetadata,
        operationName: StaticString = #function,
        message: StaticString,
        state: CloudBoardAppStateMachine.State,
        error: Error? = nil
    ) {
        self.logMetadata = logMetadata
        self.operationName = operationName
        self.message = message
        self.state = state
        if let error {
            self.error = error
        }
    }

    public func log(to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        state=\(self.state, privacy: .public)
        message=\(self.message, privacy: .public)
        """)
    }
}

struct CloudBoardAppWorkloadCheckpoint: RequestCheckpoint {
    private(set) var jobID: UUID

    private(set) var requestID: String?

    private(set) var remotePID: Int?

    var operationName: StaticString

    var serviceName: StaticString = "cb_jobhelper"

    var namespace: StaticString = "cloudboard"

    var error: Error?

    var message: StaticString

    var state: MonitoredLaunchdJobInstance.AsyncIterator.State?

    public init(
        jobID: UUID,
        requestID: String?,
        remotePID _: Int?,
        operationName: StaticString = #function,
        message: StaticString,
        state: MonitoredLaunchdJobInstance.AsyncIterator.State?,
        error: Error? = nil
    ) {
        self.jobID = jobID
        self.requestID = requestID
        self.operationName = operationName
        self.state = state
        self.message = message
        if let error {
            self.error = error
        }
    }

    public func log(to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.jobID.uuidString, privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        state=\(String(describing: self.state), privacy: .public)
        """)
    }
}
