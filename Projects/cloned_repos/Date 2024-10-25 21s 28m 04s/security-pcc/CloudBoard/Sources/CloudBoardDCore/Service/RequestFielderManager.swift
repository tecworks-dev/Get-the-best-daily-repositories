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

import AppServerSupport.OSLaunchdJob
import CloudBoardCommon
import CloudBoardJobHelperAPI
import CloudBoardMetrics
import DequeModule
import os
import Tracing

private let log: Logger = .init(
    subsystem: "com.apple.cloudos.cloudboard",
    category: "RequestFielderManager"
)

// This placeholder delegate is needed for prewarmed RequestFielder instances until
// they are used and we get a real delegate for handling responses from associated
// RequestFielder instances. Instances of this are never expected to be invoked.
actor PrewarmedJobHelperResponseDelegate: CloudBoardJobHelperAPIClientDelegateProtocol {
    var metrics: MetricsSystem

    init(metrics: MetricsSystem) {
        self.metrics = metrics
    }

    nonisolated func cloudBoardJobHelperAPIClientSurpriseDisconnect() {
        log.error("surprise disconnect of prewarmed RequestFielder")
    }

    func sendWorkloadResponse(_: CloudBoardJobHelperAPI.WorkloadResponse) {
        self.metrics.emit(Metrics.RequestFielderManager.PlaceholderDelegateWorkloadResponseInvoked(action: .increment))
        log.error("sendWorkloadResponse unexpectedly called on PrewarmedJobHelperResponseDelegate")
    }
}

struct ManagedRequestFielder: Equatable {
    static func == (lhs: ManagedRequestFielder, rhs: ManagedRequestFielder) -> Bool {
        return lhs.instance == rhs.instance
    }

    func logMetadata() -> CloudBoardDaemonLogMetadata {
        return CloudBoardDaemonLogMetadata(jobID: self.id)
    }

    let instance: RequestFielder
    let teardownPromise: Promise<Void, Error>
    let id: UUID
}

enum ManagedRequestFielderQueueError: Error {
    case usedAfterTerminateCalled
    case terminateCalledMultipleTimes
}

actor ManagedRequestFielderQueue {
    private var readyFielders: Deque<ManagedRequestFielder> = Deque()
    private var waiters: [CheckedContinuation<ManagedRequestFielder, Error>] = []
    private var terminated: Bool = false
    private var prewarmedPoolSize: Int
    private let metrics: MetricsSystem

    init(prewarmedPoolSize: Int, metrics: MetricsSystem) {
        self.prewarmedPoolSize = prewarmedPoolSize
        self.metrics = metrics
    }

    private func enqueue(_ fielder: ManagedRequestFielder, _ operation: (ManagedRequestFielder) -> Void) throws {
        guard !self.terminated else {
            throw ManagedRequestFielderQueueError.usedAfterTerminateCalled
        }
        if self.waiters.count > 0 {
            let waiter = self.waiters.removeFirst()
            self.metrics.emit(Metrics.RequestFielderManager.WaitedForCreationTotal(
                action: .increment,
                prewarmedPoolSize: self.prewarmedPoolSize
            ))
            waiter.resume(returning: fielder)
        } else {
            operation(fielder)
        }
    }

    func push(_ fielder: ManagedRequestFielder) throws {
        try self.enqueue(fielder) { self.readyFielders.append($0) }
    }

    func pushFirst(_ fielder: ManagedRequestFielder) throws {
        try self.enqueue(fielder) { self.readyFielders.insert($0, at: 0) }
    }

    func popFirst() async throws -> ManagedRequestFielder {
        guard !self.terminated else {
            throw ManagedRequestFielderQueueError.usedAfterTerminateCalled
        }

        if let element = self.readyFielders.popFirst() {
            return element
        }

        return try await withCheckedThrowingContinuation {
            self.waiters.append($0)
        }
    }

    func remove(_ fielder: ManagedRequestFielder) -> Bool {
        guard let index = self.readyFielders.firstIndex(where: { $0 == fielder }) else {
            return false
        }

        self.readyFielders.remove(at: index)
        return true
    }

    func terminate(error: Error) throws -> Deque<ManagedRequestFielder> {
        guard !self.terminated else {
            throw ManagedRequestFielderQueueError.terminateCalledMultipleTimes
        }
        self.terminated = true
        let waiters = self.waiters
        self.waiters = []
        for waiter in waiters {
            waiter.resume(throwing: error)
        }

        defer {
            self.readyFielders = []
        }
        return self.readyFielders
    }
}

enum RequestFielderManagerError: Error {
    case teardownInitiatedBecauseRefillStreamFinished
    case invalidPrewarmedPoolSizeSpecified
    case invalidMaxProcessCountSpecified
    case failedToAllocateRequestFielder
}

// RequestFielderManager is left as a class so that we don't serialize all
// calls to withClient() (support multiple jobs in parallel)
final class RequestFielderManager: Sendable {
    private let metrics: MetricsSystem
    private let tracer: any Tracer

    private let requestFielderGetRetryCount: Int = 3
    private let prewarmedPoolSize: Int
    private let maxProcessCount: Int

    private var prewarmingDisabled: Bool {
        return self.prewarmedPoolSize == 0
    }

    private var processLimitDisabled: Bool {
        return self.maxProcessCount == 0
    }

    private let prewarmedQueue: ManagedRequestFielderQueue
    private let refillStream: AsyncStream<Void>
    private let refillContinuation: AsyncStream<Void>.Continuation

    private struct State {
        var prewarmedCount: Int = 0
        var totalRunningCount: Int = 0
        var pendingCreationCount: Int = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    private func incrementPrewarmedPoolCount() {
        self.state.withLock { state in
            state.prewarmedCount += 1
            self.metrics.emit(Metrics.RequestFielderManager.PrewarmedPoolSizeGauge(
                value: state.prewarmedCount, configuredPoolSize: self.prewarmedPoolSize
            ))
        }
    }

    private func decrementPrewarmedPoolCount() {
        self.state.withLock { state in
            state.prewarmedCount -= 1
            self.metrics.emit(Metrics.RequestFielderManager.PrewarmedPoolSizeGauge(
                value: state.prewarmedCount, configuredPoolSize: self.prewarmedPoolSize
            ))
        }
    }

    init(prewarmedPoolSize: Int, maxProcessCount: Int, metrics: MetricsSystem, tracer: any Tracer) throws {
        log.log(
            "Initializing RequestFielderManager with prewarmedPoolSize: \(prewarmedPoolSize, privacy: .public), maxProcessCount: \(maxProcessCount, privacy: .public)"
        )
        if prewarmedPoolSize < 0, prewarmedPoolSize > 100 {
            log.error(
                "Invalid prewarmedPoolSize specified (\(prewarmedPoolSize, privacy: .public)), value must be in [0, 100]"
            )
            throw RequestFielderManagerError.invalidPrewarmedPoolSizeSpecified
        }
        if maxProcessCount != 0, prewarmedPoolSize > maxProcessCount {
            log.error(
                "Specified prewarmedPoolSize (\(prewarmedPoolSize, privacy: .public)), exceeds provided maxProcessCount (\(maxProcessCount, privacy: .public))"
            )
            throw RequestFielderManagerError.invalidPrewarmedPoolSizeSpecified
        }
        self.prewarmedPoolSize = prewarmedPoolSize

        if maxProcessCount < 0, maxProcessCount > 200 {
            log.error(
                "Invalid maxProcessCount specified (\(maxProcessCount, privacy: .public)), value must be in [0, 200]"
            )
            throw RequestFielderManagerError.invalidMaxProcessCountSpecified
        }
        self.maxProcessCount = maxProcessCount

        self.prewarmedQueue = ManagedRequestFielderQueue(prewarmedPoolSize: self.prewarmedPoolSize, metrics: metrics)
        (self.refillStream, self.refillContinuation) = AsyncStream<Void>.makeStream()
        self.metrics = metrics
        self.tracer = tracer
    }

    // This is non-throwing because we don't want errors managing/running a specific fielder to
    // be propagated outside of the management context of that specific instance.
    private func manageRequestFielder(_ managedRequestFielder: ManagedRequestFielder) async {
        await withTaskGroup(of: Void.self) { instanceTaskGroup in
            instanceTaskGroup.addTask {
                do {
                    try await managedRequestFielder.instance.run()
                } catch {
                    log.error("""
                    \(managedRequestFielder.logMetadata(), privacy: .public) \
                    manageRequestFielder instance.run() received an error: \
                    \(String(unredacted: error), privacy: .public)
                    """)
                }

                // Check to see if the request fielder is still on the prewarmed queue,
                // if it is, remove it and complete the promise as we are responsible for it.
                // If it's not, the current holder of the request fielder is responsible
                // for completing the promise.
                if await self.prewarmedQueue.remove(managedRequestFielder) {
                    self.metrics.emit(Metrics.RequestFielderManager.PrewarmedInstancesDiedTotal(action: .increment))
                    self.decrementPrewarmedPoolCount()
                    managedRequestFielder.teardownPromise.succeed()
                }
            }

            instanceTaskGroup.addTask {
                await withTaskCancellationHandler {
                    // This promise should always be fulfilled, even if things are cancelled. If
                    // we don't wait for this and instead return on cancellation, we leak a RequestFielder.
                    do {
                        try await Future(managedRequestFielder.teardownPromise).value
                    } catch {
                        log.error("""
                        \(managedRequestFielder.logMetadata(), privacy: .public) \
                        manageRequestFielder received error from associated \
                        future: \(String(unredacted: error), privacy: .public)
                        """)
                    }

                    do {
                        try await managedRequestFielder.instance.teardown()
                    } catch {
                        log.error("""
                        \(managedRequestFielder.logMetadata(), privacy: .public) \
                        manageRequestFielder encountered error invoking \
                        teardown: \
                        \(String(unredacted: error), privacy: .public)
                        """)
                    }
                } onCancel: {
                    log.warning(
                        "TaskCancellationHandler invoked on task responsible for waiting on teardownPromise"
                    )
                }
            }

            instanceTaskGroup.addTask {
                do {
                    try await managedRequestFielder.instance
                        .invokeWorkloadRequest(.warmup(.init(setupMessageReceived: .now)))
                    await managedRequestFielder.instance.markWarmupComplete()
                } catch {
                    // Make sure mark warmup as complete and initiate teardown on the RequestFielder.
                    // Dont rethrow this error because we want to let run complete so the RequestFielder
                    // can be torn down and removed from the prewarmed queue.
                    log.error("""
                    \(managedRequestFielder.logMetadata(), privacy: .public) \
                    warmup failed with error: \
                    \(String(unredacted: error), privacy: .public) \
                    tearing down
                    """)
                    self.metrics.emit(Metrics.RequestFielderManager.InvokeWarmupErrorTotal(action: .increment))
                    await managedRequestFielder.instance.markWarmupComplete(error: error)
                }
            }

            await instanceTaskGroup.waitForAll()
        }

        self.requestFielderConsumed()
    }

    func run() async throws {
        let managedJobs = LaunchdJobHelper.fetchManagedLaunchdJobs(
            type: CloudBoardJobType.cbJobHelper,
            state: OSLaunchdJobState.neverRan,
            skippingInstances: true,
            logger: CloudBoardJobHelperXPCClientProvider.log
        )
        guard managedJobs.count == 1 else {
            log.error("unexpected cb_jobhelper managed jobs count \(managedJobs.count, privacy: .public)")
            throw RequestFielderError.tooManyManagedJobs
        }
        log.log("discovered cb_jobhelper launchd job")

        self.state.withLock { state in
            for _ in 0 ..< self.prewarmedPoolSize {
                self.refillContinuation.yield()
                state.pendingCreationCount += 1
            }
        }

        // We don't want the child tasks to throw here since each is responsible for handling
        // the errors in creating and managing its associated fielder instance.
        await withDiscardingTaskGroup { group in
            // Each time we receive a signal on this stream, create one new RequestFielder
            for await _ in self.refillStream {
                do {
                    // Need to make sure that we call markWarmupComplete() on this RequestFielder
                    // in all paths after creation to ensure fulfillment of the underlying promise.
                    let newRequestFielder = try RequestFielder(
                        managedJobs[0],
                        delegate: PrewarmedJobHelperResponseDelegate(metrics: self.metrics),
                        metrics: self.metrics,
                        tracer: self.tracer
                    )
                    let id = await newRequestFielder.id
                    let managedRequestFielder = ManagedRequestFielder(
                        instance: newRequestFielder,
                        teardownPromise: Promise(),
                        id: id
                    )
                    log.info("\(managedRequestFielder.logMetadata(), privacy: .public) created RequestFielder")
                    do {
                        try await self.prewarmedQueue.push(managedRequestFielder)
                    } catch {
                        log.error("""
                        \(managedRequestFielder.logMetadata(), privacy: .public) failed to put \
                        new RequestFielder on queue: \
                        \(String(unredacted: error), privacy: .public) \
                        , tearing down
                        """)
                        // Need to cleanup the promises we've created and explicitly teardown the RequestFielder
                        // ourselves because we have yet to call manageRequestFielder() (and won't for this instance).
                        managedRequestFielder.teardownPromise.fail(with: error)
                        await managedRequestFielder.instance.markWarmupComplete(error: error)
                        do {
                            try await managedRequestFielder.instance.teardown()
                        } catch {
                            log.error("""
                            \(managedRequestFielder.logMetadata(), privacy: .public) failed to \
                            invoke teardown() on RequestFielder with error \
                            \(String(unredacted: error), privacy: .public) \
                            , potentially leaked process
                            """)
                        }
                        throw error
                    }
                    group.addTask {
                        await self.manageRequestFielder(managedRequestFielder)
                        self.state.withLock { state in
                            state.totalRunningCount -= 1
                        }
                        self.requestFielderConsumed()
                    }

                    self.incrementPrewarmedPoolCount()
                    self.state.withLock { state in
                        state.totalRunningCount += 1
                        state.pendingCreationCount -= 1
                    }
                } catch {
                    log.error("""
                    failed to create RequestFielder instance: \
                    \(String(unredacted: error), privacy: .public)
                    """)
                    self.metrics.emit(Metrics.RequestFielderManager.FailedToCreateTotal(action: .increment))
                    // Enqueue a new work item to create a prewarmed fielder
                    self.refillContinuation.yield()
                }
            }

            log.error(
                "RequestFielderManager run() refillStream finished, tearing down any remaining prewarmed instances"
            )
            do {
                let error = RequestFielderManagerError.teardownInitiatedBecauseRefillStreamFinished
                for managedRequestFielder in try await self.prewarmedQueue.terminate(error: error) {
                    managedRequestFielder.teardownPromise.fail(with: error)
                    self.decrementPrewarmedPoolCount()
                }
            } catch {
                log.error("""
                RequestFielderManager run() teardown failed to invoke \
                terminate() on prewarmed queue: \
                \(String(unredacted: error), privacy: .public)
                """)
            }
        }
    }

    // Called whenever a RequestFielder goes away for any reason.
    // Does the logical check and determines
    // "do we need a new one - yes/no?", if yes enqueue a void message into the
    // stream that the run refill tasks awaits on.
    // Avoid throwing and async routines to maintain simplicity here
    // called when one is consumed from the free queue AND when one actually
    // dies.
    private func requestFielderConsumed() {
        // While the # of (currently running + creation in-flight processes) is less than the ceiling
        // AND the # of (currently prewarmed + creation in-flight processes) is less than the prewarmed pool size,
        // submit new creation requests.
        self.state.withLock { state in
            var pendingPlusRunningProcessCount: Int {
                state.totalRunningCount + state.pendingCreationCount
            }

            var pendingPlusPrewarmedCount: Int {
                state.prewarmedCount + state.pendingCreationCount
            }

            while self.processLimitDisabled ||
                pendingPlusRunningProcessCount < self.maxProcessCount,
                pendingPlusPrewarmedCount < self.prewarmedPoolSize {
                state.pendingCreationCount += 1
                self.refillContinuation.yield()
            }
        }
    }

    private func dequeueRequestFielder(
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol
    ) async throws -> ManagedRequestFielder {
        let waitForAllocationMeasurement = ContinuousTimeMeasurement.start()

        let retryMaxCount = self.requestFielderGetRetryCount
        for attempt in 1 ... retryMaxCount {
            // If prewarming is disabled, signal to request a new RequestFielder
            if self.prewarmingDisabled {
                self.state.withLock { state in
                    state.pendingCreationCount += 1
                }
                self.refillContinuation.yield()
            }

            let requestFielder: ManagedRequestFielder
            do {
                requestFielder = try await self.prewarmedQueue.popFirst()
            } catch {
                log.error("""
                dequeueRequestFielder popFirst returned error: \
                \(String(unredacted: error), privacy: .public)
                """)
                throw error
            }
            self.decrementPrewarmedPoolCount()
            self.requestFielderConsumed()

            do {
                // This could fail if the RequestFielder died and we de-queued it
                // before manageRequestFielder() handled the death notification and
                // removed it from the prewarmed queue.
                try await requestFielder.instance.set(delegate: delegate)
                self.metrics.emit(
                    Metrics.RequestFielderManager.AttemptsToAllocateRequestFielderHistogram(value: attempt)
                )
                self.metrics.emit(
                    Metrics.RequestFielderManager.TimeToAllocateRequestFielderHistogram(
                        duration: waitForAllocationMeasurement.duration,
                        prewarmingEnabled: !self.prewarmingDisabled,
                        prewarmedPoolSize: self.prewarmedPoolSize
                    )
                )
                return requestFielder
            } catch {
                log.error("""
                \(requestFielder.logMetadata(), privacy: .public) dequeueRequestFielder failed \
                to use dequeued fielder: \
                \(String(unredacted: error), privacy: .public)
                """)
                // Ensure the request fielder is torn down
                requestFielder.teardownPromise.succeed()
            }
        }

        self.metrics.emit(Metrics.RequestFielderManager.FailedToAllocateTotal(
            action: .increment,
            retryCount: retryMaxCount
        ))
        throw RequestFielderManagerError.failedToAllocateRequestFielder
    }

    // Assumption here is that we have ownership of the ManagedRequestFielder at this point,
    // if someone else does use it, they have violated the ownership contract defined
    // by use of the withRequestFielder() call (which expects all usage of the client to be contained
    // within the body). There's no great way to defend against violation of this contract.
    private func returnRequestFielder(_ requestFielder: ManagedRequestFielder) async {
        let instanceUsed = await requestFielder.instance.used
        // If the instance was used or prewarming is disabled, return the request fielder
        if instanceUsed || self.prewarmingDisabled {
            requestFielder.teardownPromise.succeed()
            return
        }

        do {
            try await requestFielder.instance.set(delegate: PrewarmedJobHelperResponseDelegate(metrics: self.metrics))
        } catch {
            log.error("""
            \(requestFielder.logMetadata(), privacy: .public) failed to reclaim \
            unused RequestFielder instance: \
            \(String(unredacted: error), privacy: .public)
            """)
            self.metrics.emit(Metrics.RequestFielderManager.FailedToReclaimTotal(action: .increment))
            // We complete the teardown promise, which signals to the associated task in manageRequestFielder()
            // that we should initiate teardown of this instance.
            requestFielder.teardownPromise.succeed()
            return
        }

        log.info("\(requestFielder.logMetadata(), privacy: .public) reclaimed unused RequestFielder instance")
        do {
            try await self.prewarmedQueue.pushFirst(requestFielder)
        } catch {
            log.error("""
            \(requestFielder.logMetadata(), privacy: .public) failed to return unused \
            RequestFielder instance to prewarmed queue: \
            \(String(unredacted: error), privacy: .public), \
            initiating teardown
            """)
            self.metrics.emit(Metrics.RequestFielderManager.FailedToReclaimTotal(action: .increment))
            // We complete the teardown promise, which signals to the associated task in manageRequestFielder()
            // that we should initiate teardown of this instance.
            requestFielder.teardownPromise.succeed()
            return
        }
        self.incrementPrewarmedPoolCount()
        self.metrics.emit(Metrics.RequestFielderManager.ReclaimedPrewarmedTotal(action: .increment))
    }

    // All usage of the RequestFielder provided to the body must be completed
    // by the time the body returns.
    func withRequestFielder<ReturnValue>(
        delegate: CloudBoardJobHelperAPIClientDelegateProtocol,
        _ body: (RequestFielder) async throws -> ReturnValue
    ) async throws
    -> ReturnValue {
        let requestFielder: ManagedRequestFielder
        do {
            try await requestFielder = self.dequeueRequestFielder(delegate: delegate)
        } catch {
            log.error("""
            withRequestFielder(): failed to obtain RequestFielder: \
            \(String(unredacted: error), privacy: .public)
            """)
            throw error
        }

        log.info("\(requestFielder.logMetadata(), privacy: .public) allocated for usage")

        do {
            let returnValue = try await body(requestFielder.instance)
            await self.returnRequestFielder(requestFielder)
            log.info("withRequestFielder returning")
            return returnValue
        } catch {
            log.error("""
            \(requestFielder.logMetadata(), privacy: .public) \
            withRequestFielder() body returned error \
            \(String(unredacted: error), privacy: .public)
            """)
            await self.returnRequestFielder(requestFielder)
            throw error
        }
    }
}
