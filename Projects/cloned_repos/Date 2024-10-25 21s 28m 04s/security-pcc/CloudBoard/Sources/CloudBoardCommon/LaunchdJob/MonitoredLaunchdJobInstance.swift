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

// Copyright © 2023 Apple Inc. All rights reserved.

import AppServerSupport.OSLaunchdJob
import CloudBoardMetrics
import Foundation
import os
import System

public struct MonitoredLaunchdJobInstance {
    public init(_ managedJob: ManagedLaunchdJob, metrics: (any MetricsSystem)? = nil) throws {
        self.job = try LaunchdJob(managedJob, metrics: metrics)
    }

    public init(_ managedJob: ManagedLaunchdJob, uuid: UUID, metrics: (any MetricsSystem)? = nil) throws {
        self.job = try LaunchdJob(managedJob, uuid: uuid, metrics: metrics)
    }

    internal var job: LaunchdJob
    public var uuid: UUID { self.job.uuid }
}

extension MonitoredLaunchdJobInstance: AsyncSequence {
    public nonisolated func makeAsyncIterator() -> AsyncIterator {
        return AsyncIterator(self.job)
    }

    public typealias Element = AsyncIterator.State

    public actor AsyncIterator: AsyncIteratorProtocol {
        internal var job: LaunchdJob
        private var state: State
        private var seenInitialNeverRanEvent: Bool = false
        private var sigkillTask: Task<Void, Error>?

        public typealias Element = State

        init(_ job: LaunchdJob) {
            self.job = job
            self.state = .initialized
        }

        public func next() async throws -> State? {
            let uuid = self.job.uuid
            guard !self.state.isFinal() else {
                Self.log.debug(
                    "Job \(uuid, privacy: .public) reached final state"
                )
                return nil
            }

            switch self.state {
            case .initialized:
                var newState: State
                Self.log.debug(
                    "Job \(uuid, privacy: .public) initialized, creating"
                )
                do {
                    try await self.job.create()
                    newState = .created
                } catch LaunchdJobError.submitFailed {
                    newState = .neverRan
                } catch LaunchdJobError.createFailed {
                    newState = .neverRan
                }
                await self.transition(to: newState)
                return self.state
            case .created:
                var newState: State
                Self.log.debug("Job \(uuid, privacy: .public) created, running")
                do {
                    try await self.job.run()
                    newState = .starting
                } catch LaunchdJobError.spawnFailed(let errNo) {
                    newState = .terminated(.spawnFailed(errNo))
                } catch LaunchdJobError.exited(let status) {
                    newState = .terminated(.exited(ExitStatus(from: status)))
                }
                await self.transition(to: newState)
                return self.state
            default:
                for await (jobInfo, errno) in self.job {
                    await self.handleLaunchdJobEvent(jobInfo: jobInfo, errno: errno)
                    return self.state
                }
            }
            return nil
        }
    }
}

extension MonitoredLaunchdJobInstance.AsyncIterator {
    public static let log: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "MonitoredLaunchdJob"
    )

    private func handleLaunchdJobEvent(jobInfo: OSLaunchdJobInfo?, errno: Int32) async {
        let errno = Errno(rawValue: errno)
        let uuid = self.job.uuid
        guard let jobInfo else {
            // This is unexpected. Assume the job no longer exists. Once an
            // error is returned, we won't receive any more updates.
            Self.log.error("""
            \(uuid, privacy: .public): Received launchd job event without \
            context: \(errno, privacy: .public)
            """)
            await self.transition(to: .terminated(.launchdError(errno)))
            return
        }

        Self.log.info("""
        \(uuid, privacy: .public): Received launchd job event with state: \
        \(jobInfo.state, privacy: .public)
        """)

        switch jobInfo.state {
        case OSLaunchdJobState.running:
            switch self.state {
            case .starting:
                let pid = await (self.job.instanceHandle?.getCurrentJobInfo()?.pid).map { Int($0) }
                await self.transition(to: .running(pid: pid))
            default:
                fatalError("""
                \(uuid): Unexpectedly received notification that launchd job is now \
                running with stashed/previous MonitoredLaunchdJobInstance state: \(self.state))
                """)
            }
        case OSLaunchdJobState.exited:
            let exitStatus = ExitStatus(from: jobInfo.lastExitStatus)
            Self.log.log("""
            \(uuid, privacy: .public): Job exited with status \
            \(exitStatus, privacy: .public)
            """)
            // Cancel monitoring for exited job. Required to avoid mach port leak.
            await self.transition(to: .terminated(.exited(exitStatus)))
        case OSLaunchdJobState.spawnFailed:
            let errNo = Errno(rawValue: jobInfo.lastSpawnError)
            await self.transition(to: .terminated(.spawnFailed(errNo)))
        case OSLaunchdJobState.neverRan:
            // if we've never seen
            if !self.seenInitialNeverRanEvent {
                Self.log.debug(
                    "\(uuid, privacy: .public): Ignoring initial NeverRanEvent"
                )
                self.seenInitialNeverRanEvent = true
                break
            }
            await self.transition(to: .neverRan)
        }
    }

    /// Terminates the job by sending SIGTERM to the process. If the job does
    /// not terminate after the grace period then a SIGKILL is sent to the
    /// process.
    ///
    /// - Parameters:
    ///   - force: Send a SIGKILL to the process immediately instead of first
    ///   sending a SIGTERM.
    ///   - gracePeriod: Time to wait before sending SIGKILL after SIGTERM.
    ///
    public func terminate(force: Bool = false, gracePeriod: Duration = .seconds(10)) async {
        let instanceToTerminate: OSLaunchdJob?
        switch self.state {
        case .initialized, .created:
            Self.log.info("Launchd job instance asked to terminate but job is not running")
            return
        case .starting, .running:
            instanceToTerminate = await self.job.instanceHandle
        case .terminating:
            if !force {
                Self.log.info("Launchd job instance asked to terminate but job is already terminating")
                // Nothing to do as we are already terminating
                return
            }
            instanceToTerminate = await self.job.instanceHandle
        case .terminated:
            Self.log.info("Launchd job instance asked to terminate but job has already terminated")
            return
        case .neverRan:
            Self.log.info("Launchd job instance asked to terminate but job never ran")
            return
        }
        await self.transition(to: .terminating)

        Self.log.info("Terminating launchd job instance")

        guard let jobInfo = instanceToTerminate?.getCurrentJobInfo() else {
            Self.log.error("Unable to retrieve OSLaunchdJobInfo for termination. Treating this as unclean shutdown.")
            await self.transition(to: .terminated(.uncleanShutdown))
            return
        }

        guard jobInfo.state == OSLaunchdJobState.running else {
            Self.log.debug("Launchd job instance asked to terminate but job is unexpectedly not running")
            await self.transition(to: .terminated(.uncleanShutdown))
            return
        }

        LaunchdJobHelper.sendTerminationSignal(
            pid: jobInfo.pid, signal: force ? .sigkill : .sigterm, logger: Self.log
        )

        // Schedule task to send SIGKILL in case the process doesn't terminate on its own
        if !force {
            let log = Self.log
            self.sigkillTask = Task {
                try await withTaskCancellationHandler {
                    log.debug("""
                    Scheduling SIGKILL to be sent: grace_period=\
                    \(gracePeriod, privacy: .public)
                    """)
                    try await Task.sleep(for: gracePeriod)
                    await self.terminate(force: true)
                } onCancel: {
                    log.debug("Scheduled SIGKILL task has been canceled")
                }
            }
        }
    }

    private func transition(to state: State) async {
        let uuid = self.job.uuid
        let initialState = self.state
        Self.log.info("""
        Job \(uuid, privacy: .public) transitioning from \
        '\(initialState, privacy: .public)' to \
        '\(state, privacy: .public)'
        """)
        if state.isFinal() {
            // Cancel monitoring for failed/terminated jobs. As per the
            // OSLaunchdJobMonitorHandler documentation, the monitor will not
            // be called again in case of an error. Required to avoid mach
            // port leak.
            switch self.state {
            case .starting, .running, .terminating:
                await self.job.stopMonitoring()
                self.sigkillTask?.cancel()
            default:
                // Nothing to do
                ()
            }
        }
        self.state = state
    }
}

extension MonitoredLaunchdJobInstance.AsyncIterator {
    public enum TerminationCondition {
        // Instance exited with the provided exit status
        case exited(ExitStatus)
        // Failed to terminate cleanly and unable to determine exit status of the instance
        case uncleanShutdown
        // An unexpected unrecoverable launchd error occured
        case launchdError(Errno)
        // Failed to spawn the instance due to the provided error
        case spawnFailed(Error)
    }

    public enum ExitStatus: CustomStringConvertible, Sendable {
        /// OS status with reason namespace and reason code
        case osStatus(Int, Int)
        /// Wait4 status with exit code
        case wait4Status(Int?)
        case unknown

        public var description: String {
            switch self {
            case .unknown:
                return "unknown"
            case .osStatus(let namespace, let code):
                let reasonNamespace = TerminationCondition.OSReasonNamespace(rawValue: namespace)
                let namespaceDescription = reasonNamespace?.description ?? "\(namespace)"
                let reasonDescription = reasonNamespace?.codeDescription(for: code) ?? "\(code)"
                return "OS reason namespace: \(namespaceDescription), reason: \(reasonDescription), code: \(code)"
            case .wait4Status(let code):
                return code != nil ? "\(code!)" : "unknown wait4 status"
            }
        }

        init(from exitStatus: OSLaunchdJobExitStatus?) {
            guard let exitStatus else {
                self = .unknown
                return
            }
            if exitStatus.os_reason_namespace > 0 {
                self = .osStatus(Int(exitStatus.os_reason_namespace), Int(exitStatus.os_reason_code))
            } else if ExitStatus.WIFEXITED(exitStatus.wait4Status) {
                self = .wait4Status(Int(ExitStatus.WEXITSTATUS(exitStatus.wait4Status)))
            } else {
                self = .wait4Status(nil)
            }
        }

        /*
         Routines for interpreting status values from wait4()
         */
        private static func _WSTATUS(_ status: CInt) -> CInt {
            status & 0x7F
        }

        private static func WIFEXITED(_ status: CInt) -> Bool {
            self._WSTATUS(status) == 0
        }

        private static func WEXITSTATUS(_ status: CInt) -> CInt {
            (status >> 8) & 0xFF
        }
    }

    /// Externally visible state.
    public enum State: CustomStringConvertible {
        /// Initial state.
        case initialized
        /// The launchd job instance has been created but not started yet.
        case created
        /// The launchd job instance has been spawned.
        case starting
        /// The launchd job instance is running.
        case running(pid: Int?)
        /// The launchd job instance has been asked to terminate (by sending a SIGTERM).
        ///
        /// Note: There is no guarantee that this state will be entered during the instance's lifetime. This is only
        /// entered when the instance is terminated via ``MonitredLaunchdJobInstance.terminate``.
        case terminating
        /// The job instance has terminated with the provided exit status.
        case terminated(TerminationCondition)
        case neverRan

        func isFinal() -> Bool {
            switch self {
            case .terminated:
                return true
            default:
                return false
            }
        }

        public var description: String {
            switch self {
            case .initialized:
                return "Initialized"
            case .created:
                return "Created"
            case .starting:
                return "Starting"
            case .running:
                return "Running"
            case .terminating:
                return "Terminating"
            case .terminated(let terminationCondition):
                return "Terminated (\(terminationCondition))"
            case .neverRan:
                return "NeverRan"
            }
        }
    }
}
