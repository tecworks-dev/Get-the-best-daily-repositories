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

import CloudBoardConfigurationDAPI
import CloudBoardLogging
import CloudBoardMetrics
import ConfigurationServiceClient
import Foundation
import os

/// A delegate of ``Fetcher``, usually an object that can apply configuration and fallback.
protocol FetcherDelegate: AnyObject, Sendable {
    /// Applies the given configuration.
    func applyConfiguration(_ configuration: NodeConfigurationPackage) async throws

    /// Applies the fallback configuration.

    /// Applies the fallback configuration.
    func applyFallback() async throws
}

/// An actor that runs a configuration fetch and apply loop.
///
/// The fetcher is responsible for fetching a new configuration from the
/// upstream service and applying it to the delegate.
///
/// The fetcher will attempt to fetch a new configuration from the upstream
/// service at a regular interval.
///
/// If the first fetch fails, the fetcher instructs the delegate to apply
/// the fallback configuration, and keeps trying to fetch a new configuration
/// from the upstream service.
///
/// If the fetcher reconnects to the upstream service successfully, it then
/// publishes a new configuration to the delegate.
actor Fetcher {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "Fetcher"
    )

    /// A set of configuration values for the fetcher tick scheduling logic.
    struct SchedulingConfiguration {
        /// The rough interval between ticks, in other words the time between subsequent
        /// fetches to the upstream service.
        ///
        /// The actual interval is somewhat randomized to avoid a thundering herd of
        /// fetchers all trying to fetch at the same time.
        let tickInterval: Duration

        /// The maximum tolerance between ticks, expressed as a ratio of `tickInterval`.
        ///
        /// Valid range is between 0.0 and 1.0, both inclusive.
        ///
        /// The fetcher randomly adds a random amount of the tolerance to the tick interval
        /// to spread out fetches.
        ///
        /// Goes in both directions, so the added interval is `(-1*tolerance, tolerance)`.
        let maximumToleranceRatio: Double

        /// The minimum interval between one tick finishing and the next one starting, expressed as a ratio
        /// of `tickInterval`.
        ///
        /// Valid range is between 0.0 and 1.0, both inclusive.
        ///
        /// This is different to `tickInterval`, which controls the time between subsequent
        /// ticks starting. However `minimumInterTickDelayRatio` allows ensuring that the fetcher
        /// doesn't start the next tick right after it finished the previous one, in case the
        /// tick took a long time to run.
        let minimumInterTickDelayRatio: Double

        init(tickInterval: Duration, maximumToleranceRatio: Double, minimumInterTickDelayRatio: Double) {
            self.tickInterval = tickInterval
            precondition(
                maximumToleranceRatio >= 0.0 && maximumToleranceRatio <= 1.0,
                "Invalid maximumTolerance value: \(maximumToleranceRatio)"
            )
            self.maximumToleranceRatio = maximumToleranceRatio
            precondition(
                minimumInterTickDelayRatio >= 0.0 && minimumInterTickDelayRatio <= 1.0,
                "Invalid maximumTolerance value: \(minimumInterTickDelayRatio)"
            )
            self.minimumInterTickDelayRatio = minimumInterTickDelayRatio
        }
    }

    /// The schedule used for computing the time to make the next fetch.
    private let schedule: SchedulingConfiguration

    /// The data source for fetching the configuration from the upstream service.
    private let dataSource: FetcherDataSource

    /// The metrics system to use.
    private let metrics: MetricsSystem

    /// The state machine that drives the fetcher.
    private(set) var stateMachine: FetcherStateMachine {
        willSet {
            let oldState = self.stateMachine.state
            let newState = newValue.state
            guard oldState != newState else {
                return
            }
            Self.logger.debug(
                "State machine transitioning from: \(oldState, privacy: .public) to: \(newState, privacy: .public)."
            )
        }
    }

    /// The delegate that will be instructed to perform actions on the fetcher's behalf.
    private weak var delegate: FetcherDelegate?

    /// Creates a new fetcher.
    /// - Parameters:
    ///   - schedule: The schedule for calls to the upstream service.
    ///   - dataSource: The data source for fetching from the upstream service.
    ///   - metrics: The metrics system to use.
    init(
        schedule: SchedulingConfiguration,
        dataSource: FetcherDataSource,
        metrics: MetricsSystem
    ) {
        self.schedule = schedule
        self.dataSource = dataSource
        self.metrics = metrics
        self.stateMachine = .init()
        Self.logger.log("Configured with schedule: \(schedule, privacy: .public)")
        Self.logger.log("Configured with an upstream service of type: \(type(of: dataSource), privacy: .public)")
    }

    /// Sets the delegate that will be instructed to perform actions on the fetcher's behalf.
    ///
    /// Note that for consistent results, you should set the delegate before calling `run()`.
    /// - Parameter delegate: The delegate to call, or nil to remove.
    func set(delegate: FetcherDelegate?) {
        self.delegate = delegate
    }

    /// A result type for the result of a fetch.
    typealias FetcherResult<Value> = Result<Value, FetcherStateMachine.StringError>

    /// Runs the fetcher.
    func run() async throws {
        Self.logger.notice("Fetcher is starting.")
        try await withLogging(operation: "run", sensitiveError: false, logger: Self.logger) {
            self.metrics.emit(Metrics.Fetcher.IsWaitingForFirstGauge(value: 1))
            var waitingForFirstConfigOrFallbackStartInstant: ContinuousClock.Instant? = .now
            var lastTickInstant = ContinuousClock.now
            var action: FetcherStateMachine.Action = self.stateMachine.tick()
            self.metrics.emit(Metrics.Fetcher.TickCounter(action: .increment))
            while true {
                switch action {
                case .fetchConfig(currentRevisionIdentifier: let currentRevisionIdentifier):
                    let result = try await fetchConfig(currentRevisionIdentifier: currentRevisionIdentifier)
                    action = self.stateMachine.fetchedConfig(result)
                case .applyConfig(let configurationPackage):
                    let result = try await applyConfiguration(configurationPackage)
                    action = self.stateMachine.attemptedToApplyConfig(result)
                case .scheduleTick:
                    self.metrics.emit(Metrics.Fetcher.TickDurationHistogram(durationSinceStart: lastTickInstant))
                    let nextTickInstant = self.schedule.nextTickInstant(lastTickStartInstant: lastTickInstant)
                    try await self.sleepUntil(nextTickInstant)
                    lastTickInstant = nextTickInstant
                    self.metrics.emit(Metrics.Fetcher.TickCounter(action: .increment))
                    action = self.stateMachine.tick()
                case .reportFirstConfigAndScheduleTick(let configurationPackage):
                    try await self.reportFirstConfig(configurationPackage)
                    if let start = waitingForFirstConfigOrFallbackStartInstant {
                        self.metrics.emit(Metrics.Fetcher.FirstDelayHistogram(durationSinceStart: start))
                        waitingForFirstConfigOrFallbackStartInstant = nil
                    }
                    action = .scheduleTick
                case .reportUpdatedConfigAndScheduleTick(old: let old, new: let new):
                    try await self.reportUpdatedConfig(old: old, new: new)
                    action = .scheduleTick
                case .reportConfigRecoveredFromFallbackAndScheduleTick(let configurationPackage):
                    try await self.reportRecoveredFromFallback(configurationPackage)
                    action = .scheduleTick
                case .reportErrorAndApplyConfig(let error, let configurationPackage):
                    self.reportError(error)
                    action = .applyConfig(configurationPackage)
                case .applyFallbackAndScheduleTick:
                    try await self.applyFallback()
                    if let start = waitingForFirstConfigOrFallbackStartInstant {
                        self.metrics.emit(Metrics.Fetcher.FirstDelayHistogram(durationSinceStart: start))
                        waitingForFirstConfigOrFallbackStartInstant = nil
                    }
                    action = .scheduleTick
                case .reportErrorAndScheduleTick(let error):
                    self.reportError(error)
                    action = .scheduleTick
                case .markUnhealthyAndStopLoop(let unhealthyReason):
                    Self.logger.error("Marking unhealthy and exiting due to: \(unhealthyReason, privacy: .public)")
                    return
                }
            }
        }
    }

    private func reportConfigurationChanged(from oldRevision: String?, to newRevision: String) {
        if let oldRevision {
            self.metrics.emit(Metrics.Fetcher.CurrentRevisionGauge(revision: oldRevision, isCurrent: false))
            self.metrics.emit(Metrics.Fetcher.IsFallbackGauge(value: 0))
        }
        self.metrics.emit(Metrics.Fetcher.CurrentRevisionGauge(revision: newRevision, isCurrent: true))
        self.metrics.emit(Metrics.Fetcher.IsRevisionGauge(value: 1))
        self.metrics.emit(Metrics.Fetcher.IsWaitingForFirstGauge(value: 0))
    }

    private func reportError(_ error: FetcherStateMachine.StringError) {
        Self.logger.error("Received an error: \(error.message, privacy: .public)")
        self.metrics.emit(Metrics.Fetcher.ErrorCounter(action: .increment))
    }

    private func reportFirstConfig(_ config: NodeConfigurationPackage) async throws {
        Self.logger.notice("First configuration applied: \(config, privacy: .public)")
        self.reportConfigurationChanged(from: nil, to: config.revisionIdentifier)
    }

    private func reportRecoveredFromFallback(_ config: NodeConfigurationPackage) async throws {
        Self.logger.notice("Recovered from fallback with config: \(config, privacy: .public)")
        self.reportConfigurationChanged(from: nil, to: config.revisionIdentifier)
    }

    private func reportUpdatedConfig(old: NodeConfigurationPackage, new: NodeConfigurationPackage) async throws {
        Self.logger.notice("Updated config from \(old, privacy: .public) to \(new, privacy: .public)")
        self.reportConfigurationChanged(from: old.revisionIdentifier, to: new.revisionIdentifier)
    }

    private func sleepUntil(_ instant: ContinuousClock.Instant) async throws {
        let sleepDuration = ContinuousClock.now.duration(to: instant).components.seconds
        Self.logger
            .debug(
                "Will sleep until (\(sleepDuration, privacy: .public) seconds from now, schedule: \(self.schedule, privacy: .public))"
            )
        self.metrics.emit(Metrics.Fetcher.SleepDurationHistogram(value: Double(sleepDuration)))
        return try await withLogging(
            operation: "sleepUntil",
            sensitiveError: false,
            logger: Self.logger
        ) {
            try await Task.sleep(until: instant)
        }
    }

    private func fetchConfig(currentRevisionIdentifier: String?) async throws -> FetcherResult<FetchLatestResult> {
        self.metrics.emit(Metrics.Fetcher.UpstreamRequestCounter(action: .increment))
        do {
            return try await withLogging(
                operation: "fetchConfig (current: \(currentRevisionIdentifier ?? "<nil>"))",
                sensitiveError: false,
                logger: Self.logger
            ) {
                let result = try await self.dataSource.fetchLatest(
                    currentRevisionIdentifier: currentRevisionIdentifier
                )
                Self.logger.debug("Upstream responded with: \(result, privacy: .public)")
                switch result {
                case .upToDate:
                    self.metrics.emit(Metrics.Fetcher.ConfigUpToDateCounter(action: .increment))
                case .newAvailable:
                    self.metrics.emit(Metrics.Fetcher.NewConfigFromUpstreamCounter(action: .increment))
                }
                return .success(result)
            }
        } catch let error as CancellationError {
            throw error
        } catch {
            self.metrics.emit(Metrics.Fetcher.UpstreamFailureCounter(action: .increment))
            return .failure(.init(message: "\(error)"))
        }
    }

    private func applyFallback() async throws {
        guard let delegate else {
            Self.logger.error("Missing delegate.")
            return
        }
        self.metrics.emit(Metrics.Fetcher.FallbackToRegistryCounter(action: .increment))
        self.metrics.emit(Metrics.Fetcher.IsWaitingForFirstGauge(value: 0))
        self.metrics.emit(Metrics.Fetcher.IsFallbackGauge(value: 1))
        self.metrics.emit(Metrics.Fetcher.IsRevisionGauge(value: 0))
        do {
            try await withErrorLogging(
                operation: "applyFallback",
                sensitiveError: false,
                logger: Self.logger,
                level: .debug
            ) {
                try await delegate.applyFallback()
            }
        } catch let error as CancellationError {
            throw error
        } catch {
            // The error already gets logged above, but we don't pipe it back to the state machine.
        }
    }

    private func applyConfiguration(
        _ configuration: NodeConfigurationPackage
    ) async throws -> FetcherResult<NodeConfigurationPackage> {
        guard let delegate else {
            Self.logger.error("Missing delegate.")
            return .failure(.init(message: "Missing delegate."))
        }
        self.metrics.emit(Metrics.Fetcher.NewConfigToRegistryCounter(action: .increment))
        do {
            return try await withErrorLogging(
                operation: "applyConfiguration \(configuration.revisionIdentifier)",
                sensitiveError: false,
                logger: Self.logger,
                level: .debug
            ) {
                let startInstant = ContinuousClock.now
                try await delegate.applyConfiguration(configuration)
                self.metrics.emit(Metrics.Fetcher.RegistryReplySuccessCounter(action: .increment))
                self.metrics
                    .emit(Metrics.Fetcher.RegistryApplyingSuccessDurationHistogram(durationSinceStart: startInstant))
                return .success(configuration)
            }
        } catch let error as CancellationError {
            throw error
        } catch {
            self.metrics.emit(Metrics.Fetcher.RegistryReplyFailureCounter(action: .increment))
            // The error already gets logged above, and we provide it back to the state machine as well.
            return .failure(.init(message: "\(error)"))
        }
    }
}

extension Fetcher.SchedulingConfiguration {
    /// Calculates when the next tick should happen, based on the latest tick start instant.
    ///
    /// If the latest tick start instant is in the past, the next tick will happen immediately.
    ///
    /// - Parameter lastTickStartInstant: The latest tick start instant.
    /// - Returns: The next tick instant.
    func nextTickInstant(lastTickStartInstant: ContinuousClock.Instant) -> ContinuousClock.Instant {
        let jitterRatio = Double.random(in: -maximumToleranceRatio ... maximumToleranceRatio)
        let jitter = tickInterval * jitterRatio
        let nextTickInstant = lastTickStartInstant + tickInterval + jitter
        return max(nextTickInstant, ContinuousClock.now + (tickInterval * minimumInterTickDelayRatio))
    }
}

extension Fetcher.SchedulingConfiguration: CustomStringConvertible {
    var description: String {
        "tickInterval: \(tickInterval), maximumToleranceRatio: \(maximumToleranceRatio), minimumInterTickDelayRatio: \(minimumInterTickDelayRatio)"
    }
}

extension Fetcher: ConfigurationServerInfoDelegate {
    func currentConfigurationVersionInfo(
        connectionID _: ConnectionID
    ) async -> ConfigurationVersionInfo {
        let result: ConfigurationVersionInfo = switch self.stateMachine.state {
        case .initial, .terminalUnhealthy, .failedApplyingFirstConfig, .pendingFirstConfig:
            .init(appliedVersion: nil)
        case .appliedConfig(let config):
            .init(appliedVersion: .revision(config.revisionIdentifier))
        case .appliedFallback, .pendingFromFallback, .failedApplyingFromFallback:
            .init(appliedVersion: .fallback)
        }
        let description = result.appliedVersion?.description ?? "<nil>"
        Fetcher.logger.debug("currentConfigurationVersionInfo called, returning: \(description, privacy: .public)")
        return result
    }
}
