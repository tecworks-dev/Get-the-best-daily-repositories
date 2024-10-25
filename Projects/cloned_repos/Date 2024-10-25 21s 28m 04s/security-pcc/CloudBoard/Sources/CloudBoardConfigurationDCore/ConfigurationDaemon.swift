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

// Copyright © 2023 Apple. All rights reserved.

import CFPreferenceCoder
import CloudBoardAsyncXPC
import CloudBoardConfigurationDAPI
import CloudBoardLogging
import CloudBoardMetrics
import DarwinPrivate.dirhelper
import Foundation
import os
import System

/// An actor that represents the top level of the configuration daemon and runs all the nested tasks.
public actor ConfigurationDaemon {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudBoardConfigurationDaemon"
    )
    static let metricsClientName = "cb_configurationd"

    /// Creates a new daemon that uses the local XPC listener.
    /// - Parameter configurationPath: A path to the configuration path. If nil, the configuration
    ///   gets read from the preferences.
    /// - Throws: If loading the configuration fails.
    public init(configurationPath: String? = nil) async throws {
        Self.configureTempDir()
        let daemonConfiguration: ConfigurationDConfiguration = if let configurationPath {
            try ConfigurationDConfiguration.fromFile(path: configurationPath, secureConfigLoader: .real)
        } else {
            try ConfigurationDConfiguration.fromPreferences()
        }
        let launchDelay = daemonConfiguration.launchDelay
        try await withLogging(operation: "launchDelay \(launchDelay) seconds", sensitiveError: false) {
            try await Task.sleep(for: .seconds(launchDelay))
        }
        Self.logger.info("Bootstrapping metrics with client name \(Self.metricsClientName, privacy: .public)")
        let metricsSystem = CloudMetricsSystem(clientName: Self.metricsClientName)
        let dataSource = try Fetcher.dataSourceFromConfiguration(
            daemonConfiguration,
            logger: Self.logger,
            metrics: metricsSystem
        )
        let xpcListener = CloudBoardAsyncXPCListener.configurationDaemon
        let launchStatsStore = OnDiskLaunchStatsStore(
            directoryURL: FileManager.default.temporaryDirectory
        )
        self.init(
            fetcherSchedule: .init(daemonConfiguration.fetcherSchedule),
            fetcherDataSource: dataSource,
            xpcListener: xpcListener,
            metrics: metricsSystem,
            launchStatsStore: launchStatsStore
        )
    }

    /// The schedule of the fetcher.
    private let fetcherSchedule: Fetcher.SchedulingConfiguration

    /// The data source of the fetcher.
    private let fetcherDataSource: FetcherDataSource

    /// The underlying XPC listener.
    private let xpcListener: CloudBoardAsyncXPCListener

    /// The metrics system to use.
    private let metrics: any MetricsSystem

    /// Launch stats for the daemon.
    private let launchStatsStore: LaunchStatsStoreProtocol

    /// Creates a new daemon.
    /// - Parameters:
    ///   - fetcherPollingInterval: The polling interval of the fetcher.
    ///   - fetcherDataSource: The data source of the fetcher.
    ///   - xpcListener: The underlying XPC listener.
    ///   - metrics: The metrics system to use.
    ///   - launchStatsStore: Launch stats for the daemon.
    init(
        fetcherSchedule: Fetcher.SchedulingConfiguration,
        fetcherDataSource: FetcherDataSource,
        xpcListener: CloudBoardAsyncXPCListener,
        metrics: any MetricsSystem,
        launchStatsStore: LaunchStatsStoreProtocol
    ) {
        self.fetcherSchedule = fetcherSchedule
        self.fetcherDataSource = fetcherDataSource
        self.xpcListener = xpcListener
        self.metrics = metrics
        self.launchStatsStore = launchStatsStore
    }

    /// Emit the launch metrics.
    private func emitLaunchMetrics(stats: LaunchStats) {
        self.metrics.emit(Metrics.Daemon.LaunchCounter(action: .increment))
        self.metrics.emit(Metrics.Daemon.LaunchNumberGauge(value: stats.launchNumber))
        if let previousLaunchDate = stats.previousLaunchDate {
            let timeSinceLastLaunch = stats.launchDate.timeIntervalSince(previousLaunchDate)
            self.metrics.emit(Metrics.Daemon.TimeSinceLastLaunchHistogram(value: timeSinceLastLaunch))
        }
    }

    /// Updates the metrics with the current values, called periodically.
    /// - Parameter startInstant: The instant at which the daemon started.
    private func updateMetrics(startInstant: ContinuousClock.Instant) {
        let uptime = Int(clamping: startInstant.duration(to: .now).components.seconds)
        self.metrics.emit(Metrics.Daemon.UptimeGauge(value: uptime))
    }

    /// Runs the daemon and all its nested tasks.
    public func run() async {
        let launchStats = await launchStatsStore.loadAndUpdateLaunchStats()
        Self.logger.notice("Starting the configuration daemon.")
        Self.logger.info("Launch stats: \(launchStats, privacy: .public).")
        self.emitLaunchMetrics(stats: launchStats)
        let registry = Registry(metrics: metrics)
        let fetcher = Fetcher(
            schedule: fetcherSchedule,
            dataSource: fetcherDataSource,
            metrics: metrics
        )
        let server = ConfigurationServer(
            xpcServer: ConfigurationAPIXPCServer(
                listener: xpcListener
            )
        )
        let startInstant = ContinuousClock.now
        do {
            try await withErrorLogging(
                operation: "Configuration daemon task group",
                sensitiveError: false,
                logger: Self.logger,
                level: .debug
            ) {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    await fetcher.set(delegate: registry)
                    await server.set(delegate: registry)
                    await server.set(infoDelegate: fetcher)
                    group.addTaskWithLogging(
                        operation: "Fetcher task",
                        sensitiveError: false,
                        logger: Self.logger,
                        level: .debug
                    ) {
                        try await fetcher.run()
                    }
                    group.addTaskWithLogging(
                        operation: "Server task",
                        sensitiveError: false,
                        logger: Self.logger,
                        level: .debug
                    ) {
                        try await server.run()
                    }
                    group.addTaskWithLogging(
                        operation: "Update metrics task",
                        sensitiveError: false,
                        logger: Self.logger,
                        level: .debug
                    ) {
                        while true {
                            await self.updateMetrics(startInstant: startInstant)
                            try await Task.sleep(for: .seconds(5))
                        }
                    }
                    try await group.next()
                    group.cancelAll()
                }
            }
            Self.logger.notice("Exiting the configuration daemon")
        } catch {
            Self.logger.fault("Fatal error, exiting: \(String(unredacted: error), privacy: .public)")
        }
        self.metrics.emit(Metrics.Daemon.ExitedLoopCounter(action: .increment))
    }

    private static func configureTempDir() {
        guard _set_user_dir_suffix(CFPreferences.cbConfigPreferencesDomain) else {
            let error = Errno(rawValue: errno)
            Self.logger.fault("""
            Failed to set temporary directory suffix: \(error, privacy: .public)
            """)
            fatalError("Failed to set temporary directory suffix: \(error)")
        }
        // Ensure the directory is created and TMPDIR is set.
        _ = NSTemporaryDirectory()
    }
}
