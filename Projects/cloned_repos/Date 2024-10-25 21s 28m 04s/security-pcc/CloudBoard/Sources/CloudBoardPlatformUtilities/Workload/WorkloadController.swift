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

import CloudBoardAsyncXPC
import CloudBoardCommon
import CloudBoardController
import CloudBoardMetrics
import os

public enum WorkloadControllerError: Error, Equatable, CustomStringConvertible {
    case controllerUnknownState
    case controllerDisconnected

    public var description: String {
        switch self {
        case .controllerUnknownState:
            return "Controller reported unknown state"
        case .controllerDisconnected:
            return "Controller disconnected"
        }
    }
}

public actor WorkloadController {
    private var state: ControllerState
    private let healthPublisher: ServiceHealthMonitor
    private var server: CloudBoardControllerAPIXPCServer?
    private var serviceDiscoveryPublisher: ServiceDiscoveryPublisherProtocol?
    private var providerPause: (() async throws -> Void)?
    private var concurrentRequestCount: Int
    private var announcedService: Bool = false
    private var metrics: MetricsSystem

    private var workloadProperties: WorkloadProperties?
    private var workloadConfig: WorkloadConfig?

    public enum ControllerState: Equatable, CustomStringConvertible {
        case initializing
        case ready
        case busyTransitioning
        case busy
        case error(WorkloadControllerError?)

        public var description: String {
            switch self {
            case .initializing:
                return "Initializing"
            case .ready:
                return "Ready"
            case .busyTransitioning:
                return "Busy (Transitioning)"
            case .busy:
                return "Busy"
            case .error(let error):
                return "Error: \(String(describing: error))"
            }
        }
    }

    public static let log: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "WorkloadController"
    )

    public init(
        healthPublisher: ServiceHealthMonitor,
        testConfig: (WorkloadProperties, WorkloadConfig)? = nil,
        metrics: MetricsSystem
    ) {
        self.metrics = metrics
        self.state = .initializing
        self.healthPublisher = healthPublisher
        self.concurrentRequestCount = 0

        if let (testWorkloadProperties, testWorkloadConfig) = testConfig {
            self.workloadProperties = testWorkloadProperties
            self.workloadConfig = testWorkloadConfig
            self.state = .ready
        }
        self.metrics.emit(Metrics.WorkloadStatus.HealthStatus(value: 1, healthStatus: self.state))
    }

    public func run(
        serviceDiscoveryPublisher: ServiceDiscoveryPublisherProtocol?,
        concurrentRequestCountStream: AsyncStream<Int>,
        providerPause: @escaping () async throws -> Void
    ) async throws {
        self.serviceDiscoveryPublisher = serviceDiscoveryPublisher
        self.providerPause = providerPause
        self.server = CloudBoardControllerAPIXPCServer.localListener()
        Self.log.info("Starting WorkloadController server")
        await self.server?.connect(listenerDelegate: self, serverDelegate: self)
        Self.log.info("WorkloadController server started")

        // For testing only -- if initialized with a test config instead of
        // using real controller then announce the config immediately.
        if case .ready = self.state,
           let properties = self.workloadProperties,
           let config = self.workloadConfig {
            Self.log.info("""
            Announcing test service \(properties.workloadName, privacy: .public)
            """)
            self.sendHealthPublisherUpdate()
            self.serviceDiscoveryPublisher?.announceService(
                name: properties.workloadName,
                workloadConfig: config.routingTags
            )
        }

        try await self.concurrentRequestCountMonitor(
            concurrentRequestCountStream: concurrentRequestCountStream
        )
    }

    // on follower nodes we dont need service discovery
    public func runInFollowerMode() async throws {
        Self.log.info("Starting WorkloadController server")
        self.server = CloudBoardControllerAPIXPCServer.localListener()
        await self.server?.connect(listenerDelegate: self, serverDelegate: self)
        Self.log.info("WorkloadController server started")

        while true {
            // Sleep for an hour. This is the longest reasonable interval to approximate
            // "wait forever" without producing unreasonably large numbers.
            try await Task.sleep(for: .seconds(60 * 60))
        }
    }

    private func sendServiceDiscoveryUpdate() {
        guard let properties = self.workloadProperties,
              let config = self.workloadConfig else {
            return
        }

        switch self.state {
        case .ready:
            if self.announcedService == false {
                self.serviceDiscoveryPublisher?.announceService(
                    name: properties.workloadName,
                    workloadConfig: config.routingTags
                )
                self.announcedService = true
            }
        case .error:
            guard self.announcedService else {
                return
            }
            self.serviceDiscoveryPublisher?.retractService(
                name: properties.workloadName
            )
            self.announcedService = false
        case .initializing, .busy, .busyTransitioning:
            return
        }
    }

    private func sendHealthPublisherUpdate() {
        if case .error = self.state {
            self.healthPublisher.updateStatus(.unhealthy)
            return
        }

        guard let properties = self.workloadProperties else {
            Self.log.warning("""
            Skipping health publisher update because workload properties \
            have not been received
            """)
            return
        }
        guard let config = self.workloadConfig else {
            Self.log.warning("""
            Skipping health publisher update because workload config \
            has not been received
            """)
            return
        }

        let maxBatchSize: Int
        let optimalBatchSize: Int
        if case .ready = self.state {
            maxBatchSize = config.maxBatchSize
            optimalBatchSize = config.optimalBatchSize
        } else {
            maxBatchSize = 0
            optimalBatchSize = 0
        }
        self.healthPublisher.updateStatus(.healthy(.init(
            workloadType: properties.workloadName,
            tags: config.routingTags,
            maxBatchSize: maxBatchSize,
            currentBatchSize: config.currentBatchSize,
            optimalBatchSize: optimalBatchSize
        )))
    }

    public func shutdown() async throws {
        try await self.server?.shutdown()
    }

    private func concurrentRequestCountMonitor(
        concurrentRequestCountStream: AsyncStream<Int>
    ) async throws {
        for try await concurrentRequestCount in concurrentRequestCountStream {
            self.concurrentRequestCount = concurrentRequestCount
            self.sendHealthPublisherUpdate()
        }
    }
}

extension WorkloadController: CloudBoardControllerAPIServerDelegateProtocol {
    public func registerWorkload(
        config: WorkloadConfig,
        properties: WorkloadProperties
    ) async throws {
        Self.log.log("""
        Received 'RegisterWorkload' from CloudBoardController: \
        config=\(config.description, privacy: .public), \
        properties=\(properties.description, privacy: .public)
        """)

        if config.optimalBatchSize > config.maxBatchSize {
            Self.log.warning(
                "Optimal batch size is larger than max batch size"
            )
        }

        self.workloadConfig = config
        self.workloadProperties = properties
        self.announcedService = false

        self.sendServiceDiscoveryUpdate()
        self.sendHealthPublisherUpdate()
    }

    public func updateHealthStatus(
        status: WorkloadControllerStatus
    ) async throws {
        Self.log.log("""
        Received 'UpdateHealthStatus' from CloudBoardController: \
        status=\(status.description, privacy: .public)
        """)

        let oldWorkloadState = self.state
        switch status.state {
        case .initializing:
            self.state = .initializing
        case .ready:
            self.state = .ready
        case .busy:
            if case .busyTransitioning = self.state {
                throw CloudBoardControllerAPIError.alreadyTransitioning(.busy)
            }
            self.state = .busyTransitioning
        case .error:
            self.state = .error(nil)
        default:
            self.state = .error(.controllerUnknownState)
        }

        if oldWorkloadState != self.state {
            self.metrics.emit(Metrics.WorkloadStatus.HealthStatus(value: 0, healthStatus: oldWorkloadState))
            self.metrics.emit(Metrics.WorkloadStatus.HealthStatus(value: 1, healthStatus: self.state))
        }

        self.sendServiceDiscoveryUpdate()
        self.sendHealthPublisherUpdate()

        if case .busyTransitioning = self.state {
            do {
                try await self.providerPause!()
            } catch {
                throw CloudBoardControllerAPIError.alreadyTransitioning(.busy)
            }
        }
        if case .busyTransitioning = self.state {
            self.state = .busy
            self.metrics.emit(Metrics.WorkloadStatus.HealthStatus(value: 0, healthStatus: .busyTransitioning))
            self.metrics.emit(Metrics.WorkloadStatus.HealthStatus(value: 1, healthStatus: .busy))
        }
    }
}

extension WorkloadController: CloudBoardAsyncXPCListenerDelegate {
    public func invalidatedConnection(_ connection: CloudBoardAsyncXPCConnection) async {
        Self.log.log("""
        Received 'InvalidatedConnection' from CloudBoardController: \
        id=\(String(describing: connection.id), privacy: .public)
        """)

        // Ignoring ID since we only support one controller connection.
        self.state = .error(
            WorkloadControllerError.controllerDisconnected
        )

        self.sendServiceDiscoveryUpdate()
        self.sendHealthPublisherUpdate()
    }
}
