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

import CloudAttestation
@_exported import CloudMetricsFramework
import CryptoKit
import Foundation
import os
import OSPrivate.os.transactionPrivate // Make the Ensembler dirty
import notify

@_spi(Daemon) import Ensemble // Helper functions

// MARK: - Ensembler Global Constants -

public let kEnsemblerPrefix = "com.apple.cloudos.AppleComputeEnsembler"
let kSkipDarwinInitCheckPreferenceKey = "SkipDarwinInitCheck"

// MARK: - Ensembler

internal enum DerivedKeyType: CustomStringConvertible {
    
    case MeshEncryption
    case TlsPsk
    case KeyConfirmation(String)
    case TestEncryptDecrypt
    
    var description: String {
        switch(self) {
        case .MeshEncryption:
            return "ensembled-mesh-encryption"
        case .TlsPsk:
            return "ensembled-tls-psk"
        case .KeyConfirmation(let nodeUDID):
            return "ensembled-key-confirmation-\(nodeUDID)"
        case .TestEncryptDecrypt:
            return "ensembled-test-encrypt-decrypt"
        }
    }
}

public func initCloudMetricsFrameworkBackend() {
	CloudMetrics.bootstrap(clientName: "ensembled")
}

private func handleStateTransition(
	to currentState: EnsemblerStatus,
	isLeader: Bool,
	rank: Int,
	chassisID: String,
    nodeCnt: Int
) {
	Ensembler.logger.info("Exporting metrics")

	// increment counter for the number of transitions
	let counter = Counter(
		label: "ensembled_statetransitions_total",
		dimensions: [
			("to_state", "\(currentState)"),
			("is_leader", "\(isLeader)"),
			("rank", "\(rank)"),
			("chassisID", chassisID),
            ("nodeCount", "\(nodeCnt)"),
		]
	)
	counter.increment()

	// update the guage metrics as well.
	updateCurrentStateCountGaugeMetrics(
		newState: currentState,
		isLeader: isLeader,
		rank: rank,
		chassisID: chassisID,
        nodeCnt: nodeCnt
	)

	Ensembler.logger.info("Done exporting metrics.")
}

/// Updates a gauge counter which represents a set of different states numerically as 0 or 1, where
/// the gauge with a given dimension set to one of the states set to 0 means that something is not
/// in that state
/// but a gauge with a given dimension set to a different state set to 1 means that something IS IN
/// that state.
/// For the case of EnsemblerStatus, if the node is in initialized state, the guage with the
/// dimension
/// "tostate": "initialized" is set to 1 and guage with dimension "tostate": all other states is set
/// to 0.
///  This allows both easily aggreagating the count of nodes in a given state across the fleet
/// at a point in time as well as viewing the history of the state for a given node as it
/// transitions from one state
/// to another.
/// - Parameters:
///   - currentState: the state that is active for the gauge
///   - mkGauge: a function to create the gauge object, this should set the label and dimensions on
/// the gauge
private func updateGenericStateGaugeCounter<T: CustomStringConvertible & CaseIterable & Equatable>(
	currentState: T,
	mkGauge: (_ state: T) -> Gauge
) {
	// Set all guages for all other states to 0 before setting the current
	// state to 1 to avoid sampling races.
	for state in T.allCases {
		guard state != currentState else { continue }

		let gauge = mkGauge(state)
		gauge.record(0)
	}

	let gauge = mkGauge(currentState)
	gauge.record(1)
}

/// Update the current state metrics gauges for this compute node with
/// either the specified new state
/// - Parameter newState: the new state
/// - Parameter isLeader: whether the node sending metrics is leader or not. Leader if true,
/// Follower if false.
///                       However in case of  .uninitialized state, this should be ignored.
/// - Parameter rank: rank of the node, In case of .uninitialized state, this should be ignored.
/// - Parameter chassisID: chassisID of the node. In case of .uninitialized state, this should be
/// ignored.
///
private func updateCurrentStateCountGaugeMetrics(
	newState: EnsemblerStatus,
	isLeader: Bool,
	rank: Int,
	chassisID: String,
    nodeCnt: Int
) {
	updateGenericStateGaugeCounter(currentState: newState) { state in
		Gauge(
			label: "ensembled_state_total",
			dimensions: [
				("to_state", "\(state)"),
				("is_leader", "\(isLeader)"),
				("rank", "\(rank)"),
				("chassisID", chassisID),
                ("nodeCount", "\(nodeCnt)"),
			]
		)
	}
}

/// Public interface to the ensembler
public class Ensembler {
	let ensembleConfig: EnsembleConfiguration
	public let ensembleID: String?
	let currentNodeConfig: NodeConfiguration
	let UDID: String
	let imTheBoss: Bool
	var slots = [Slot]()
	let transaction: os_transaction_t
	let autoRestart: Bool
	let doDarwinInitCheck: Bool
	let darwinInitTimeout: Int?
	var everyoneFound = false
	var doneInitializing = false

	/// Public view of the ensemble configuration
	public var nodeMap: [String: EnsembleNodeInfo] = [:]

	// `toBackendQ` is dedicated to the backend. It should NOT be used by the ensembler.
	static let toBackendQ = DispatchSerialQueue(label: "\(kEnsemblerPrefix).to.backend.queue")

	// `fromBackendQ` serves two purposes
	//   1. Serialize operations from the backend. For example, two `incomingMessage()` requests
	//   arriving at the ensembler concurrently are coordinated through `fromBackendQ`.
	//   2. Offload delegate handlers onto a new thread of execution. Specifically, a thread
	//   handling a backend delegate operation is NOT allowed to call back into the backend.
	//   Otherwise, ensembled can deadlock. And this can happen on the failure code path, which
	//   calls into the backend to deactivate the mesh. Thus, backend delegate operations should
	//   always be async enqueued.
	static let fromBackendQ = DispatchSerialQueue(label: "\(kEnsemblerPrefix).from.backend.queue")

	// `drainingQ` serializes all reads/writes to `_draining`.
	let drainingQ = DispatchSerialQueue(label: "\(kEnsemblerPrefix).draining.queue")

	private let attestationAsyncQueue = AsyncQueue()
	private var plainTextUUID = UUID().uuidString
	static let logger = Logger(subsystem: kEnsemblerPrefix, category: "Ensembler")

	// TODO: This class can be referenced before this is initialized; gross
	private var router: Router?
	private var backend: Backend?

	public var status: EnsemblerStatus {
		get {
			return self.stateMachine.state
		}
	}
	private var stateMachine: any StateMachine

	private var _draining = false
	public var draining: Bool {
		get {
			return drainingQ.sync {
				return self._draining
			}
		}
		set {
			drainingQ.sync {
				self._draining = newValue
			}
		}
	}

	private var useStubAttestation: Bool
	private var cloudAttestation: Attestation?
	private var sharedKey: SymmetricKey?
    private var initialSharedkey: SymmetricKey?
	let maxControlMsgSize = 2048
	var dataMap: [String: Data] = [:]

	private let jobQuiescenceMonitor = JobQuiescenceMonitor()

	private func setStatus(_ state: EnsemblerStatus) -> Bool {
		do {
			try self.stateMachine.goto(targetState: state)
		} catch {
			Self.logger.error(
				"Oops: Failed on state transition \(self.status) -> \(state): \(error)"
			)
			self.ensembleFailed()
			return false
		}

		// update the metrics about the state
		handleStateTransition(
			to: self.status,
			isLeader: self.imTheBoss,
			rank: self.currentNodeConfig.rank,
			chassisID: self.currentNodeConfig.chassisID,
			nodeCnt: self.nodeMap.count
		)

		//post a notification for entering to ready and failed state
		if (state == .ready) {
			Ensembler.logger.info(
				"Posting notifiication: \(kEnsembleStatusReadyEventName)"
			)
			notify_post(kEnsembleStatusReadyEventName)
		}
		if (state.inFailedState()) {
			Ensembler.logger.info(
				"Posting notifiication: \(kEnsembleStatusFailedEventName)"
			)
			notify_post(kEnsembleStatusFailedEventName)
		}

		return true
	}

	init(
		ensembleConfig: EnsembleConfiguration,
		autoRestart: Bool = false,
		skipDarwinInitCheck: Bool = false,
		darwinInitTimeout: Int? = nil,
		useStubAttestation: Bool = false
	) throws {
		self.ensembleConfig = ensembleConfig
		self.ensembleID = ensembleConfig.ensembleID
		// This is OK to log publically because the private fields are obfuscated.
		Ensembler.logger.info(
			"""
			Initalizing ensembler: ensembleConfig [.public]: \
			\(String(reportableDescription: self.ensembleConfig), privacy: .public)
			"""
		)
		Ensembler.logger.info(
			"""
			Initalizing ensembler: ensembleConfig [.private]: \
			\(self.ensembleConfig, privacy: .private)
			"""
		)
		self.autoRestart = autoRestart
		Ensembler.logger.info(
			"Initalizing ensembler: autoRestart: \(self.autoRestart, privacy: .public)"
		)
		self.doDarwinInitCheck = !skipDarwinInitCheck
		Ensembler.logger.info(
			"Initalizing ensembler: doDarwinInitCheck: \(self.doDarwinInitCheck, privacy: .public)"
		)
		self.darwinInitTimeout = darwinInitTimeout
		Ensembler.logger.info(
			"""
			Initalizing ensembler: darwinInitTimeout: \
			\(String(describing: self.darwinInitTimeout), privacy: .public)
			"""
		)
		self.useStubAttestation = useStubAttestation
		Ensembler.logger.info(
			"""
			Initalizing ensembler: useStubAttestation: \(self.useStubAttestation, privacy: .public)
			"""
		)
		self.UDID = try getNodeUDID()
		guard let nodeConfig = self.ensembleConfig.nodes[self.UDID] else {
			Ensembler.logger.error(
				"Failed to find configuration for UDID: \(self.UDID, privacy: .private)"
			)
			throw InitializationError.cannotFindSelfInConfiguration
		}
		Ensembler.logger.info(
			"Initalizing ensembler: UDID: \(self.UDID, privacy: .private)"
		)

		self.currentNodeConfig = nodeConfig
		// This is OK to log publically because the private fields are obfuscated.
		Ensembler.logger.info(
			"""
			Initalizing ensembler: currentNodeConfig [.public]: \
			\(String(reportableDescription: self.currentNodeConfig), privacy: .public)
			"""
		)
		Ensembler.logger.info(
			"""
			Initalizing ensembler: currentNodeConfig [.private]: \
			\(self.currentNodeConfig, privacy: .private)
			"""
		)
		self.imTheBoss = (self.currentNodeConfig.rank == 0 ? true : false)
		Ensembler.logger.info(
			"Initalizing ensembler: imTheBoss: \(self.imTheBoss, privacy: .public)"
		)

		for rank in 0..<self.ensembleConfig.nodes.count {
			if rank == self.currentNodeConfig.rank {
				self.slots.append(Util.slot())
			} else {
				self.slots.append(.notInitialized)
			}
		}

		// send metrics with initializing status
		// calling this explicitly, since calling setStatus is not allowed before initializing all
		// members.
		handleStateTransition(
			to: .initializing,
			isLeader: self.imTheBoss,
			rank: self.currentNodeConfig.rank,
			chassisID: self.currentNodeConfig.chassisID,
			nodeCnt: self.ensembleConfig.nodes.count
		)

		for node in self.ensembleConfig.nodes {
			self.nodeMap[node.key] = EnsembleNodeInfo(
				leader: node.value.rank == 0 ? true : false,
				UDID: node.key,
				rank: node.value.rank,
				hostName: node.value.hostName
			)
		}
		Ensembler.logger.info(
			"Initalizing ensembler: nodeMap.count: \(self.nodeMap.count, privacy: .public)"
		)

		// Create the `StateMachine`, used to track the ensemble.
		if self.imTheBoss {
			self.stateMachine = try LeaderStateMachine(singleNode: self.nodeMap.count == 1)
		} else {
			self.stateMachine = try FollowerStateMachine(singleNode: self.nodeMap.count == 1)
		}

		// The ensembler has an in-memory state machine and is thus always dirty.
		self.transaction = os_transaction_create(kEnsemblerPrefix)

		let backendConfig = BackendConfiguration(
			queue: Ensembler.toBackendQ,
			node: self.currentNodeConfig,
			ensemble: self.ensembleConfig,
			delegate: self
		)

		let backendType: BackendType
		// We use stub backend if node count is 1
		// For all other cases, we assume CIOBackend if nothing is provided
		if self.nodeMap.count == 1 {
			backendType = .StubBackend

			// For a one-node ensemble, out of an abundance of caution, lock the backend.
			do {
				let tmpBackend = try CIOBackend(configuration: backendConfig)
				try tmpBackend.lock()
			} catch {
				Self.logger.error(
					"""
					Oops: Failed to lock CIO backend for 1-node ensemble: \
					Maybe this system doesn't have a CIO backend? Ignoring error: \(error)
					"""
				)
			}
		} else {
			backendType = ensembleConfig.backendType ?? .CIOBackend
		}
		Ensembler.logger.info(
			"Initalizing ensembler: backendType: \(backendType, privacy: .public)"
		)

		switch backendType {
		case .CIOBackend:
			self.backend = try CIOBackend(configuration: backendConfig)
		case .StubBackend:
			self.backend = try StubBackend(configuration: backendConfig)
		default:
			Ensembler.logger.error(
				"Invalid backend: \(backendType, privacy: .public)"
			)
			throw InitializationError.invalidBackend
		}

		guard let backend = self.backend else {
			Ensembler.logger.error("Initalizing ensembler: nil backend: This should never happen!")
			throw InitializationError.invalidBackend
		}
		let routerConfiguration = RouterConfiguration(
			backend: backend,
			node: self.currentNodeConfig,
			ensemble: self.ensembleConfig,
			delegate: self
		)

		switch (self.ensembleConfig.nodes.count) {
		case(1):
			Ensembler.logger.info("Initalizing ensembler: 1-node ensemble: Skip router creation")
		case (2):
			self.router = try Router2(configuration: routerConfiguration)
		case (4):
			self.router = try Router4(configuration: routerConfiguration)
		case (8):
			self.router = try Router8Hypercube(configuration: routerConfiguration)
		default:
			Ensembler.logger.error(
				"""
				Illegal ensemble size: \(self.ensembleConfig.nodes.count, privacy: .public)
				"""
			)
			throw InitializationError.invalidRouterTopology(count: self.ensembleConfig.nodes.count)
		}

		try self.jobQuiescenceMonitor.start(ensembler: self)

		Ensembler.logger.info(
			"The Ensembler has been assembled and is ready to assemble ensembles."
		)
	}

	/// Read preferences if no configuration is offered
	public convenience init(
		autoRestart: Bool = false,
		skipDarwinInitCheckOpt: Bool? = false,
		darwinInitTimeout: Int? = nil,
		useStubAttestation: Bool = false,
		allowDefaultOneNodeConfig: Bool = false
	) throws {
		let ensembleConfig: EnsembleConfiguration
		var skipDarwinInitCheck: Bool
		do {
			ensembleConfig = try readEnsemblerPreferences()
			skipDarwinInitCheck = false
		} catch {
			Ensembler.logger.error(
				"""
				Failed to read ensemble config from CFPrefs: \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			if !allowDefaultOneNodeConfig {
				throw error
			}
			Ensembler.logger.info("Proceeding with default 1-node ensemble config.")
			ensembleConfig = try getDefaultOneNodePreferences()
			skipDarwinInitCheck = true
		}
		if let skipDarwinInitCheckOpt {
			skipDarwinInitCheck = skipDarwinInitCheckOpt
		} else {
			Ensembler.logger.info(
				"""
				Did not find \(kSkipDarwinInitCheckPreferenceKey, privacy: .public) preference, \
				defaulting to \(skipDarwinInitCheck, privacy: .public).
				"""
			)
		}
		try self.init(
			ensembleConfig: ensembleConfig,
			autoRestart: autoRestart,
			skipDarwinInitCheck: skipDarwinInitCheck,
			darwinInitTimeout: darwinInitTimeout,
			useStubAttestation: useStubAttestation
		)
	}

	private func dumpEnsembleDebugMap() {
		Ensembler.logger.info(
			"\(Util.ensembleDebugMap(ensembleConfig: self.ensembleConfig, slots: self.slots))"
		)
	}

	private func checkDarwinInit() -> Bool {
		Ensembler.logger.info(
			"""
			Running pre-activation check: darwin-init applied matches the BMC's expected darwin-init
			"""
		)
		let ok: Bool
		do {
			ok = try DarwinInitChecker(darwinInitTimeout: self.darwinInitTimeout).run()
		} catch {
			Ensembler.logger.error(
				"""
				DarwinInitChecker failed: \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			return false
		}
		return ok
	}

	/// Is the mesh currently locked?
	///  - Returns: `Bool` with lock status  or `false` if no backend found
	private func isLocked() -> Bool {
		let locked = self.backend?.isLocked() ?? false
		Ensembler.logger.info(
			"Backend locked status: \(locked ? "LOCKED" : "unlocked", privacy: .public)"
		)
		return locked
	}

	/// Activate the mesh in the underlying backend
	/// This function should NOOP if mesh is already active
	public func activate() throws {
		// Bail if we're not initializing; no need for this to be fatal
		guard self.status == .initializing else {
			Ensembler.logger.error(
				"Oops: Attempted to activate from an illegal state: \(self.status)"
			)
			throw InitializationError.invalidActivationState
		}

		do {
			guard let backend = self.backend else {
				Ensembler.logger.error(
					"Oops: Uninitialized backend: self.backend=\(String(describing: self.backend))"
				)
				throw InitializationError.invalidBackend
			}

			if self.doDarwinInitCheck {
				if !self.setStatus(.initializingDarwinInitCheckInProgress) {
					throw EnsembleError.illegalStateTransition
				}
				let darwinInitOK = self.checkDarwinInit()
				if darwinInitOK {
					if !self.setStatus(.initializingActivationChecksOK) {
						throw EnsembleError.illegalStateTransition
					}
				} else {
					// Note: When a preactivation check fails, there is no way to notify others because
					// the ensemble is not activated. So just mark ourselves as failed and return.
					if !self.setStatus(.failedActivationChecks) {
						throw EnsembleError.illegalStateTransition
					}
					return
				}
			}

			if self.useStubAttestation {
				Ensembler.logger.info("Initalizing ensembler: Using stub attestation")
				self.cloudAttestation = try StubAttestation(
					nodes: self.ensembleConfig.nodes,
					nodeUDID: self.UDID,
					isLeader: self.imTheBoss
				)
			} else {
				Ensembler.logger.info(
					"""
					Initalizing ensembler: cloudAttestation -> \
					EnsemblerCloudAttestation(
					 nodes: \(self.ensembleConfig.nodes),
					 nodeUDID: \(self.UDID),
					 isLeader: \(self.imTheBoss)
					)
					"""
				)
				self.cloudAttestation = try EnsemblerCloudAttestation(
					nodes: self.ensembleConfig.nodes,
					nodeUDID: self.UDID,
					isLeader: self.imTheBoss
				)
			}

			if self.imTheBoss {
				guard let cloudAttestation = self.cloudAttestation else {
					throw InitializationError.unexpectedBehavior(
						"Oops: Failed to unwrap self.cloudAttesation. But we just initialized it!"
					)
				}
				self.sharedKey = try cloudAttestation.getSymmetricKey()
			}

			if !backend.isLocked() {
				// This is the expected case in production where we activate the ensemble after
				// rebooting and install the "[ACDC|Trusted] Support" cryptex.
				try backend.activate()
			} else if self.autoRestart {
				// TODO: rdar://123743845 (Auto restart should check isActivate() when it becomes available)
				// TODO: And looking at the surrounding code, the check above should also check it.
				Ensembler.logger.error(
					"""
					The ensemble has already been activated. Attempting to auto-restart it. \
					This should never happen in production!
					"""
				)
				backend.setActivatedFlag()
				self.coordinateEnsemble()
			} else {
				// If we're initializing but locked, the process probably restarted. We're toast.
				Ensembler.logger.error("Ensembler initialized in unrecoverable state, failing!")
				throw InitializationError.ensembleAlreadyActive
			}

			// if there is only one node, its a single node ensemble which is valid configuration
			// we transition to ready state
			if self.nodeMap.count == 1 {
				Ensembler.logger.info(
					"It's a single node ensemble configuration. Mark ensemble as ready."
				)
				self.goToReady()
				self.dumpEnsembleDebugMap()
				return
			}
		} catch {
			self.ensembleFailed()
			throw error
		}
	}

	/// Triggers the workflow to rotate the keys
	public func rotateKey() throws {
		guard self.status == .ready else {
			Ensembler.logger.error(
				"""
				Oops: Attempting to rotate key from an illegal state: Expected .ready, \
				found \(self.status, privacy: .public)
				"""
			)
			throw InitializationError.invalidOperation
		}

		guard self.imTheBoss == true else {
			Ensembler.logger.error("Oops: Rotate key can be called only on leader")
			throw InitializationError.invalidOperation
		}

		if self.nodeMap.count == 1 {
			Ensembler.logger.info(
				"""
				Ensembler.rotateKey(): Returning in the .ready state because single-node ensembles \
				don't have a shared key to rotate.
				"""
			)
			return
		}
        // regenerate the plainttextuuid for each rotate operation
        plainTextUUID = UUID().uuidString
		guard let cloudAttestation = self.cloudAttestation else {
			Ensembler.logger.error("Oops: No CAF object. This should never happen!")
			throw InitializationError.unexpectedBehavior(
				"Oops: No CAF object. This should never happen!"
			)
		}

		resetEnsembleKeyDistributionStatus()

		try cloudAttestation.reKey()
		self.sharedKey = try cloudAttestation.getSymmetricKey()

		if !self.setStatus(.keyRotationAttesting) {
			throw EnsembleError.illegalStateTransition
		}
		do {
			try broadcastMessage(msg: EnsembleControlMessage.rotateKey)
		} catch {
			Ensembler.logger.error(
				"""
				Failed to broadcast rotate key: \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			ensembleFailed()
		}
	}

	public func encryptData(data: Data) throws -> Data {
		guard let sharedKey = self.sharedKey else {
			throw InitializationError.unexpectedBehavior(
				"Oops: Cannot encrypt before initializing the shared key!"
			)
		}
        
        guard let keyData = try self.backend?.getCryptoKey() else {
            throw InitializationError.unexpectedBehavior(
                "Oops: Error getting crypt key!"
            )
        }
        
        // The key we get from CIOMesh is already a derived key,
        // we are deriving a new key from the derived key here.
        let encryptKey = SymmetricKey(data: keyData)
        let derivedKey = try getDerivedKey(baseKey: encryptKey, type: .TestEncryptDecrypt)
       
		return try encrypt(data: data, sharedKey: derivedKey)
	}

	public func decryptData(data: Data) throws -> Data {
		guard let sharedKey = self.sharedKey else {
			throw InitializationError.unexpectedBehavior(
				"Oops: Cannot decrypt before initializing the shared key!"
			)
		}
        
        guard let keyData = try self.backend?.getCryptoKey() else {
            throw InitializationError.unexpectedBehavior(
                "Oops: Error getting crypt key!"
            )
        }
        // The key we get from CIOMesh is already a derived key,
        // we are deriving a new key from the derived key here.
        let decryptKey = SymmetricKey(data: keyData)
        let derivedKey = try getDerivedKey(baseKey: decryptKey, type: .TestEncryptDecrypt)
        
		return try decrypt(data: data, sharedKey: derivedKey)
	}
    
    public func getAuthCode(data: Data) throws -> Data {
        
        // we will be using the initial shared key. This key will not be updated on rotation.
        guard let sharedKey = self.initialSharedkey else {
            throw EnsembleError.internalError(error: "Cannot generate Authcode since initialSharedkey is nil")
        }
        
        // use the HKDF-SHA384 derived key
        let derivedKey = try getDerivedKey(baseKey: sharedKey, type: .TlsPsk)
        
        let authenticationCode = HMAC<SHA256>.authenticationCode(for: data, using: derivedKey)

        let authenticationData = authenticationCode.withUnsafeBytes {
			return Data($0)
        }
        
        return authenticationData
    }
    
    public func getMaxBuffersPerKey() throws -> UInt64? {
        return try self.backend?.getMaxBuffersPerKey()
    }

    public func getMaxSecondsPerKey() throws -> UInt64? {
        return try self.backend?.getMaxSecondsPerKey()
    }

	/// Deactivate the mesh in the underlying backend
	/// This function should NOOP if mesh is already deactivated
	public func deactivate() throws {
		try self.backend?.deactivate()
	}

	func drain() {
		// Serialize the update to `draining` through `fromBackendQ`, that way any subsequent
		// actions read the new value.
		Ensembler.fromBackendQ.sync {
			self.draining = true
		}
		do {
			try self.broadcastMessage(msg: EnsembleControlMessage.ensembleDraining)
		} catch {
			Self.logger.error("Oops, failed to broadcast .ensembleDraining msg: \(error)")
		}
	}
}

// MARK: - Delegates -

// Assume we are queue protected in these delegates

extension Ensembler: BackendDelegate {
	private func broadcastMessage(msg: EnsembleControlMessage) throws {
		// EnsembleControlMessage.description does not expose any private data.
		Ensembler.logger.info("Ensembler.broadcastMessage(): \(msg, privacy: .public)")
		let msgData: Data
		do {
			msgData = try JSONEncoder().encode(msg)
		} catch {
			Ensembler.logger.error(
				"""
				Failed to encode message \(String(describing: msg), privacy: .public) for \
				broadcast: \(error)
				"""
			)
			throw error
		}

		DispatchQueue.global().async {
			// No need to tell ourselves
			for node in self.ensembleConfig.nodes where node.key != self.UDID {
				do {
					try self.backend?.sendControlMessage(node: node.value.rank, message: msgData)
				} catch {
					Ensembler.logger.error(
						"""
						Failed to send ensemble control message \
						\(msg, privacy: .public) to rank: \(node.value.rank) \
						(UDID: \(node.key, privacy: .private))
						"""
					)
				}
			}
		}
	}

	private func sendMessageTo(msg: EnsembleControlMessage, destination: Int) throws {
		// EnsembleControlMessage.description does not expose any private data.
		Ensembler.logger.info(
			"""
			Ensembler.sendMessageTo(): \(msg, privacy: .public), \
			destination: \(destination, privacy: .public)
			"""
		)
		let msgData: Data
		do {
			msgData = try JSONEncoder().encode(msg)
		} catch {
			Ensembler.logger.error(
				"""
				Failed to encode message \(String(describing: msg), privacy: .public) \
				for leader: \(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			throw error
		}

		DispatchQueue.global().async {
			do {
				try self.backend?.sendControlMessage(node: destination, message: msgData)
			} catch {
				Ensembler.logger.error(
					"Failed to send message \(String(describing: msg), privacy: .public) to leader"
				)
			}
		}
	}

	private func sendBigMessageTo(msg: EnsembleControlMessage, destination: Int) throws {
		// EnsembleControlMessage.description does not expose any private data.
		Ensembler.logger.info(
			"""
			Ensembler.sendBigMessageTo(): \(msg, privacy: .public), \
			destination: \(destination, privacy: .public)
			"""
		)
		let msgData: Data
		do {
			msgData = try JSONEncoder().encode(msg)
		} catch {
			Ensembler.logger.error(
				"""
				Failed to encode message \(msg, privacy: .public) \
				for leader: \(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			throw error
		}

		DispatchQueue.global().async {
			do {
				try self.sendBigControlMessage(node: destination, message: msgData)
			} catch {
				Ensembler.logger.error("Failed to send message \(msg, privacy: .public) to leader")
			}
		}
	}

	private func checkEnsembleForReadiness() -> Bool {
		for node in self.nodeMap.values {
			if node.found != true {
				Ensembler.logger.info(
					"""
					Ensembler.checkEnsembleForReadiness(): \
					node \(node.rank, privacy: .public) NOT yet found: \
					returning `false`
					"""
				)
				return false
			}
		}
		Ensembler.logger.info(
			"Ensembler.checkEnsembleForReadiness(): found all nodes: returning `true`"
		)
		return true
	}

	private func resetEnsembleKeyDistributionStatus() {
		for node in self.nodeMap.values {
			self.nodeMap[node.UDID]?.keyShared = false
		}
		Ensembler.logger.info("Done resetting `keyShared` to `false` for all nodeMap entries.")
	}

	private func checkEnsembleForKeyDistribution() -> Bool {
		for node in self.nodeMap.values {
			// skip checking for leader, since leader has the sharedkey already.
			if node.rank == 0 {
				continue
			}
			if node.keyShared != true {
				Ensembler.logger.info(
					"""
					Ensembler.checkEnsembleForKeyDistribution(): \
					node \(node.rank, privacy: .public): keyShared is `false`: \
					returning `false`
					"""
				)
				return false
			}
		}
		Ensembler.logger.info(
			"""
			Ensembler.checkEnsembleForKeyDistribution(): \
			keyShared is `true` for all nodes: returning `true`
			"""
		)
		return true
	}

	private func setCryptoKey() {
		do {
			guard let sharedKey = self.sharedKey else {
				Ensembler.logger.error(
					"""
					Oops: setCryptoKey() called before self.sharedKey was initialized.
					"""
				)
				throw InitializationError.unexpectedBehavior(
					"""
					Oops: setCryptoKey() called before self.sharedKey was initialized.
					"""
				)
			}

            // store the key in initialSharedkey for use in TLS PSK Options.
            // this initialSharedkey will not be updated on rotation.
            if self.initialSharedkey == nil {
                self.initialSharedkey = self.sharedKey
            }
            
            // use the HKDF-SHA384 derived key
            let derivedKey = try getDerivedKey(baseKey: sharedKey, type: .MeshEncryption)
            
			var keyData = derivedKey.withUnsafeBytes {
				return Data(Array($0))
			}

            defer {
                keyData.withUnsafeMutableBytes { keyPtr in
                    guard let baseAddress = keyPtr.baseAddress.self else {
                        Ensembler.logger.error(
                            """
                            Failed to clear key data: Failed to get base address of key data
                            """
                        )
                        return
                    }
                    
                    guard 0 == memset_s(baseAddress, keyPtr.count, 0, keyPtr.count) else {
                        Ensembler.logger.error(
                            """
                            Failed to clear key data: memset_s failed
                            """
                        )
                        return
                    }
                }
                Ensembler.logger.info(
                    """
                    Ensembler.setCryptoKey(): \
                    Succcessfully cleared the keydata
                    """
                )
            }
            
			Ensembler.logger.info("Ensembler.setCryptoKey(): Call backend.setCryptoKey()")
			try self.backend?.setCryptoKey(key: keyData, flags: 0)
			Ensembler.logger.info(
				"Ensembler.setCryptoKey(): Successfully set crypto key in CIOMesh"
			)
		} catch {
			// This is technically harmless as only our status matters
			Ensembler.logger.error(
				"""
				Failed to set crypto key in CIOMesh: \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			ensembleFailed()
		}
	}

	// handle the pairing request from follower
	// 1. Get the follower attestation
	// 2. call into cloud attestation to get the pairing data
	// 3. send back the leader attesation data and pairing data to the follower to complete pairing
	// 4. send a plaintext, which we will validate when we get the announcement from follower that
	// it got its key
	private func handlePairNode(udid: String, attestation: String) {
		let status = self.status
		self.attestationAsyncQueue.async {
			guard status == .attesting || status == .keyRotationAttesting, self.nodeMap[udid]?.found == true else {
				Ensembler.logger.error(
					"Ensembler.handlePairNode(): Handling extraneous message from UDID \(udid)"
				)
				self.ensembleFailed()
				return
			}

			guard let destination = self.nodeMap[udid]?.rank,
			      destination != 0 else {
				Ensembler.logger.error(
					"Ensembler.handlePairNode(): Cannot find rank for node \(udid), or rank is 0. "
				)
				self.ensembleFailed()
				return
			}

			guard let cloudAttestation = self.cloudAttestation else {
				Ensembler.logger.error("Ensembler.handlePairNode(): Oops, CAF isn't initialized!")
				self.ensembleFailed()
				return
			}

			do {
				Ensembler.logger.info(
					"""
					Ensembler.handlePairNode(): \
					pairing with rank \(destination, privacy: .public) (UDID: \(udid))
					"""
				)
				let followerAttestation = try AttestationBundle(jsonString: attestation)
				Ensembler.logger.info("Ensembler.handlePairNode(): getting leader attesation bundle")
				let leaderAttestation = try await cloudAttestation.getAttestationBundle()
				Ensembler.logger.info("Ensembler.handlePairNode(): getting leader pairing data")
				let pairingData = try await cloudAttestation.getPairingData(
					followerUDID: udid,
					followerAttestationBundle: followerAttestation
				)

				Ensembler.logger.info(
					"""
					Ensembler.handlePairNode(): \
					sending .completePairing message to the follower \(udid)
					"""
				)
				try self.sendBigMessageTo(
					msg: EnsembleControlMessage
						.completePairing(
							leaderAttestation: leaderAttestation.jsonString(),
							pairingData: pairingData,
							plainText: self.plainTextUUID
						),
					destination: destination
				)
				Ensembler.logger.info(
					"""
					Ensembler.handlePairNode(): \
					sent .completePairing message to rank \(destination, privacy: .public) \
					(UDID: \(udid))
					"""
				)
			} catch {
				Ensembler.logger.error(
					"""
					Error pairing with rank \(destination, privacy: .public) (UDID: \(udid)): \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
				self.ensembleFailed()
			}
		}
	}

	private func decrypt(data: Data, sharedKey: SymmetricKey) throws -> Data {
		let box = try CryptoKit.AES.GCM.SealedBox(combined: data)

		let decrypted = try CryptoKit.AES.GCM.open(box, using: sharedKey)
		return decrypted
	}

	// handle the shared key announcement ( i.e follower announce that it got its key )
	// 1. we validate the plaintext encrypted by follower can be decrypted by the shared key.
	// 2. when all followers has announced that it got its key and we can decrypt the message
	//    we know that all nodes has got its key.
	private func handleAnnounceSharedKey(udid: String, encryptedMsg: Data) {
		Ensembler.logger.info("Ensembler.handleAnnounceSharedKey(udid: \(udid))")
		let status = self.status
		guard status == .attesting || status == .keyRotationAttesting, self.nodeMap[udid]?.found == true else {
			Ensembler.logger.error(
				"""
				Ensembler.handleAnnounceSharedKey(): \
				Oops: Received extraneous message from UDID \(udid)
				"""
			)
			self.ensembleFailed()
			return
		}

		guard let destination = self.nodeMap[udid]?.rank,
		      destination != 0 else {
			Ensembler.logger.error("Cannot find rank for node \(udid), or rank is 0. ")
			ensembleFailed()
			return
		}

		guard let sharedKey = self.sharedKey else {
			Ensembler.logger.error(
				"Ensembler.handleAnnounceSharedKey(): The shared key isn't initialized!"
			)
			ensembleFailed()
			return
		}

		self.nodeMap[udid]?.keyShared = true
		do {
            // use the HKDF-SHA384 derived key
            let derivedKey = try getDerivedKey(baseKey: sharedKey, type: .KeyConfirmation(udid))
            
			// verify if leader can decrypt the message sent my follower and if the plaintext is
			// same as what leader sent
			let decryptedData = try decrypt(data: encryptedMsg, sharedKey: derivedKey)
            
			let decryptedText = String(decoding: decryptedData, as: UTF8.self)
			if self.plainTextUUID != decryptedText {
				Ensembler.logger.error(
					"""
					Ensembler.handleAnnounceSharedKey(): \
					The plaintext and the decrypted text does not match.
					"""
				)
				ensembleFailed()
				return
			}

			Ensembler.logger.info(
				"""
				Ensembler.handleAnnounceSharedKey(): \
				Send .acceptSharedKey message to \
				rank \(destination, privacy: .public) (UDID: \(udid)).
				"""
			)
			try self.sendMessageTo(
				msg: EnsembleControlMessage.acceptSharedKey,
				destination: destination
			)
			Ensembler.logger.info(
				"""
				Ensembler.handleAnnounceSharedKey(): \
				Successfully sent .acceptSharedKey message to \
				rank \(destination, privacy: .public) (UDID: \(udid)).
				"""
			)
		} catch {
			Ensembler.logger.error(
				"""
				Ensembler.handleAnnounceSharedKey(): Error sending accept sharedKey message to \
				rank \(destination, privacy: .public) (UDID: \(udid)): \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
			ensembleFailed()
		}

		if self.checkEnsembleForKeyDistribution() == true {
			do {
				Ensembler.logger.info(
					"""
					Ensembler.handleAnnounceSharedKey(): \
					Send .ensembleSecureComplete message to \
					rank \(destination, privacy: .public) (UDID: \(udid)).
					"""
				)
				try self.broadcastMessage(msg: EnsembleControlMessage.ensembleSecureComplete)
				Ensembler.logger.info(
					"""
					Ensembler.handleAnnounceSharedKey(): \
					Successfully sent .ensembleSecureComplete message to \
					rank \(destination, privacy: .public) (UDID: \(udid)).
					"""
				)
			} catch {
				Ensembler.logger.error(
					"""
					Ensembler.handleAnnounceSharedKey(): \
					Failed to broadcast ensemble secure complete status: \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
				ensembleFailed()
			}

			// Now that we know the key is present in all nodes, we can set it in CIOMesh driver.
			Ensembler.logger.info("Ensembler.handleAnnounceSharedKey(): Call setCryptoKey().")
			self.setCryptoKey()
			Ensembler.logger.info("Ensembler.handleAnnounceSharedKey(): setCryptoKey() returned.")
			self.goToReady()
		}
	}

	private func handleNewNode(udid: String) {
		guard self.status != .ready, self.nodeMap[udid]?.found != true else {
			Ensembler.logger.error(
				"Ensembler.handleNewNode(): Handling extraneous message from \(udid)"
			)
			return
		}

		guard let destination = self.nodeMap[udid]?.rank else {
			Ensembler.logger.error("Ensembler.handleNewNode(): Cannot find rank for node \(udid)")
			return
		}

		Ensembler.logger.info(
			"Ensembler.handleNewNode(): rank: \(destination, privacy: .public) (UDID: \(udid))"
		)

		self.nodeMap[udid]?.found = true

		if destination != self.currentNodeConfig.rank {
			do {
				Ensembler.logger.info(
					"""
					Ensembler.handleNewNode(): send .acceptNode message \
					to rank \(destination) (UDID: \(udid))
					"""
				)
				try self.sendMessageTo(
					msg: EnsembleControlMessage.acceptNode,
					destination: destination
				)
				Ensembler.logger.info(
					"""
					Ensembler.handleNewNode(): successfully sent .acceptNode message \
					to rank \(destination) (UDID: \(udid))
					"""
				)
			} catch {
				Ensembler.logger.error(
					"""
					Ensembler.handleNewNode(): Cannot send acceptance message \
					to rank \(destination) (UDID: \(udid)): \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
			}
		}

		// If the ensemble is complete, let's tell everyone, and initiate the
		// flow to get the key from cloudattestation. Once follower gets ensembleComplete
		// they will transition to .pairing status and trigger the flow to get
		// the shared key.
		if self.checkEnsembleForReadiness() == true {
			if !self.setStatus(.attesting) {
				return
			}
			self.everyoneFound = true
			self.dumpEnsembleDebugMap()
			do {
				Ensembler.logger.info(
					"Ensembler.handleNewNode(): broadcast .ensembleComplete message"
				)
				try self.broadcastMessage(
					msg: EnsembleControlMessage.ensembleComplete(slots: self.slots)
				)
				Ensembler.logger.info(
					"Ensembler.handleNewNode(): successfully broadcasted .ensembleComplete message"
				)
			} catch {
				// This is technically harmless as only our status matters
				Ensembler.logger.error(
					"""
					Ensembler.handleNewNode(): Failed to broadcast node complete status: \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
			}
		}
	}

	public func getHealth() -> EnsembleHealth {
		let status = self.status
		switch status {
		// `.failed` states. These clearly indicate that the ensemble is unhealthy.
		case .failed, .failedActivationChecks, .failedWhileDraining:
			return EnsembleHealth(healthState: HealthState.unhealthy, internalState: status)

		// `.healthy` states. These clearly indicate that the ensemble is healthy.
		case .ready:
			return EnsembleHealth(healthState: HealthState.healthy, internalState: status)

		// These states can be entered in two cases:
		//   1. Installing the initial key. Here, the ensemble is still starting, and we should
		//   return .initializing.
		//   2. During a key rotation. Here, the ensemble has already been established as healthy.
		//   Thus, we should return .healthy.
		default:
			if !self.doneInitializing {
				return EnsembleHealth(healthState: HealthState.initializing, internalState: status)
			} else {
				return EnsembleHealth(healthState: HealthState.healthy, internalState: status)
			}
		}
	}

	public func checkConnectivity() throws -> [String] {
		if self.nodeMap.count != 8 {
			throw InitializationError.unexpectedBehavior(
				"""
				Oops: Cable diagnostics only supported for an 8-node ensemble. \
				Current ensemble is \(self.nodeMap.count) nodes.
				"""
			)
		}

		// When all nodes have established a connection with each other then the connection is
		// assumed to be good.
		guard !self.everyoneFound else {
			return []
		}

		guard let backend = self.backend else {
			throw InitializationError.invalidBackend
		}

		let cio = try backend.getCIOCableState()
		var cableStatus: [Bool] = .init(repeating: false, count: 8)
		var expectedPartners: [Int] = .init(repeating: -1, count: 8)
		var actualPartners: [Int] = .init(repeating: -1, count: 8)

		for (i, c) in cio.enumerated() {
			let cableConnectedObj = c["cableConnected"]
			guard let cableConnected = cableConnectedObj as? Int else {
				Ensembler.logger.error(
					"""
					Failed on getCIOCableState() CIO mesh API: Unable to parse entry i=\(i) -> \
					c["cableConnected"]=\(String(describing: cableConnectedObj)) as Int.
					"""
				)
				throw InitializationError.unexpectedBehavior(
					"Failed on getCIOCableState(): Unable to parse c[\"cableConnected\"] as Int."
				)
			}

			let expectedPartnerObj = c["expectedPartnerHardwareNode"]
			guard let expectedPartner = expectedPartnerObj as? Int else {
				Ensembler.logger.error(
					"""
					Failed on getCIOCableState() CIO mesh API: Unable to parse entry i=\(i) -> \
					c["expectedPartnerHardwareNode"]=\(String(describing: expectedPartnerObj)) \
					as Int.
					"""
				)
				throw InitializationError.unexpectedBehavior(
					"""
					Failed on getCIOCableState(): Unable to parse \
					c[\"expectedPartnerHardwareNode\"] as Int.
					"""
				)
			}

			let actualPartnerObj = c["actualPartnerHardwareNode"]
			guard let actualPartner = actualPartnerObj as? Int else {
				Ensembler.logger.error(
					"""
					Failed on getCIOCableState() CIO mesh API: Unable to parse entry i=\(i) -> \
					c["actualPartnerHardwareNode"]=\(String(describing: actualPartnerObj)) \
					as Int.
					"""
				)
				throw InitializationError.unexpectedBehavior(
					"""
					Failed on getCIOCableState(): Unable to parse \
					c[\"actualPartnerHardwareNode\"] as Int.
					"""
				)
			}

			cableStatus[i] = cableConnected == 1
			expectedPartners[i] = expectedPartner
			actualPartners[i] = actualPartner
		}

		var diagnostics: [String] = []

		if !cableStatus[0] || !cableStatus[2] {
			diagnostics.append("PortB Cable not functioning")
		}
		if !cableStatus[1] || !cableStatus[3] {
			diagnostics.append("PortA Cable not functioning")
		}
		if !cableStatus[4] || !cableStatus[5] || !cableStatus[6] || !cableStatus[7] {
			diagnostics.append("Internal Cable not functioning")
		}

		if expectedPartners[0] != actualPartners[0] ||
			expectedPartners[2] != actualPartners[2] {
			diagnostics.append("PortB Cable not plugged correctly")
		}
		if expectedPartners[1] != actualPartners[1] ||
			expectedPartners[3] != actualPartners[3] {
			diagnostics.append("PortA Cable not plugged correctly")
		}
		if expectedPartners[4] != actualPartners[4] ||
			expectedPartners[5] != actualPartners[5] ||
			expectedPartners[6] != actualPartners[6] ||
			expectedPartners[7] != actualPartners[7] {
			diagnostics.append("Internal Cable not plugged correctly")
		}

		return diagnostics
	}

	/// Simple public function that sends a fixed test message to any node
	public func sendTestMessage(destination: Int) throws {
		try self.sendMessageTo(msg: EnsembleControlMessage.testMessage, destination: destination)
	}

	func channelChangeInternal(node: Int, chassis: String, channelIndex: Int, connected: Bool) {
		guard self.router != nil else {
			Ensembler.logger.error("Received channelChange event before router was initialized")
			return
		}
		if connected == false {
			Ensembler.logger.error(
				"""
				Ensembler.channelChange(): \
				Received disconnect during channelChange event \
				from node \(node, privacy: .public): \
				chassis: \(chassis) \
				channelIndex: \(channelIndex, privacy: .public)
				"""
			)
		} else {
			Ensembler.logger.info(
				"""
				Ensembler.channelChange(): \
				Received channelChange event: \
				from node \(node, privacy: .public) \
				chassis: \(chassis) \
				channelIndex: \(channelIndex, privacy: .public) \
				connected: \(connected, privacy: .public)
				"""
			)
		}
		self.router?.channelChange(
			channelIndex: channelIndex,
			node: node,
			chassis: chassis,
			connected: connected
		)
		Ensembler.logger.info("Ensembler.channelChange(): router.channelChange() done")
	}

	func channelChange(node: Int, chassis: String, channelIndex: Int, connected: Bool) {
		let dispatchGroup = DispatchGroup()
		dispatchGroup.enter()
		Ensembler.fromBackendQ.async {
			defer {
				dispatchGroup.leave()
			}
			self.channelChangeInternal(node: node,
									   chassis: chassis,
									   channelIndex: channelIndex,
									   connected: connected)
		}
		dispatchGroup.wait()
	}

	func connectionChangeInternal(
		direction: BackendConnectionDirection,
		node: Int,
		channelIndex: Int,
		connected: Bool
	) {
		guard self.router != nil else {
			Ensembler.logger.error("Received connectionChange event before router was initialized")
			return
		}
		if connected == false {
			Ensembler.logger.error(
				"""
				Ensembler.connectionChange(): \
				Received disconnect during connectionChange event \
				from node \(node, privacy: .public):
				direction: \(String(describing: direction), privacy: .public)
				channelIndex: \(channelIndex, privacy: .public)
				"""
			)
		} else {
			Ensembler.logger.info(
				"""
				Ensembler.connectionChange(): \
				Received connectionChange event: \
				from node \(node, privacy: .public) \
				channelIndex: \(channelIndex, privacy: .public) \
				connected: \(connected, privacy: .public) \
				direction: \(direction, privacy: .public)
				"""
			)
		}
		self.router?.connectionChange(
			direction: direction,
			channelIndex: channelIndex,
			node: node,
			connected: connected
		)
		Ensembler.logger.info("Ensembler.connectionChange(): router.connectionChange() done")
	}

	func connectionChange(
		direction: BackendConnectionDirection,
		node: Int,
		channelIndex: Int,
		connected: Bool
	) {
		let dispatchGroup = DispatchGroup()
		dispatchGroup.enter()
		Ensembler.fromBackendQ.async {
			defer {
				dispatchGroup.leave()
			}
			self.connectionChangeInternal(direction: direction,
										  node: node,
										  channelIndex: channelIndex,
										  connected: connected)
		}
		dispatchGroup.wait()
	}

	private func incomingMessageForLeader(node: Int,
										  controlMsg: EnsembleControlMessage,
										  sender: [String: NodeConfiguration].Element) {
		switch(controlMsg) {
		case .announceNode(let slot):
			self.slots[sender.value.rank] = slot
			self.handleNewNode(udid: sender.key)
		case .pairNode(let attestationdata):
			self.handlePairNode(udid: sender.key, attestation: attestationdata)
		case .announceSharedKey(let encryptedMsg):
			self.handleAnnounceSharedKey(udid: sender.key, encryptedMsg: encryptedMsg)
		case .ensembleFailed:
			ensembleFailed()
		default:
			Self.logger.error(
				"""
				Ensembler.incomingMessageForLeader(): \
				Received unexpected command message \(String(describing: controlMsg))
				"""
			)
			ensembleFailed()
		}
	}

	private func incomingMessageForFollower(node: Int,
											controlMsg: EnsembleControlMessage,
											sender: [String: NodeConfiguration].Element) throws {
		switch(controlMsg) {
		case .acceptNode:
			if !self.setStatus(.accepted) {
				throw EnsembleError.illegalStateTransition
			}
		case .ensembleComplete(let slots):
			if !self.setStatus(.pairing) {throw EnsembleError.illegalStateTransition }
			self.everyoneFound = true
			self.slots = slots
			self.dumpEnsembleDebugMap()
			pairWithLeader()
		case .rotateKey:
			if !self.setStatus(.keyRotationPairing) {
				throw EnsembleError.illegalStateTransition
			}
			pairWithLeader()
		case .completePairing(let attestation, let pairingData, let message):
			handleCompletePairing(udid: sender.key,
								  attestation: attestation,
								  pairingData: pairingData,
								  message: message)
		case .acceptSharedKey:
			if !self.setStatus(.keyAccepted) {
				throw EnsembleError.illegalStateTransition
			}
		case .ensembleSecureComplete:
			handleEnsembleSecureComplete()
		case .ensembleFailed:
			// If we get a failure message as a follower, then leader is already broadcasting the
			// failure to the rest of the ensemble.
			self.goToFailed()
		default:
			Self.logger.error(
				"""
				Ensembler.incomingMessageForFollower(): \
				Received unexpected command message \(String(describing: controlMsg))
				"""
			)
			ensembleFailed()
		}
	}

	private func incomingMessageInternal(node: Int, message: Data) {
		let controlMsg: EnsembleControlMessage
		do {
			controlMsg = try JSONDecoder().decode(EnsembleControlMessage.self, from: message)
		} catch {
			Self.logger.error(
				"Ensembler.incomingMessageInternal(): Control message decoding failed: \(error)"
			)
			return
		}

		guard let sender = self.ensembleConfig.nodes.first(where: { $0.value.rank == node }) else {
			Self.logger.error(
				"Ensembler.incomingMessageInternal(): Cannot find node \(node) in configuration"
			)
			return
		}

		Self.logger.info(
			"""
			Ensembler.incomingMessageInternal(): Received \(controlMsg, privacy: .public) message \
			from node \(sender.key) with rank \(node, privacy: .public)
			"""
		)

		switch(controlMsg) {
		// Generic operations
		case .ForwardMessage(let forward):
			self.router?.forwardMessage(forward)
		case .testMessage:
			Ensembler.logger.info(
				"""
				Ensembler.incomingMessage(): Received test message \
				from node \(sender.key) with rank \(node, privacy: .public)
				"""
			)
		case .ensembleDraining:
			self.draining = true
		case .bigMessageStart(_):
			self.dataMap[sender.key] = Data()
		case .bigMessageChunk(let data):
			self.dataMap[sender.key]?.append(data)
		case .bigMessageEnd:
			guard let finaldata = dataMap[sender.key] else {
				Ensembler.logger.error(
					"""
					Ensembler.incomingMessageInternal(): \
					Error assembling chunks of data from big message
					"""
				)
				ensembleFailed()
				return
			}
			self.incomingMessageInternal(node: node, message: finaldata)
		default:
			// Non-generic operations
			do {
				if self.imTheBoss {
					self.incomingMessageForLeader(node: node, controlMsg: controlMsg, sender: sender)
				} else {
					try self.incomingMessageForFollower(node: node,
														controlMsg: controlMsg,
														sender: sender)
				}
			} catch {
				Self.logger.error(
					"""
					Ensembler.incomingMessageInternal(): \
					Failed to handle command message \(String(describing: controlMsg)): \(error)
					"""
				)
			}
		}

		Self.logger.info(
			"""
			Ensembler.incomingMessageInternal(): \
			Finished handling message \(controlMsg, privacy: .public) \
			from node \(sender.key) with rank \(node, privacy: .public)
			"""
		)
	}

	func incomingMessage(node: Int, message: Data) {
		let dispatchGroup = DispatchGroup()
		dispatchGroup.enter()
		Ensembler.fromBackendQ.async {
			defer {
				dispatchGroup.leave()
			}
			self.incomingMessageInternal(node: node, message: message)
		}
		dispatchGroup.wait()
	}
}

extension Ensembler: RouterDelegate {
	private func broadcastOurFailure() {
		Ensembler.logger.info(
			"Ensembler.broadcastOurFailure(): rank \(self.currentNodeConfig.rank)"
		)
		let msg = EnsembleControlMessage.ensembleFailed
		do {
			if self.imTheBoss {
				try self.broadcastMessage(msg: msg)
			} else {
				try self.sendMessageTo(msg: msg, destination: 0)
			}
		} catch {
			Ensembler.logger.error("Failed to broadcast failure: \(error)")
		}
	}

	// send pairWithleader control message to leader.
	// pair with the leader to get the leader attestation bundle and pairting data
	// so the follower can get the sharedkey from cloud attestation.
	private func pairWithLeader() {
		Ensembler.logger.info(
			"Ensembler.pairWithLeader(): rank \(self.currentNodeConfig.rank)"
		)

		guard let cloudAttestation = self.cloudAttestation else {
			Ensembler.logger.error("Ensembler.pairWithLeader(): Oops, CAF isn't initialized!")
			self.ensembleFailed()
			return
		}

		self.attestationAsyncQueue.async {
			var msg: Data
			do {
				let attestationBundle = try await cloudAttestation.getAttestationBundle()
				msg = try JSONEncoder().encode(EnsembleControlMessage.pairNode(
					followerAttestation: attestationBundle.jsonString()))
			} catch {
				Ensembler.logger.error(
					"""
					Ensembler.pairWithLeader(): Failed to encode pair with leader message, \
					ensemble cannot be assembled: \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
				self.ensembleFailed()
				return
			}

			do {
				Ensembler.logger.info(
					"Ensembler.pairWithLeader(): Pairing with leader to get shared key"
				)
				try self.sendBigControlMessage(node: 0, message: msg)
			} catch {
				Ensembler.logger.error(
					"""
					"Ensembler.pairWithLeader(): \
					Pairing with leader control message failed, will retry: \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
				self.ensembleFailed()
			}

			Ensembler.logger.info("Ensembler.pairWithLeader(): Pairing with leader complete")
		}
	}

	private func encrypt(data: Data, sharedKey: SymmetricKey) throws -> Data {
		Ensembler.logger.info("Encrypting the plaintext data")
		let box = try CryptoKit.AES.GCM.seal(data, using: sharedKey)
		guard let encryptedData = box.combined else {
			Ensembler.logger.error("Error encrypting data")
			throw EnsembleError.internalError(error: "Error encrypting data")
		}

		return encryptedData
	}

	// process completePairing message from leader.
	// 1. Get the attestation data and pairing data from leader
	// 2. Get the shared key from cloud attestation
	// 3. save the shared key
	// 4. send the announceSharedKey to leader to indicate that the follower has got its key
	private func handleCompletePairing(
		udid: String,
		attestation: String,
		pairingData: EnsembleChannelSecurity.PairingData?,
		message: String
	) {
		self.attestationAsyncQueue.async {
			guard let _ = self.nodeMap[udid]?.rank else {
				Ensembler.logger.error(
					"Ensembler.handleCompletePairing(): Cannot find rank for node \(udid)"
				)
				self.ensembleFailed()
				return
			}

			guard let cloudAttestation = self.cloudAttestation else {
				Ensembler.logger.error(
					"Ensembler.handleCompletePairing(): Oops, CAF isn't initialized!"
				)
				self.ensembleFailed()
				return
			}

			do {
				Ensembler.logger.info(
					"""
					Ensembler.handleCompletePairing(): Handling completePairing message from leader
					"""
				)
				let leaderAttestation = try AttestationBundle(jsonString: attestation)

				let sharedKey = try await cloudAttestation.completePairing(
					leaderPairingData: pairingData,
					leaderAttestationBundle: leaderAttestation
				)
				self.sharedKey = sharedKey

				Ensembler.logger.info("Ensembler.handleCompletePairing(): Obtained shared key")
                
                Ensembler.logger.info(
                    """
                    Ensembler.handleCompletePairing(): Validating if the message is a UUID
                    """
                )
                
                guard let uuid = UUID(uuidString: message) else {
                    Ensembler.logger.error("The message expected from leader is not of UUID format.")
                    self.ensembleFailed()
                    return
                }
                
				guard let data = message.data(using: .utf8) else {
					Ensembler.logger.error("Error converting message")
					self.ensembleFailed()
					return
				}

                // use the HKDF-SHA384 derived key
                let derivedKey = try self.getDerivedKey(baseKey: sharedKey, type: .KeyConfirmation(self.UDID))
                                                    
				let encryptedMsg = try self.encrypt(data: data, sharedKey: derivedKey)
				Ensembler.logger.info(
					"""
					Ensembler.handleCompletePairing(): sending .announceSharedKey message \
					to leader with encrypted message
					"""
				)

				let msg = try JSONEncoder().encode(
					EnsembleControlMessage.announceSharedKey(encryptedText: encryptedMsg))
				try self.sendBigControlMessage(node: 0, message: msg)

				Ensembler.logger.info(
					"""
					Ensembler.handleCompletePairing(): \
					pairing complete, key obtained and announceSharedKey message sent to leader \
					with encrypted message
					"""
				)

				if !self.setStatus(.pairingComplete) {
					return
				}
			} catch {
				Ensembler.logger.error(
					"""
					Ensembler.handleCompletePairing(): \
					completePairing process not complete: \
					\(String(reportableError: error), privacy: .public) (\(error))
					"""
				)
				self.ensembleFailed()
			}
		}
	}

	private func handleEnsembleSecureComplete() {
		// Now that we know the key is present in all nodes, we can set it in CIOMesh driver.
		self.setCryptoKey()
		self.goToReady()
	}

	// send announce control message to leader.
	private func yammerAtLeader() {
		var msg: Data
		do {
			let slot = self.slots[self.currentNodeConfig.rank]
			msg = try JSONEncoder().encode(EnsembleControlMessage.announceNode(slot: slot))
		} catch {
			Ensembler.logger.error(
				"Failed to encode node announcement message, ensemble can not be assembled"
			)
			self.ensembleFailed()
			return
		}
		DispatchQueue.global().async {
			repeat {
				do {
					Ensembler.logger.info(
						"Ensembler.yammerAtLeader(): Announcing presence to leader"
					)
					try self.backend?.sendControlMessage(node: 0, message: msg)
				} catch {
					Ensembler.logger.error(
						"""
						Ensembler.yammerAtLeader(): \
						Failed to send node announcement control message, will retry: \
						\(String(reportableError: error), privacy: .public) (\(error))
						"""
					)
					// Wait a little longer because something bad happened
					Thread.sleep(forTimeInterval: 9.0)
				}
				// Delay 1 second between messages for sanity I guess
				Thread.sleep(forTimeInterval: 1.0)
			} while self.status == .coordinating
		}
	}

	private func divideAndSend(node: Int, message: Data) throws {
		Ensembler.logger.info(
			"""
			Ensembler.divideAndSend(): Dividing and sending message \
			of size \(message.count, privacy: .public) \
			to rank \(node, privacy: .public)
			"""
		)
		let preamble = try JSONEncoder()
			.encode(EnsembleControlMessage.bigMessageStart(size: message.count))
		try self.backend?.sendControlMessage(node: node, message: preamble)

		for i in stride(from: 0, to: message.count, by: self.maxControlMsgSize) {
			let start = i
			let end = (i + self.maxControlMsgSize)
			if end <= message.count {
				Ensembler.logger.info(
					"""
					Ensembler.divideAndSend(): \
					sending data chunk start (end <= message.count): \
					start=\(start, privacy: .public), \
					end=\(end, privacy: .public), \
					message count= \(message.count, privacy: .public)
					"""
				)
				let chunkData = message.subdata(in: Range(start ... end - 1))
				let chunk = try JSONEncoder()
					.encode(EnsembleControlMessage.bigMessageChunk(data: chunkData))
				try self.backend?.sendControlMessage(node: node, message: chunk)
			} else {
				let end = start + (message.count % self.maxControlMsgSize)
				Ensembler.logger.info(
					"""
					Ensembler.divideAndSend(): \
					sending data chunk start (end > message.count): \
					start=\(start, privacy: .public), \
					end=\(end, privacy: .public), \
					message count= \(message.count, privacy: .public)
					"""
				)
				let chunkData = message.subdata(in: Range(start ... end - 1))
				let chunk = try JSONEncoder()
					.encode(EnsembleControlMessage.bigMessageChunk(data: chunkData))
				try self.backend?.sendControlMessage(node: node, message: chunk)
			}
		}
		Ensembler.logger.info("Ensembler.divideAndSend(): sending bigMessageEnd")
		let end = try JSONEncoder().encode(EnsembleControlMessage.bigMessageEnd)
		try self.backend?.sendControlMessage(node: node, message: end)
	}

	// helper method to break and send big message
	func sendBigControlMessage(node: Int, message: Data) throws {
		Ensembler.logger.info(
			"""
			Ensembler.sendBigControlMessage(node: \(node, privacy: .public), \
			message: [\(message.count, privacy: .public)])
			"""
		)
		if message.count > self.maxControlMsgSize {
			Ensembler.logger.info("Ensembler.sendBigControlMessage(): call divideAndSend()")
			try self.divideAndSend(node: node, message: message)
		} else {
			Ensembler.logger.info(
				"Ensembler.sendBigControlMessage(): call backend.sendControlMessage()"
			)
			try self.backend?.sendControlMessage(node: node, message: message)
		}
		Ensembler.logger.info(
			"Ensembler.sendBigControlMessage(): Done sending big control message.)"
		)
	}

	private func goToReady() {
		self.doneInitializing = true
		if !self.setStatus(.ready) {
			return
		}
	}

	private func goToFailed() {
		if !self.draining {
			if !self.setStatus(.failed) {
				return
			}
		} else {
			if !self.setStatus(.failedWhileDraining) {
				return
			}
		}
	}

	// called when there is any failure during activation and readying process.
	func ensembleFailed() {
		let status = self.status
		guard !status.inFailedState() else {
			Ensembler.logger.info(
				"""
				Ensembler.ensembleFailed(): \
				We already acknowledged failure so no need to do it again.
				"""
			)
			return
		}
		Ensembler.logger.error("Ensembler.ensembleFailed(): Marking ensemble as failed")
		self.goToFailed()
		self.broadcastOurFailure()
		do {
			// Attempt a deactivation
			try self.backend?.deactivate()
		} catch {
			Ensembler.logger.error(
				"""
				Ensembler.ensembleFailed(): \
				We somehow failed at failure itself and failed to deactivate backend \
				\(String(reportableError: error), privacy: .public) (\(error))
				"""
			)
		}
		self.dumpEnsembleDebugMap()
	}

	private func coordinateEnsemble() {
		// It is time to talk to our friends
		Ensembler.logger.info(
			"Ensembler.coordinateEnsemble(): The node is ready and is now coordinating"
		)
		if !self.setStatus(.coordinating) {
			return
		}
		guard self.imTheBoss == false else {
			// We handle ourselves like any other node in the group
			self.handleNewNode(udid: self.UDID)
			return
		}
		// We are a follower; whine until we are allowed into the ensemble
		self.yammerAtLeader()
	}

	func ensembleReady() {
		do {
			try self.backend?.lock()
		} catch {
			Ensembler.logger.error("Failed to lock ensemble, failing.")
			self.ensembleFailed()
			return
		}
		self.coordinateEnsemble()
	}
    
    private func getDerivedKey(baseKey: SymmetricKey, type: DerivedKeyType) throws -> SymmetricKey {
        return try deriveKey(baseKey: baseKey, salt: "\(type)-salt", info: "\(type)-info")
    }
    
    private func deriveKey(baseKey: SymmetricKey, salt: String, info: String ) throws -> SymmetricKey {
        
        guard let salt = salt.data(using: .utf8) else {
            Ensembler.logger.error("Error forming salt from \(salt).")
            throw  InitializationError.keyDerivationError("Error forming salt from \(salt).")
        }
        
        guard let info = info.data(using: .utf8) else {
            Ensembler.logger.error("Error forming info from \(info).")
            throw  InitializationError.keyDerivationError("Error forming info from \(info).")
        }
        
        let hkdfResultKey = HKDF<SHA384>.deriveKey(inputKeyMaterial: baseKey, salt: salt, info: info,  outputByteCount: 32)
        
        return hkdfResultKey
    }
}
