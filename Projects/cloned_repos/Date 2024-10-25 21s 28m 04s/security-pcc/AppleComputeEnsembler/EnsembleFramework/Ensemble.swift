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

//
//  Ensemble.swift
//  Ensemble
//
//  Created by Alex T Newman on 12/21/23.
//

import CryptoKit
import Foundation
import OSLog
import XPC
import Network
import Security



@_implementationOnly import MobileGestaltPrivate // For fetching UDID

// MARK: - Ensemble Protocol -

// MARK: - Constants

/// Mach service name exposed by ensembled
@_spi(Daemon)
public let kEnsemblerServiceName = "com.apple.cloudos.AppleComputeEnsembler.service"

@available(macOS 15.0, iOS 18.0, *)
public let kEnsembleStatusReadyEventName = "com.apple.cloudos.AppleComputeEnsembler.Notification.ready"

@available(macOS 15.0, iOS 18.0, *)
public let kEnsembleStatusFailedEventName = "com.apple.cloudos.AppleComputeEnsembler.Notification.failed"

// MARK: - Errors

@available(macOS 15.0, iOS 18.0, *)
public enum EnsembleError: Error {
	/// XPC Client failed to connect
	case clientConnectionFailure
	/// Failed to construct/send/receive an XPC message
	case messagingFailure
	/// Command received but failed to complete/execute
	case commandFailure(error: String?)
	/// Can not fetch the EnsembleStatus (failure within the daemon)
	case statusNotFound
	/// Shared key is not set yet.
	case shareKeyNotFound
	/// Cannot determine own UDID
	case cannotDetermineUDID
	/// Cannot find node info
	case cannotFindNodesInfo
	/// Cannot find node info about ourselves
	case cannotFindSelfInfo
	/// Cannot find the node leader
	case cannotFindLeaderInfo
	/// Failed to retrieve the node's draining state
	case drainingNotFound
}

// MARK: - Client Protocol (Public)

/// User-visible status of ensemble
@available(macOS 15.0, iOS 18.0, *)
public enum EnsemblerStatus: Codable, CaseIterable, CustomStringConvertible {
	/// Cannot determine ensemble state
	case unknown
	/// Ensemble failed to load or initialize configuration
	case uninitialized
	/// Configuration loaded and initialization in progress
	case initializing
	/// Checking that the darwin-init applied successfully
	case initializingDarwinInitCheckInProgress
	/// Activating the backend since all of the activation checks passed
	case initializingActivationChecksOK
	/// Ready to Ensemble, talking to our pals
	case coordinating
	/// The leader has accepted us (followers only)
	case accepted
	/// The node is pairing with leader to get shared key
	case pairing
	/// The node has completed pairing
	case pairingComplete
    ///  Pairing due to key being rotated
    case keyRotationPairing
    ///  Attesting due to key being rotated
    case keyRotationAttesting
	/// The node has got the key and the key is validated by leader.
	case keyAccepted
	/// Ensemble in failed state; currently unrecoverable
	case failed
	/// An activation check failed before we even tried to activate the backend
	case failedActivationChecks
	/// Ensemble failed during a drain event. Almost certainly, a neighbor rebooted and we're about to reboot too!
	case failedWhileDraining
	/// All nodes checked in (state derived from leader), but not yet got the shared key.
	case attesting
	/// All nodes got their shared key, and set the key in CIO Mesh
	case ready

	// This `description` is logged. Do not expose private info!
	public var description: String {
		switch self {
		case .unknown:
			return "unknown"
		case .uninitialized:
			return "uninitialized"
		case .initializing:
			return "initializing"
		case .initializingDarwinInitCheckInProgress:
			return "initializingDarwinInitCheckInProgress"
		case .initializingActivationChecksOK:
			return "initializingActivationChecksOK"
		case .coordinating:
			return "coordinating"
		case .accepted:
			return "accepted"
		case .pairing:
			return "pairing"
		case .pairingComplete:
			return "pairingComplete"
        case .keyRotationPairing:
            return "keyRotationPairing"
        case .keyRotationAttesting:
            return "keyRotationAttesting"
		case .keyAccepted:
			return "keyAccepted"
		case .failed:
			return "failed"
		case .failedActivationChecks:
			return "failedActivationChecks"
		case .failedWhileDraining:
			return "failedWhileDraining"
		case .attesting:
			return "attesting"
		case .ready:
			return "ready"
		}
	}

	public func inFailedState() -> Bool {
		switch self {
		case .failed:
			fallthrough
		case .failedWhileDraining:
			fallthrough
		case .failedActivationChecks:
			return true
		default:
			return false
		}
	}
}

/// User visible information about nodes
@available(macOS 15.0, iOS 18.0, *)
public struct EnsembleNodeInfo: Codable {
	public let leader: Bool
	public let UDID: String
	public let rank: Int
	public let hostName: String?
	public var found: Bool?
	public var keyShared: Bool?

	public init(leader: Bool, UDID: String, rank: Int, hostName: String? = nil) {
		self.leader = leader
		self.UDID = UDID
		self.rank = rank
		self.hostName = hostName
	}
}

/// Copied verbatim from CMAC.
@available(macOS 15.0, iOS 18.0, *)
public enum HealthState: String, Sendable, Codable, CodingKeyRepresentable, Equatable {
	// responsive with known/confirmed bad
	case unhealthy
	// responsive with known/confirmed good
	case healthy
	// unresponsive - unknown good or unknown bad, but assume bad
	case unknown
	// responsive with unconfirmed good or bad (like during bootup)
	// this state specifically means that a caller should retry/checking again
	// the health state with some timeout before considering it unhealthy in
	// case the health domain application is stuck
	case initializing
}

/// Health status (derived from `EnsembleStatus`).
@available(macOS 15.0, iOS 18.0, *)
public struct EnsembleHealth: Sendable, Codable {
	public let healthState: HealthState
	public let metadata: [String: String]

	public init(healthState: HealthState, internalState: EnsemblerStatus) {
		self.healthState = healthState
		self.metadata = ["ensemblerStatus": "\(internalState)"]
	}
}

// MARK: - XPC Protocol (Private)

/// Possible commands for ensembled
@_spi(Daemon)
public enum EnsemblerRequest: Codable {
	/// Return Ensemble Status
	case getStatus
	/// Return whether the ensembler is draining
	case getDraining
	/// Return the ensemble's health
	case getHealth
	/// Get cable diagnostics for a failed ensemble. Only works on 8-node ensembles.
	case getCableDiagnostics
	/// Activate ensembed if not auto-initialized
	case activate
	// Debug Functionality
	/// Shutdown ensemble (Unimplemented)
	case shutdown
	/// Re-Attempt Configuration; only valid on configuration failure
	case reloadConfiguration
	/// Send a test message within the ensemble
	case sendTestMessage(Int)
	/// Get information on nodes assigned to ensemble
	case getNodeMap
	// Currently unimplemented, copied from the test tool (needed?)
	/// Get ensemble routes (Unimplemented)
	case getRoutes
	/// Get ensemble CIOMap (Unimplemented)
	case getCIOMap
	/// Encrypts the plain text data
	case encryptData(Data)
	/// Decrypts the encrypted  data
	case decryptData(Data)
	/// Rotate shared key
	case rotateSharedKey
	/// Get max buffers per key, after which CIO mesh will fail allocation
	case getMaxBuffersPerKey
	/// Get max seconds per key, after which CIO mesh will fail allocation
	case getMaxSecsPerKey
	/// Get the ensemble ID
	case getEnsembleID
	/// Get authetnication code
	case getAuthCode(Data)
}

/// XPC response from ensembled to the framework
@_spi(Daemon)
public struct EnsemblerResponse: Codable {
	/// Result of operation requested
	public var result: Bool
	/// Current status
	public var status: EnsemblerStatus?
	/// Current draining state (`true` if draining, `false` otherwise)
	public var draining: Bool?
	/// Error (if relevant)
	public var error: String?
	/// Dictionary of UDID/NodeInfo
	public var nodesInfo: [String: EnsembleNodeInfo]?
	/// Current health
	public var health: EnsembleHealth?
	/// List of Strings encoding cable diagnostics for an 8-node ensemble that failed to activate.
	public var cableDiagnostics: [String]?
	/// encrypted text
	public var encrypted: Data?
	/// decrypted text
	public var decrypted: Data?
	/// max buffers per key
    public var maxBuffersPerKey: UInt64?
    /// max seconds per key
	public var maxSecsPerKey: UInt64?
    /// ensemble ID
	public var ensembleID: String?
	/// authentication code
	public var authCode: Data?

	public init(
		result: Bool,
		status: EnsemblerStatus? = nil,
		draining: Bool? = nil,
		error: String? = nil,
		nodesInfo: [String: EnsembleNodeInfo]? = nil,
		encrypted: Data? = nil,
		decrypted: Data? = nil,
		health: EnsembleHealth? = nil,
		cableDiagnostics: [String]? = nil,
        maxSecsPerKey: UInt64? = nil,
		maxBuffersPerKey: UInt64? = nil,
		ensembleID: String? = nil,
		authCode: Data? = nil
	) {
		self.result = result
		self.status = status
		self.draining = draining
		self.error = error
		self.nodesInfo = nodesInfo
		self.health = health
		self.cableDiagnostics = cableDiagnostics
		self.encrypted = encrypted
		self.decrypted = decrypted
		self.maxSecsPerKey = maxSecsPerKey
		self.maxBuffersPerKey = maxBuffersPerKey
		self.ensembleID = ensembleID
		self.authCode = authCode
	}
}

// MARK: - Helper Functions (Private)

/// Quick and dirty helper function for fetching the UDID
@_spi(Daemon)
public func getNodeUDID() throws -> String {
	guard let udid = MGGetStringAnswer("UniqueDeviceID" as CFString) as? String else {
		throw EnsembleError.cannotDetermineUDID
	}
	return udid
}

// MARK: - Ensemble Session -

/// A client representing an XPC connection to ensembled. Session lifecycle is tied to the client
@available(macOS 15.0, iOS 18.0, *)
public class EnsemblerSession {
	let session: XPCSession
	let targetQ: DispatchQueue?
	let UDID: String

	static let logger = Logger(subsystem: "com.apple.cloudos.Ensemble", category: "EnsemblerClient")

	/// Optionally accept a dispatch queue to be passed to XPCSession
	///  - NOTE: will fall back to default queue if `nil`
	public init(targetQ: DispatchQueue?) throws {
		self.targetQ = targetQ
		do {
			self.session = try XPCSession(
				machService: kEnsemblerServiceName,
				targetQueue: self.targetQ
			)
			self.UDID = try getNodeUDID()
		} catch {
			EnsemblerSession.logger.error("Failed to establish XPC session with ensembled")
			throw EnsembleError.clientConnectionFailure
		}
	}

	deinit {
		self.session.cancel(reason: "client destroyed")
	}

	private func validateResponse(response: EnsemblerResponse) throws {
		guard response.result == true else {
			EnsemblerSession.logger
				.error(
					"Ensembler command failed, error: \(String(describing: response.error), privacy: .public)"
				)
			throw EnsembleError.commandFailure(error: response.error)
		}
	}

	private func grabStatus(response: EnsemblerResponse) throws -> EnsemblerStatus {
		guard let status = response.status else {
			EnsemblerSession.logger.error("No ensemble status found")
			throw EnsembleError.statusNotFound
		}
		EnsemblerSession.logger
			.info("Received status: \(String(describing: status), privacy: .public)")
		return status
	}

	private func grabDraining(response: EnsemblerResponse) throws -> Bool {
		guard let draining = response.draining else {
			EnsemblerSession.logger.error("No draining state found")
			throw EnsembleError.drainingNotFound
		}
		EnsemblerSession.logger
			.info("Received draining: \(draining, privacy: .public)")
		return draining
	}

	private func checkForReadiness(status: EnsemblerStatus?) -> Bool {
		guard status != nil else {
			EnsemblerSession.logger.error("Ensemble health status uninitialized")
			return false
		}

		switch status {
		case .ready:
			EnsemblerSession.logger.debug("Ensemble considered ready due to ready status")
			return true
		default:
			EnsemblerSession.logger
				.info(
					"Ensemble considered unready due to \(String(describing: status), privacy: .public) status"
				)
		}

		return false
	}

	private func validateNodesInfo(response: EnsemblerResponse) throws {
		try self.validateResponse(response: response)
		guard response.nodesInfo != nil else {
			EnsemblerSession.logger.error("No node info found in ensembler response")
			throw EnsembleError.cannotFindNodesInfo
		}
	}

	private func grabPeers(response: EnsemblerResponse) throws -> [EnsembleNodeInfo]? {
		try self.validateNodesInfo(response: response)
		guard let peers = response.nodesInfo?.filter({ $0.key != self.UDID }) else {
			EnsemblerSession.logger.info("No peers found in ensemble")
			return nil
		}

		return Array(peers.values)
	}

	private func grabSelf(response: EnsemblerResponse) throws -> EnsembleNodeInfo {
		try self.validateNodesInfo(response: response)
		guard let info = response.nodesInfo?[self.UDID] else {
			EnsemblerSession.logger.error("Could not find self in node info")
			throw EnsembleError.cannotFindSelfInfo
		}

		return info
	}

	private func grabLeader(response: EnsemblerResponse) throws -> EnsembleNodeInfo {
		try self.validateNodesInfo(response: response)
		guard let leader = response.nodesInfo?.first(where: { $0.value.leader == true }) else {
			EnsemblerSession.logger.error("Cannot find lead nodes")
			throw EnsembleError.cannotFindLeaderInfo
		}

		return leader.value
	}

	// MARK: - Client Methods (Public)

	/// Create a new EnsembleSession using the default queue for the underlying XPC Session
	public convenience init() throws {
		// While this is fairly easy to achieve with optionals/defaults, we maintain a simple
		// initializer as a binding interface contract
		try self.init(targetQ: nil)
	}

	/// Blocking call to get current status of ensemble
	public func getStatus() throws -> EnsemblerStatus {
		EnsemblerSession.logger.debug("Requesting ensemble status")
		let response = try sendCommandSync(command: EnsemblerRequest.getStatus)
		try self.validateResponse(response: response)
		return try self.grabStatus(response: response)
	}

	/// Async-Friendly(TM) call to get current status of the ensemble
	public func getStatus() async throws -> EnsemblerStatus {
		EnsemblerSession.logger.debug("Requesting ensemble status asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.getStatus)
		try self.validateResponse(response: response)
		return try self.grabStatus(response: response)
	}

	/// Simple blocking boolean determination of ensemble health
	///  - Returns: `true` if healthy and `false` if not healthy or indeterminate due to any issue
	/// including XPC issues
	public func isReady() -> Bool {
		EnsemblerSession.logger.debug("Triggering simple ensemble readiness check")
		do {
			let status = try getStatus()
			return self.checkForReadiness(status: status)
		} catch {
			EnsemblerSession.logger
				.error("Failed to perform simple health check so reporting false")
			return false
		}
	}

	/// Simple Async-Friendly(TM) boolean determination of ensemble health
	/// - Returns: `true` if healthy and `false` if not healthy or indeterminate due to any issue
	/// including XPC issues
	public func isReady() async -> Bool {
		EnsemblerSession.logger.debug("Triggering simple async ensemble readiness check")
		do {
			let status = try await getStatus()
			return self.checkForReadiness(status: status)
		} catch {
			EnsemblerSession.logger
				.error("Failed to perform simple health check so reporting false")
			return false
		}
	}

	/// Simple blocking boolean determination of ensemble draining
	///  - Returns: `true` if draining and `false` otherwise
	public func isDraining() throws -> Bool {
		EnsemblerSession.logger.debug("Requesting ensemble draining state")
		let response = try sendCommandSync(command: EnsemblerRequest.getDraining)
		try self.validateResponse(response: response)
		return try self.grabDraining(response: response)
	}

	/// Simple Async-Friendly(TM) determination of ensemble draining
	/// - Returns: `true` if draining and `false` otherwise
	public func isDraining() async throws -> Bool {
		EnsemblerSession.logger.debug("Requesting ensemble draining asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.getDraining)
		try self.validateResponse(response: response)
		return try self.grabDraining(response: response)
	}

	/// Blocking call to receive node information for self
	/// - Returns: `EnsembleNodeInfo` including information for current node or throws an error if
	/// it cannot be provided
	public func getNodeInfo() throws -> EnsembleNodeInfo {
		EnsemblerSession.logger.debug("Requesting node information")
		let response = try sendCommandSync(command: .getNodeMap)
		return try self.grabSelf(response: response)
	}

	/// Async-Friendly(TM) call to receive node information for self
	/// - Returns: `EnsembleNodeInfo` including information for current node or throws an error if
	/// it cannot be provided
	public func getNodeInfo() async throws -> EnsembleNodeInfo {
		EnsemblerSession.logger.debug("Requesting node information asynchronously")
		let response = try await sendCommandAsync(command: .getNodeMap)
		return try self.grabSelf(response: response)
	}

    
    /// Blocking call to get NWProtocolTLS.Options
    /// - Returns: `NWProtocolTLS.Options`- An object that contains security options to use for TLS handshakes.
    public func getTlsOptions() throws -> NWProtocolTLS.Options {
        EnsemblerSession.logger.debug("Requesting sec options")
        //we need authStr string to get authcode back from ensembled ( using the shared key),
        // Because we rely on this on all the nodes, we need to have hardcoded string. ( unless, we come up with other protocol to distributed this string).
        // The authcode we get will be eventually used to get the NWProtocolTLS.Options
        let authStr = "ensembled"
        guard let data = authStr.data(using: .utf8) else {
            throw EnsembleError.commandFailure(
                error: "Error converting to Data"
            )
        }
        let authCode = try getAuthCode(data: data)
        let options = try getTlsOptions(authStr: authStr, authCode: authCode )
        return options
    }

    /// Async-Friendly(TM) call to get NWProtocolTLS.Options
    /// - Returns: `NWProtocolTLS.Options`- An object that contains security options to use for TLS handshakes.
    public func getTlsOptions() async throws -> NWProtocolTLS.Options {
        EnsemblerSession.logger.debug("Requesting sec options")
        let authStr = "ensembled"
        guard let data = authStr.data(using: .utf8) else {
            throw EnsembleError.commandFailure(
                error: "Error converting to Data"
            )
        }
        let authCode = try  await getAuthCode(data: data)
        let options = try getTlsOptions(authStr: authStr, authCode: authCode )
        return options
    }
    
	/// Blocking call to receive a list of peers in the ensemble (not including self)
	/// - Returns: List of peers if any exist or `nil`
	/// - Note: We assume that no peers is a legal state so no error is thrown
	public func getPeerInfo() throws -> [EnsembleNodeInfo]? {
		EnsemblerSession.logger.debug("Request peer node info")
		let response = try sendCommandSync(command: .getNodeMap)
		return try self.grabPeers(response: response)
	}

	/// Async-Friendly(TM)  call to receive a list of peers in the ensemble (not including self)
	/// - Returns: List of peers if any exist or `nil`
	/// - Note: We assume that no peers is a legal state so no error is thrown
	public func getPeerInfo() async throws -> [EnsembleNodeInfo]? {
		EnsemblerSession.logger.debug("Request peer node info asynchronously")
		let response = try await sendCommandAsync(command: .getNodeMap)
		return try self.grabPeers(response: response)
	}

	/// Blocking call to receive node information for the ensemble leader
	/// - Returns: `EnsembleNodeInfo` including information for leader node or throws an error if it
	/// cannot be provided
	public func getLeaderInfo() throws -> EnsembleNodeInfo {
		EnsemblerSession.logger.debug("Requesting ensemble leader information")
		let response = try sendCommandSync(command: .getNodeMap)
		return try self.grabLeader(response: response)
	}

	/// Async-Friendly(TM) call to receive node information for the ensemble leader
	/// - Returns: `EnsembleNodeInfo` including information for leader node or throws an error if it
	/// cannot be provided
	public func getLeaderInfo() async throws -> EnsembleNodeInfo {
		EnsemblerSession.logger.debug("Requesting ensemble leader information asynchronously")
		let response = try await sendCommandAsync(command: .getNodeMap)
		return try self.grabLeader(response: response)
	}

	/// Blocking call to receive maxBuffersPerKey
	/// - Returns: `maxBuffersPerKey
    public func getMaxBuffersPerKey() throws -> UInt64 {
		EnsemblerSession.logger.debug("Getting max buffers per key")
		let response = try sendCommandSync(command: .getMaxBuffersPerKey)
        guard let maxBuffersPerKey = response.maxBuffersPerKey else {
            throw EnsembleError.commandFailure(
                error: "Error getting max buffers per key."
            )
        }
        return maxBuffersPerKey
	}

	/// Async-Friendly(TM) call to receive maxBuffersPerKey
	/// - Returns: maxBuffersPerKey
	public func getMaxBuffersPerKey() async throws -> UInt64 {
		EnsemblerSession.logger.debug("Getting max buffers per key asynchronously")
		let response = try await sendCommandAsync(command: .getMaxBuffersPerKey)
        guard let maxBuffersPerKey = response.maxBuffersPerKey else {
            throw EnsembleError.commandFailure(
                error: "Error getting max buffers per key."
            )
        }
        return maxBuffersPerKey
	}

	/// Blocking call to receive maxSecsPerKey
	/// - Returns: `maxSecsPerKey
	public func getMaxSecsPerKey() throws -> UInt64 {
		EnsemblerSession.logger.debug("Getting max seconds per key")
		let response = try sendCommandSync(command: .getMaxSecsPerKey)
        guard let maxSecsPerKey = response.maxSecsPerKey else {
            throw EnsembleError.commandFailure(
                error: "Error getting max seconds per key."
            )
        }
        return maxSecsPerKey
	}

	/// Async-Friendly(TM) call to receive maxSecsPerKey
	/// - Returns: maxSecsPerKey
	public func getMaxSecsPerKey() async throws -> UInt64 {
		EnsemblerSession.logger.debug("Getting max seconds per key asynchronously")
		let response = try await sendCommandAsync(command: .getMaxSecsPerKey)
        guard let maxSecsPerKey = response.maxSecsPerKey else {
            throw EnsembleError.commandFailure(
                error: "Error getting max seconds per key."
            )
        }
        return maxSecsPerKey
	}

	/// Blocking call boolean check if the current node is the ensemble leader
	/// - Returns: `true` if leader, throws `EnsembleError` if we cannot make the determination
	public func isLeader() throws -> Bool {
		EnsemblerSession.logger.debug("Checking if we're the leader")
		let response = try sendCommandSync(command: .getNodeMap)
		let node = try grabSelf(response: response)
		return node.leader
	}

	/// Async-Friendly(TM)  call boolean check if the current node is the ensemble leader
	/// - Returns: `true` if leader, throws `EnsembleError` if we cannot make the determination
	public func isLeader() async throws -> Bool {
		EnsemblerSession.logger.debug("Checking if we're the leader")
		let response = try await sendCommandAsync(command: .getNodeMap)
		let node = try grabSelf(response: response)
		return node.leader
	}

	/// Blocking call to get the ensemble's health state
	public func getHealth() throws -> EnsembleHealth {
		EnsemblerSession.logger.debug("Prompting ensembled to get health")
		let response = try sendCommandSync(command: EnsemblerRequest.getHealth)
		try self.validateResponse(response: response)
		guard let health = response.health else {
			return EnsembleHealth(healthState: HealthState.initializing,
								  internalState: EnsemblerStatus.uninitialized)
		}
		return health
	}

	/// Async-Friendly(TM) call to get the ensemble's health state
	public func getHealth() async throws -> EnsembleHealth {
		EnsemblerSession.logger.debug("Prompting ensembled to get health asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.getHealth)
		try self.validateResponse(response: response)
		guard let health = response.health else {
			return EnsembleHealth(healthState: HealthState.initializing,
								  internalState: EnsemblerStatus.uninitialized)
		}
		return health
	}

	/// Blocking call to get cable diagnostics for a failed ensemble
	public func getCableDiagnostics() throws -> [String] {
		EnsemblerSession.logger.debug("Prompting ensembled to get cable diagnostics")
		let response = try sendCommandSync(command: EnsemblerRequest.getCableDiagnostics)
		try self.validateResponse(response: response)
		guard let diags = response.cableDiagnostics else {
			throw EnsembleError.commandFailure(
				error: "No cable diagnostics were returned. That's all I know."
			)
		}
		return diags
	}

	/// Async-Friendly(TM) call to get cable diagnostics for a failed ensemble
	public func getCableDiagnostics() async throws -> [String] {
		EnsemblerSession.logger.debug("Prompting ensembled to get cable diagnostics asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.getCableDiagnostics)
		try self.validateResponse(response: response)
		guard let diags = response.cableDiagnostics else {
			throw EnsembleError.commandFailure(
				error: "No cable diagnostics were returned. That's all I know."
			)
		}
		return diags
	}

    /// Blocking call to get authentication code
    private func getAuthCode(data: Data) throws -> Data {
        EnsemblerSession.logger.info("Getting authentication code")
        let response = try sendCommandSync(command: EnsemblerRequest.getAuthCode(data))
        try self.validateResponse(response: response)
        guard let authCode = response.authCode else {
            throw EnsembleError.commandFailure(
                error: "No authcode returned."
            )
        }
        return authCode
    }

    /// Async-Friendly(TM) call to get authentication code
    private func getAuthCode(data: Data) async throws -> Data {
        EnsemblerSession.logger.info("Getting authentication code async")
        let response = try await sendCommandAsync(command: EnsemblerRequest.getAuthCode(data))
        try self.validateResponse(response: response)
        guard let authCode = response.authCode else {
            throw EnsembleError.commandFailure(
                error: "No authcode returned."
            )
        }
        return authCode
    }
    
	/// Blocking call to rotate shared key
	public func rotateSharedKey() throws -> Bool {
		EnsemblerSession.logger.debug("Rotating shared key ")
		let response = try sendCommandSync(command: EnsemblerRequest.rotateSharedKey)
		return response.result
	}

	/// Async-Friendly(TM) call to rotate shared key
	public func rotateSharedKey() async throws -> Bool {
		EnsemblerSession.logger.debug("Rotating shared key")
		let response = try await sendCommandAsync(command: EnsemblerRequest.rotateSharedKey)
		return response.result
	}

	/// Blocking call to get ensemble ID
	public func getEnsembleID() throws -> String? {
		EnsemblerSession.logger.debug("Requesting ensemble ID")
		let response = try sendCommandSync(command: EnsemblerRequest.getEnsembleID)
		try self.validateResponse(response: response)
		return response.ensembleID
	}

	/// Async-Friendly call to get ensemble ID
	public func getEnsembleID() async throws -> String? {
		EnsemblerSession.logger.debug("Requesting ensemble ID asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.getEnsembleID)
		try self.validateResponse(response: response)
		return response.ensembleID
	}

	// MARK: - SPIs for the CLI

	/// Blocking call to activate ensemble and receive result on this node
	///  - NOTE: This is typically a NOOP
	@_spi(Debug)
	public func activate() throws -> Bool {
		EnsemblerSession.logger.debug("Activating ensemble")
		let response = try sendCommandSync(command: EnsemblerRequest.activate)
		return response.result
	}

	/// Async-Friendly(TM) call to activate ensemble on this node
	///  - NOTE: This is typically a NOOP
	@_spi(Debug)
	public func activate() async throws -> Bool {
		EnsemblerSession.logger.debug("Activating ensemble asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.activate)
		return response.result
	}

	/// Blocking call to encrypt plain text data
	@_spi(Debug)
	public func encryptData(data: Data) throws -> Data? {
		EnsemblerSession.logger.debug("Encrypting plain text data ")
		let response = try sendCommandSync(command: EnsemblerRequest.encryptData(data))
		return response.encrypted
	}

	/// Async-Friendly(TM) call to encrypt plain text data
	@_spi(Debug)
	public func encryptData(data: Data) async throws -> Data? {
		EnsemblerSession.logger.debug("Encrypting plain text data ")
		let response = try await sendCommandAsync(command: EnsemblerRequest.encryptData(data))
		return response.encrypted
	}

	/// Blocking call to decrypt  text data
	@_spi(Debug)
	public func decryptData(data: Data) throws -> Data? {
		EnsemblerSession.logger.debug("Decrypting data ")
		let response = try sendCommandSync(command: EnsemblerRequest.decryptData(data))
		return response.decrypted
	}

	/// Async-Friendly(TM) call to decrypt  text data
	@_spi(Debug)
	public func decryptData(data: Data) async throws -> Data? {
		EnsemblerSession.logger.debug("Decrypting data ")
		let response = try await sendCommandAsync(command: EnsemblerRequest.decryptData(data))
		return response.decrypted
	}

	/// Blocking call to prompt ensembled to reload its configuration
	///  - NOTE: This should typically only be needed in debug scenarios
	@_spi(Debug)
	public func reloadConfiguration() throws -> Bool {
		EnsemblerSession.logger.debug("Prompting ensembled to reload ensemble configuration")
		let response = try sendCommandSync(command: EnsemblerRequest.reloadConfiguration)
		return response.result
	}

	/// Async-Friendly(TM) call to prompt ensembled to reload its configuration
	///  - NOTE: This should only be needed for debug scenarios
	@_spi(Debug)
	public func reloadConfiguration() async throws -> Bool {
		EnsemblerSession.logger
			.debug("Prompting ensembled to reload ensemble configuration asynchronously")
		let response = try await sendCommandAsync(command: EnsemblerRequest.reloadConfiguration)
		return response.result
	}

	/// Blocking call to prompt ensembled to send a test message to a given node
	///  - Parameter destination: The destination node by rank
	@_spi(Debug)
	public func sendTestMessage(destination: Int) throws -> Bool {
		EnsemblerSession.logger
			.debug("Prompting ensembled to send a test message to \(destination)")
		let response = try sendCommandSync(command: EnsemblerRequest.sendTestMessage(destination))
		return response.result
	}

	/// Async-Friendly(TM) call to prompt ensembled to send a test message to a given node
	///  - Parameter destination: The destination node by rank
	@_spi(Debug)
	public func sendTestMessage(destination: Int) async throws -> Bool {
		EnsemblerSession.logger
			.debug("Prompting ensembled to send a test message to \(destination) asynchronously")
		let response = try await sendCommandAsync(
			command: EnsemblerRequest
				.sendTestMessage(destination)
		)
		return response.result
	}
}

// MARK: - Private XPC Methods

extension EnsemblerSession {
	private func sendCommandSync(command: EnsemblerRequest) throws -> EnsemblerResponse {
		let reply = try session.sendSync(command)
		return try reply.decode(as: EnsemblerResponse.self)
	}

	private func sendCommandAsync(command: EnsemblerRequest) async throws -> EnsemblerResponse {
		let reply = try await withCheckedThrowingContinuation { continuation in
			do {
				try self.session.send(command, replyHandler: continuation.resume(with:))
			} catch {
				continuation.resume(throwing: error)
			}
		}

		return try reply.decode(as: EnsemblerResponse.self)
	}
    
    // create a utility function to encode strings as preshared key data.
    private func stringToDispatchData(_ string: String) -> DispatchData? {
        guard let stringData = string.data(using: .utf8) else {
            return nil
        }
        let dispatchData = stringData.withUnsafeBytes {
            DispatchData(bytes: $0)
        }
        return dispatchData
    }
    
    private func getTlsOptions(authStr: String, authCode: Data) throws -> NWProtocolTLS.Options {
        let tlsOptions = NWProtocolTLS.Options()

        let authenticationData = authCode.withUnsafeBytes {
            DispatchData(bytes: $0)
        }

        guard let data = stringToDispatchData(authStr) else {
            throw EnsembleError.commandFailure(
                error: "Error converting to dispatch data."
            )
        }
        sec_protocol_options_add_pre_shared_key(tlsOptions.securityProtocolOptions,
                                                authenticationData as __DispatchData,
                                                data as __DispatchData)
        
        // Replace this with actual value from security framework when it is available
        // rdar://134442920
        let TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256 : UInt16 = 0xCCAC
        
        EnsemblerSession.logger.log("Using TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256 cipher suite")
        
        guard let tlsCipher =  tls_ciphersuite_t(rawValue: TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256) else {
            throw EnsembleError.commandFailure(
                error: "Error creating tls ciphersuite."
            )
        }
        
        sec_protocol_options_append_tls_ciphersuite(tlsOptions.securityProtocolOptions,
                                                    tlsCipher)
        return tlsOptions
    }
}
