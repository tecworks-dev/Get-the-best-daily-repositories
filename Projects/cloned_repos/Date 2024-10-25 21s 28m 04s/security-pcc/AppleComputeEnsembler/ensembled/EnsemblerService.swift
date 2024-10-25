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
//  EnsemblerService.swift
//  ensembled
//
//  Created by Alex T Newman on 12/21/23.
//

import DarwinPrivate.os.variant
import EnsembleConfiguration
import Foundation
import OSLog
import Security

@_spi(Daemon) import Ensemble // _spi interface used for mach service name, protocol

@_spi(Private) import XPC // Private/_spi interfaces used for entitlement-checking
import XPCPrivate
import CryptoKit

let kEnsemblerEntitlementPrefix = "com.apple.private.AppleComputeEnsembler"
let kEnsemblerStatusEntitlement = kEnsemblerEntitlementPrefix + ".read"
let kEnsemblerControlEntitlement = kEnsemblerEntitlementPrefix + ".control"
let kEnsemblerHealthEntitlement = kEnsemblerEntitlementPrefix + ".health"
let kEnsemblerCoordinationServiceEntitlement = kEnsemblerEntitlementPrefix + ".coordination-service"

public enum EnsemblerServiceError: Error {
	case unentitledCaller
	case unauthorizedOrUnknownOperation
	/// Most likely configuration failure
	case ensemblerInitializationError
	case notImplemented
	case serviceInitializationFailure(error: Error? = nil)
	case configurationError(error: Error? = nil)
	case ensemblerInFailedState
}

/// XPC server for EnsemblerService
struct EnsemblerService {
	static var ensembler: Ensembler?

	static let logger = Logger(subsystem: kEnsemblerPrefix, category: "Service")
	let kEnsembleActivationPreferenceKey = "AutoActivation"
	let kEnsembleAutoRestartPreferenceKey = "AutoRestartForDevOnly"
	let kSkipDarwinInitCheckPreferenceKey = "SkipDarwinInitCheck"
	let kDarwinInitTimeoutPreferenceKey = "DarwinInitTimeout"
	let kEnsembleUseStubAttestationPreferenceKey = "UseStubAttestation"

	// If we did not suppress activation in the daemon, we should assume that it is needed
	private func checkPrefsForActivation() -> Bool {
		guard let activate = CFPreferencesCopyValue(
			kEnsembleActivationPreferenceKey as CFString,
			kEnsemblerPrefix as CFString,
			kCFPreferencesAnyUser,
			kCFPreferencesAnyHost
		) as? Bool else {
			EnsemblerService.logger.info(
				"""
				Did not find \(self.kEnsembleActivationPreferenceKey) preference, \
				proceeding to activate...
				"""
			)
			return true
		}
		return activate
	}

	func checkPrefsForAutoRestart() -> Bool {
		guard let enableAutoRestart = CFPreferencesCopyValue(
			kEnsembleAutoRestartPreferenceKey as CFString,
			kEnsemblerPrefix as CFString,
			kCFPreferencesAnyUser,
			kCFPreferencesAnyHost
		) as? Bool else {
			EnsemblerService.logger.info(
				"""
				Did not find \(self.kEnsembleAutoRestartPreferenceKey) preference, \
				defaulting to false...
				"""
			)
			return false
		}
		if enableAutoRestart,
		   !os_variant_allows_internal_security_policies(kEnsemblerEntitlementPrefix) {
			EnsemblerService.logger.error(
				"""
				Oops: \(self.kEnsembleAutoRestartPreferenceKey) \
				can only be set to true on an internal \
				OS variant. Forcing to false.
				"""
			)
			return false
		}
		return enableAutoRestart
	}

	func checkPrefsForSkipDarwinInitCheck() -> Bool? {
		guard let skipDarwinInitCheck = CFPreferencesCopyValue(
			kSkipDarwinInitCheckPreferenceKey as CFString,
			kEnsemblerPrefix as CFString,
			kCFPreferencesAnyUser,
			kCFPreferencesAnyHost
		) as? Bool else {
			return nil
		}
		if skipDarwinInitCheck,
		   !os_variant_allows_internal_security_policies(kEnsemblerEntitlementPrefix) {
			EnsemblerService.logger.error(
				"""
				Oops: \(self.kSkipDarwinInitCheckPreferenceKey) \
				can only be set to true on an internal \
				OS variant. Forcing to false.
				"""
			)
			return false
		}
		return skipDarwinInitCheck
	}

	func checkPrefsForDarwinInitTimeout() -> Int? {
		guard let darwinInitTimeout = CFPreferencesCopyValue(
			kDarwinInitTimeoutPreferenceKey as CFString,
			kEnsemblerPrefix as CFString,
			kCFPreferencesAnyUser,
			kCFPreferencesAnyHost
		) as? Int else {
			EnsemblerService.logger.info(
				"""
				Did not find \(self.kDarwinInitTimeoutPreferenceKey) preference, \
				ignoring...
				"""
			)
			return nil
		}
		return darwinInitTimeout
	}

	private func checkPrefsForUseStubAttestation() -> Bool {
		#if os(iOS)
		guard let useStubAttestation = CFPreferencesCopyValue(
			kEnsembleUseStubAttestationPreferenceKey as CFString,
			kEnsemblerPrefix as CFString,
			kCFPreferencesAnyUser,
			kCFPreferencesAnyHost
		) as? Bool else {
			EnsemblerService.logger.info(
				"""
				Did not find \(self.kEnsembleUseStubAttestationPreferenceKey) preference, \
				defaulting to false for iOS
				"""
			)
			return false
		}

		if useStubAttestation,
		   !os_variant_allows_internal_security_policies(kEnsemblerEntitlementPrefix) {
			EnsemblerService.logger.error(
				"""
				Oops: \(self.kEnsembleUseStubAttestationPreferenceKey) \
				can only be set to true on an internal \
				OS variant. Forcing to false.
				"""
			)
			return false
		}

		return useStubAttestation
		#elseif os(macOS)
		return true
		#else
		fatalError("Oops: Unsupported OS.")
		#endif
	}

	init() {
		EnsemblerService.logger.info("Initializing EnsemblerService.")

		initCloudMetricsFrameworkBackend()

		// Attempt an early initialization of our configuration to avoid colliding with our
		// listener down the line
		do {
			let autoRestart = self.checkPrefsForAutoRestart()
			let skipDarwinInitCheck = self.checkPrefsForSkipDarwinInitCheck()
			let darwinInitTimeout = self.checkPrefsForDarwinInitTimeout()
			let useStubAttestation = self.checkPrefsForUseStubAttestation()
			let allowDefaultOneNodeConfig = os_variant_allows_internal_security_policies(
				kEnsemblerEntitlementPrefix
			)
			EnsemblerService.ensembler = try Ensembler(
				autoRestart: autoRestart,
				skipDarwinInitCheckOpt: skipDarwinInitCheck,
				darwinInitTimeout: darwinInitTimeout,
				useStubAttestation: useStubAttestation,
				allowDefaultOneNodeConfig: allowDefaultOneNodeConfig
			)
			// Has the daemon been told to unconditionally activate?
			if self.checkPrefsForActivation() == true {
				try EnsemblerService.ensembler?.activate()
			}

		} catch {
			// This state is technically not fatal. We just won't initialize.
			EnsemblerService.logger.error(
				"Failed to initialize ensembler service at load: \(error)"
			)
		}
	}

	struct SessionHandler: XPCPeerHandler {
		/// Get status of ensemble
		/// This information can be provided regardless of whether a configuration has been loaded
		func getStatus() -> Encodable {
			EnsemblerService.logger.debug("Fetching status")
			guard let status = EnsemblerService.ensembler?.status else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: true, status: .uninitialized)
			}
			// If the ensembler is initialized, we fetch the status
			EnsemblerService.logger.info("Found ensembler, returning status: \(status)")
			return EnsemblerResponse(result: true, status: status)
		}

		/// Get the draining state of the ensemble
		/// This information can be provided regardless of whether a configuration has been loaded
		func getDraining() -> Encodable {
			EnsemblerService.logger.info("Fetching draining state")
			guard EnsemblerService.ensembler != nil else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: false, draining: nil)
			}
			// If the ensembler is initialized, we fetch the draining state
			EnsemblerService.logger.info("Found ensembler, returning draining state...")
			return EnsemblerResponse(result: true, draining: EnsemblerService.ensembler?.draining)
		}

		/// Get ensemble ID
		func getEnsembleID() -> Encodable {
			EnsemblerService.logger.debug("Fetching ensemble ID")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: false, status: .uninitialized)
			}
			EnsemblerService.logger.debug("Found ensembler, getting ensemble ID...")
			return EnsemblerResponse(result: true, ensembleID: ensembler.ensembleID)
		}

		/// Get maxBuffersPerKey
		func getMaxBuffersPerKey() -> Encodable {
			EnsemblerService.logger.info("Fetching max buffers per key")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: false, status: .uninitialized)
			}

			EnsemblerService.logger.info("Found ensembler, getting max buffers per key...")
            do {
                return EnsemblerResponse(result: true, maxBuffersPerKey: try ensembler.getMaxBuffersPerKey())
            }
            catch {
                EnsemblerService.logger.error("Error getting max buffers per key: \(error)")
                return EnsemblerResponse(result: false)
            }
		}

		/// Get maxSecsPerKey
		func getMaxSecsPerKey() -> Encodable {
			EnsemblerService.logger.info("Fetching max seconds per key")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: false, status: .uninitialized)
			}

			EnsemblerService.logger.info("Found ensembler, getting max seconds per key...")
            do {
                return EnsemblerResponse(result: true, maxSecsPerKey: try ensembler.getMaxSecondsPerKey())
            }
            catch {
                EnsemblerService.logger.error("Error getting max buffers per key: \(error)")
                return EnsemblerResponse(result: false)
            }
		}

		/// Encrypts plain text data
		func encrypt(data: Data) -> Encodable {
			EnsemblerService.logger.info("Encrypting the text")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: true, status: .uninitialized)
			}

			guard ensembler.status == .ready else {
				EnsemblerService.logger.error(
					"Ensembler not ready yet. Cannot encrypt before status is ready"
				)
				return EnsemblerResponse(result: true, status: ensembler.status)
			}

			do {
				let encrypted = try ensembler.encryptData(data: data)
				return EnsemblerResponse(result: true, encrypted: encrypted)
			} catch {
				EnsemblerService.logger.error("Error encrypting text")
				return EnsemblerResponse(result: false)
			}
		}

		/// Decrypts the data
		func decrypt(data: Data) -> Encodable {
			EnsemblerService.logger.info("Decrypting the text")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: true, status: .uninitialized)
			}

			guard ensembler.status == .ready else {
				EnsemblerService.logger
					.warning("Ensembler not ready yet. Cannot decrypt before status is ready")
				return EnsemblerResponse(result: true, status: ensembler.status)
			}

			do {
				let decrypted = try ensembler.decryptData(data: data)
				return EnsemblerResponse(result: true, decrypted: decrypted)
			} catch {
				EnsemblerService.logger.error("Error decrypting text")
				return EnsemblerResponse(result: false)
			}
		}
        
        /// Get Authcode for the data
        func getAuthCode(data: Data) -> Encodable {
            EnsemblerService.logger.info("Getting the auth code")
            guard let ensembler = EnsemblerService.ensembler else {
                EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
                return EnsemblerResponse(result: false, status: .uninitialized)
            }

            guard ensembler.status == .ready else {
                EnsemblerService.logger
                    .error("Ensembler not ready yet. Cannot get auth code before status is ready")
                return EnsemblerResponse(result: false, status: ensembler.status)
            }

            do {
                let authCode = try ensembler.getAuthCode(data: data)
                return EnsemblerResponse(result: true, authCode: authCode)
            } catch {
                EnsemblerService.logger.error("Error getting auth code : \(error)")
                return EnsemblerResponse(result: false)
            }
        }

		/// Rotate the shared key
		func rotateSharedKey() -> Encodable {
			EnsemblerService.logger.info("Rotating shared Key")
			guard let ensembler = EnsemblerService.ensembler else {
				EnsemblerService.logger.error("Ensembler NOT configured! Initialization error?")
				return EnsemblerResponse(result: true, status: .uninitialized)
			}

			guard ensembler.status == .ready else {
				EnsemblerService.logger
					.warning(
						"Ensembler not ready yet. Cannot rotate the shared key before the status is ready"
					)
				return EnsemblerResponse(result: true, status: ensembler.status)
			}

			do {
				try ensembler.rotateKey()
			} catch {
				EnsemblerService.logger.error("Error rotating shared key")
				return EnsemblerResponse(result: false)
			}

			return EnsemblerResponse(result: true)
		}

		func reloadConfiguration() -> Encodable {
			EnsemblerService.logger.info("Attempting to reload configuration")
			// We have no legal re-configuration model for a running ensemble at this time
			guard EnsemblerService.ensembler == nil else {
				EnsemblerService.logger.info("Ensembler already initialized, cannot reconfigure")
				// While not technically success, we are configured and should return true
				return EnsemblerResponse(result: true)
			}
			do {
				EnsemblerService.ensembler = try Ensembler()
			} catch {
				EnsemblerService.logger.error("Failed to reload configuration: \(error)")
				return EnsemblerResponse(result: false)
			}
			EnsemblerService.logger.info("Configuration loaded.")
			return EnsemblerResponse(result: true)
		}

		func activate() -> Encodable {
			EnsemblerService.logger.info("Attempting to activate backend")
			do {
				try EnsemblerService.ensembler?.activate()
				return EnsemblerResponse(result: true)
			} catch {
				EnsemblerService.logger.error("Ensembler activation failed: \(error)")
				return EnsemblerResponse(result: false, error: String(describing: error))
			}
		}

		func getNodeMap() -> Encodable {
			EnsemblerService.logger.debug("Attempting to fetch nodeMap")
			return EnsemblerResponse(result: true, nodesInfo: EnsemblerService.ensembler?.nodeMap)
		}

		func sendTestMessage(destination: Int) -> Encodable {
			EnsemblerService.logger.info("Attempting to send test message to \(destination)")
			do {
				try EnsemblerService.ensembler?.sendTestMessage(destination: destination)
				return EnsemblerResponse(result: true)
			} catch {
				EnsemblerService.logger.error(
					"Failed to send test message to \(destination), error: \(error)"
				)
				return EnsemblerResponse(result: false, error: String(describing: error))
			}
		}

		func getHealthState() -> Encodable {
			EnsemblerService.logger.info("Attempting to get ensembler health state")
			let health = EnsemblerService.ensembler?.getHealth()
			return EnsemblerResponse(result: true, health: health)
		}

		func getCableDiagnostics() -> Encodable {
			EnsemblerService.logger.info("Attempting to get cable diagnostics")
			do {
				let diags = try EnsemblerService.ensembler?.checkConnectivity()
				return EnsemblerResponse(result: true, cableDiagnostics: diags)
			} catch {
				EnsemblerService.logger.error("Failed to get cable diagnostics, error: \(error)")
				return EnsemblerResponse(result: false, error: String(describing: error))
			}
		}
      
		func handleIncomingRequest(_ message: XPCReceivedMessage) -> Encodable? {
			var audit = message.auditToken

			// We always check for a read entitlement
			guard let readEntitlement = xpc_copy_entitlement_for_token(
				kEnsemblerStatusEntitlement,
				&audit
			)
			else {
				let err = EnsemblerServiceError.unentitledCaller
				EnsemblerService.logger.error("Refused client: \(err)")
				return EnsemblerResponse(result: false, error: String(describing: err))
			}

			guard xpc_bool_get_value(readEntitlement) == true else {
				let err = EnsemblerServiceError.unauthorizedOrUnknownOperation
				EnsemblerService.logger.error("Refused client: \(err)")
				return EnsemblerResponse(result: false, error: String(describing: err))
			}

			var controlAllowed = false

			// A control entitlement is sometimes needed
			if let controlEntitlement = xpc_copy_entitlement_for_token(
				kEnsemblerControlEntitlement,
				&audit
			) {
				controlAllowed = xpc_bool_get_value(controlEntitlement)
			}

			var healthAllowed = controlAllowed
			if !healthAllowed,
			   let healthEntitlement = xpc_copy_entitlement_for_token(
					kEnsemblerHealthEntitlement,
					&audit
			   ) {
				healthAllowed = xpc_bool_get_value(healthEntitlement)
			}

            var coordinationServiceAllowed = false

            // A coordinationservice entitlement is  needed for getting tls options
            if let coordinationServiceEntitlement = xpc_copy_entitlement_for_token(
                kEnsemblerCoordinationServiceEntitlement,
                &audit
            ) {
                coordinationServiceAllowed = xpc_bool_get_value(coordinationServiceEntitlement)
            }
            
			// Attempt to re-initialize if we failed before
			if EnsemblerService.ensembler == nil {
				_ = self.reloadConfiguration()
			}

			do {
				let request = try message.decode(as: EnsemblerRequest.self)

				// Some commands can be used with an uninitialized Ensembler
				switch (request, controlAllowed) {
				case (.getStatus, _):
					return self.getStatus()
				case (.getDraining, _):
					return self.getDraining()
				case (.reloadConfiguration, true):
					// TODO: This is kind of a noop, perhaps it should force cfprefs to re-read
					// TODO: from disk or something?
					return self.reloadConfiguration()
				default:
					break
				}

				// Some cannot. Now we bail if we have no ensembler
				guard EnsemblerService.ensembler != nil else {
					let error = EnsemblerServiceError.ensemblerInitializationError
					return EnsemblerResponse(result: false, error: String(describing: error))
				}

				// For health state and cable diagnostics, check `healthAllowed`.
                // For getting auth code used for tls options, check `coordinationServiceAllowed`
				switch (request) {
				case .getHealth, .getCableDiagnostics:
					guard healthAllowed else {
						let err = EnsemblerServiceError.unauthorizedOrUnknownOperation
						EnsemblerService.logger.error(
						   """
						   Missing entitlement: \
						   .getCableDiagnostics needs at least one of the following: \
						   [\(kEnsemblerHealthEntitlement), \(kEnsemblerControlEntitlement)]
						   """
						)
						return EnsemblerResponse(result: false, error: String(describing: err))
					}
                case .getAuthCode(let data):
                    guard coordinationServiceAllowed else {
                        let err = EnsemblerServiceError.unauthorizedOrUnknownOperation
                        EnsemblerService.logger.error(
                           """
                           Missing entitlement: \
                           .getAuthCode needs following entitlement: \
                           [\(kEnsemblerCoordinationServiceEntitlement)]
                           """
                        )
                        return EnsemblerResponse(result: false, error: String(describing: err))
                    }
                    return self.getAuthCode(data: data)
				default:
					break
				}

				switch (request, controlAllowed) {
				// Read-only functionality
				case (.getNodeMap, _):
					return self.getNodeMap()
				case (.getEnsembleID, _):
					return self.getEnsembleID()
				case (.getHealth, _):
					return self.getHealthState()
				case (.getCableDiagnostics, _):
					return self.getCableDiagnostics()
				default:
					break
				}

				// Just error out if we're in a failed state. It's unclear if it's safe to proceed
				guard EnsemblerService.ensembler?.status != .failed else {
					return EnsemblerResponse(
						result: false,
						error: """
						Ensembler is in unrecoverable state, must reboot to proceed: \
						\(String(describing: EnsemblerServiceError.ensemblerInFailedState))
						"""
					)
				}

				switch (request, controlAllowed) {
				// Read-only functionality
				case (.getMaxBuffersPerKey, _):
					return self.getMaxBuffersPerKey()
				case (.getMaxSecsPerKey, _):
					return self.getMaxSecsPerKey()
				// Control functionality
				case (.activate, true):
					return self.activate()
				case (.decryptData(let data), _):
					return self.decrypt(data: data)
				case (.encryptData(let data), _):
					return self.encrypt(data: data)
				case (.rotateSharedKey, _):
					return self.rotateSharedKey()
				case (.sendTestMessage(let destination), true):
					return self.sendTestMessage(destination: destination)
				case (_, false):
					// We don't differentiate an unentitled caller with an unknown message at
					// this point
					EnsemblerService.logger.error(
						"Potentially unentitled request: \(String(describing: request))"
					)
					let error = EnsemblerServiceError.unauthorizedOrUnknownOperation
					return EnsemblerResponse(result: false, error: String(describing: error))
				case (_, true):
					// We have to assume that you're allowed to do this thing we don't handle
					return EnsemblerResponse(
						result: false,
						error: String(describing: EnsemblerServiceError.notImplemented)
					)
				}
			} catch {
				EnsemblerService.logger.error(
					"Failed to decode message: \(String(reflecting: message))"
				)
				return EnsemblerResponse(result: false, error: String(describing: error))
			}
		}

		func handleCancellation(error: XPCRichError) {
			EnsemblerService.logger.debug("Received session cancellation: \(error)")
		}
	}

	static func listen() async throws {
		do {
			_ = try XPCListener(service: kEnsemblerServiceName) { request in
				request.accept { _ in
					SessionHandler()
				}
			}
            await EnsembleWatchdogService.activate()
            await Task.suspendIndefinitely()
		} catch {
			EnsemblerService.logger.error("Failed to create listener, error: \(error)")
			throw EnsemblerServiceError.serviceInitializationFailure(error: error)
		}
	}
}
