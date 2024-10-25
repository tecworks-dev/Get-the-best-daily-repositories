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
//  Apply.swift
//  darwin-init
//

import ArgumentParserInternal
internal import OSPrivate_os_log
internal import OSPrivate.os.transactionPrivate
import os
import RemoteServiceDiscovery
import System
@_spi(Private) import DarwinInitClient

final class Apply: AsyncParsableCommand {
	struct Failure: Error, CustomStringConvertible {
		var description: String

		init(_ description: String) {
			self.description = description
		}
	}

    static let configuration = CommandConfiguration(
        abstract: "Apply a configuration to the running system",
        discussion: """
			Applies a configuration provided as a command line argument or in the \
			"darwin-init" nvram parameter. Diagnostic output is logged using the \
			"com.apple.darwin-init" subsystem. The darwin-init daemon runs only once.
			""")

    @Argument(help: .init(
        "Configuration to apply",
        discussion: """
			<source> must either be a path to a local file, a url to a networked file, or a \
			json string containing a darwin-init config.
			"""))
    var source: DInitConfigSource?

    @Flag(name: [.short, .long, .customShort("b"), .customLong("boot")],
		  help: .init(
        "Apply the system configuration, used internally when running at boot via launchd",
        discussion: """
			The system configuration is expected to be stored in the "darwin-init" nvram \
			variable. It must contain json data (optionally base64 encoded) with a "source" \
			key containing a nested json encoded darwin-init config, a path to a local file, \
			or a url to a networked file. See darwin-init(7) for more details.
			"""))
    var system: Bool = false

	@Option(name: .shortAndLong,
		help: .init ("Watchdog timeout value",
		discussion: """
			The timeout value for the Apply command can be specified as an \
			integer followed by a unit suffix. For example, a timeout of "5min" \
			is 5 minutes. If the unit suffix is omitted, the timeout value is \
			in seconds. The timeout can also be specified via the "apply-timeout" \
			key in the config itself.
			"""))
	var timeout: Duration?

	@Option(name: .shortAndLong, help: .init("Action to take if a failure occurs",
		 discussion: """
			If a timeout or an error occurs, what action can be taken. This includes: \
			"reboot" and "shutdown" on failure. The failure action "shutdown" is not allowed on BMC. \
			The default is exit unless this machine is a BMC, and then the default failure action is "reboot". \
			This CLI option is overridden by NVRAM arguments or failureActions set from darwin-init config.
			"""))
	var failureAction: DInitFailureAction?

    func validate() throws {
		if let timeout = timeout, timeout <= .zero {
			throw ValidationError("timeout value must be greater than 0")
		}
#if !os(macOS)
		guard failureAction != .shutdown else {
			throw ValidationError("Failure action \"shutdown\" only allowed on macOS")
		}
#endif
    }

    static func validateConfig(policyString: String, config: DInitConfig) throws {
        // Ensure only valid policy is set: customer or carry
        guard let policy = ConfigSecurityPolicy(rawValue: policyString) else {
            throw PrivateCloudOSValidationError("Unknown \(DInitConfig.CodingKeys.configSecurityPolicy.rawValue): \(policyString)")
        }
        let version = config.configSecurityPolicyVersion
        switch policy {
        case .carry:
            var validator = CarryValidator(policy: policyString, requestedVersion: version, config: config)
            try validator.validate()
        case .customer:
            var validator = CustomerValidator(policy: policyString, requestedVersion: version, config: config)
            try validator.validate()
        }
    }

	func run() async throws {
		// Create an os_transaction that lives for the entire lifetime of the apply.
		let applyTransaction = os_transaction_create("com.apple.darwin-init.apply")
		// TODO: rdar://100902294 Use withExtendedLifetime
		defer { _fixLifetime(applyTransaction) }

		// Check if configuration has already been applied by looking for cookie
		// Do this before running apply so this sort of failure doesn't trigger
		// a failure action. If the failure action is reboot and we reboot after
		// this failure, the device will enter a boot loop as nothing removes the
		// file at kDInitDoneFilepath if ephemeral data mode is turned off.
		if try system && kDInitDoneFilepath.fileExists() {
			logger.warning("darwin-init has already run, exiting")
			return
		}

		if system, #available(macOS 13.3, iOS 16.4, tvOS 16.4, watchOS 9.4, *) {
			let file = fopen("/dev/console", "a")
			if let file = file {
				_ = os_log_set_hook(OSLogType.info) { type, msg in
					let msgstr = os_log_copy_decorated_message(type, msg)
					fputs(msgstr, file)
					free(msgstr)
				}
			} else {
				logger.error("Failed to fopen(/dev/console). Logging hook was NOT set.")
			}
		}

		// Set default failure action from CLI, then fall back to platform default.
		failureAction = failureAction
			?? (Computer.isBMC() ? .reboot : .exit)

		// Apply the configuration with a deadline

		do {
			try await apply()
		} catch {
			if system {
				let status = DarwinInitApplyStatus.failure(info: DarwinInitApplyFailureInfo(errorString: "\(error)"))
				try status.save()
			}

			guard let failureAction = failureAction, failureAction != .exit else {
				logger.error("darwin-init apply failed: \(error). No failure action specified, exiting...")
				throw ExitCode.failure
			}
			
			let timeoutBeforeSleep = if case DInitConfigLoader.Error.unableToLoad(_, _) = error, Computer.isBMC() {
				kDInitSleepBeforeFailureActionOnLoadFailure
			} else {
				kDInitDefaultSleepBeforeFailureAction
			}
			
			logger.error("darwin-init apply failed: \(error). Sleeping for \(timeoutBeforeSleep) before running failure action.")
			try await Task.sleep(for: timeoutBeforeSleep)

			// Don't reboot / shutdown if there's host attached to the BMC at debug port.
			// Check this after sleeping for some time to avoid racing with RSD
			if Computer.isBMC() {
				if remote_device_copy_unique_of_type(REMOTE_DEVICE_TYPE_NCM_HOST) != nil {
					logger.log("Not running the failure action because a host is attached on the debug port.")
					throw ExitCode.failure
				}
			}
			try? failureAction.execute()
			throw ExitCode.failure
		}
	}

	@Sendable
	func apply() async throws {
		// Disable BMC and Darwin Cloud SSH login on physical devices using ESC credentials as early as possible
		if (Computer.isBMC() || (!Computer.isVM() && Computer.isDarwinCloud())) && !EngineeringSSHCA.disableGlobalAccess() {
			logger.error("darwin-init failed to disable ESC SSH")
		}

		let system = self.system

		// Load config from source
		let loadedConfig = try await DInitConfigLoader.load(from: source)

		// Config timeout takes precedence over --timeout CLI argument
		let timeout = try loadedConfig.config.applyTimeout ?? timeout ?? .seconds(60) * 30

		logger.info("Effective timeout is \(timeout)")
		try await withDeadline(.now + timeout, clock: .continuous) {
			/// Set locale to `en_US_POSIX`
			if !Computer.setLocale() {
				logger.error("Unable to set locale")
			}

			if system {
				logger.log("Committing configuration.")
				try LibSecureConfig.commitConfig(loadedConfig.config)
			}

			// Override failureAction with values loaded from config or NVRAM
			if let failureAction = (loadedConfig.config.resultConfig?.failureAction ?? 
									loadedConfig.arguments?.failureAction)  {
				self.failureAction = failureAction
			}

			// Create persist storage
			do {
				try kDInitPersistStorage.createDirectory()
			} catch Errno.fileExists {
				logger.info("darwin-init persist storage already created")
			}

			// Serialize config
			var prettyJson = try loadedConfig.config.jsonString()

			// validate entire config under specified policy and only apply if valid
			if let policyString = loadedConfig.config.configSecurityPolicy {
				logger.log("Validating config under \(policyString) policy: \(prettyJson)")
				do {
					try Apply.validateConfig(policyString: policyString, config: loadedConfig.config)
				} catch {
					throw Failure("Validation failed, will NOT apply config: \(error)")
				}
				logger.log("darwin-init config is valid with respect to \(policyString) policy.")
			}

			// Apply config
			logger.log("Applying configuration: \(prettyJson)")
			var result = await loadedConfig.config.apply()
			// If we're running at boot and made it this far, then secure-config params registration succeeded
			if system {
				result.secureConfigParameters = loadedConfig.config.secureConfigParameters
			}

			// Serialize applied config
			prettyJson = try result.jsonString()
			logger.log("Applied configuration: \(prettyJson)")

			let fullyApplied = (result == loadedConfig.config)

			if system {
				// Write applied config to cookie file
				try kDInitDoneFilepath.save(prettyJson)
			}

			// Check if applied config matches expected config
			guard fullyApplied else {
				throw Failure("Failed to fully apply config")
			}

			// Last opportunity to write 'success' before we reboot.
			// Any failures to reboot should still be caught and written out.
			if system {
				let status = DarwinInitApplyStatus.success
				try status.save()
			}

			// Run rebootAfterSetup if needed
			if loadedConfig.config.rebootAfterSetup == true {
				try Computer.reboot()
			} else if let purpose = loadedConfig.config.USRAfterSetup {
                if purpose == .empty || purpose == .none {
					logger.log("Skipping userspace-reboot by request of configuration")
				} else {
					try Computer.userspaceReboot(purpose)
				}
			} else if system, let purpose = Computer.defaultUSRAction() {
				try Computer.userspaceReboot(purpose)
			}
		}
	}
}
