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
//  Computer.swift
//  darwin-init
//

import os
import System
import SystemConfiguration
import MobileGestaltPrivate

enum Computer {
    /// Sets the computerName using SystemConfiguration.
	///
	/// The computer name is used for user displays and is used by bonjour.
    static func set(computerName name: String) async throws {
        logger.log("Setting computerName to \(name)")
        let preferences = try SCPreferences.create()
		try await preferences.withLock(lockRetryLimit: 5) { preferences in
			try preferences.computerName(name)
			try preferences.commit()
			try preferences.apply()
		}
    }

    /// Sets the host name using SystemConfiguration.
	///
	/// The host name can be queried via SystemConfiguration and bsd APIs.
    static func set(hostName name: String) async throws {
        logger.log("Setting hostName to \(name)")
        let preferences = try SCPreferences.create()
		try await preferences.withLock(lockRetryLimit: 5) { preferences in
			try preferences.hostName(name)
			try preferences.commit()
			try preferences.apply()
		}
    }

	/// Sets the local host name using SystemConfiguration.
	///
	/// The local host name is used by bonjour.
	static func set(localHostName name: String) async throws {
		logger.log("Setting localHostName to \(name)")
		let preferences = try SCPreferences.create()
		try await preferences.withLock(lockRetryLimit: 5) { preferences in
			try preferences.localHostname(name)
			try preferences.commit()
			try preferences.apply()
		}
	}

    /// Configures password-less sudo for the admin group.
    ///
    /// - Returns: `true` for success, `false` for failure.
    static func configurePasswordlessSudo() -> Bool {
#if os(macOS)
        logger.log("Configuring password-less sudo for admin group")
        do {
            try kDInitPasswordlessSuodersFile.path.save(
                kDInitPasswordlessSuodersFile.contents
            )
        } catch {
            logger.fault("failed to create sudoers file with error \(error.localizedDescription)")
            return false
        }
        return true
#else
        logger.fault("Password-less sudo has no effect because it is not available on this platform")
        return false
#endif
    }

	/// Set the current locale for all users. Default locale is `en_US_POSIX`
	///
	/// - Returns: true if successful (result can be ignored).
	static func setLocale(_ locale: String = "en_US_POSIX") -> Bool {
		let key = "AppleLocale"
		let value = locale as CFString

		let domain = CFPreferences.Domain(
			applicationId: kCFPreferencesAnyApplication,
			userName: kCFPreferencesCurrentUser,
			hostName: kCFPreferencesAnyHost)
		CFPreferences.set(value: value, for: key, in: domain)

		guard CFPreferences.synchronize(domain: domain) else {
			logger.error("Failed to set AppleLocale to \(locale, privacy: .public): Failed to synchronize preferences.")
			return false
		}
		CFPreferences.flushCaches()

		guard let actualValue = CFPreferences.getValue(for: key, in: domain) else {
			logger.error("Failed to set AppleLocale to \(locale, privacy: .public): Failed persist preferences.")
			return false
		}

		guard CFEqual(actualValue, value) else {
			logger.error("Failed to set AppleLocale to \(locale, privacy: .public): Invalid persisted value.")
			return false
		}

		logger.log("Set AppleLocale to \(locale, privacy: .public).")
		return true
	}
	
	static func getAutomatedDeviceGroup() -> String? {
		guard let group = shim_automatedDeviceGroup() else {
			logger.log("Failed to find an AutomatedDeviceGroup for this device")
			return nil
		}
		logger.log("Found AutomatedDeviceGroup: \(group, privacy: .public)")
		return group
	}
	
	static func setAutomatedDeviceGroup(to group: String) -> Bool {
		shim_setAutomatedDeviceGroup(group)
		let actual = getAutomatedDeviceGroup()
		guard actual == group else {
			logger.error("Failed to set AutomatedDeviceGroup to \(group, privacy: .public)")
			return false
		}
		logger.log("Successfully set AutomatedDeviceGroup to \(group, privacy: .public)")
		return true
	}

    static func uniqueChipID() -> Int? {
        let id = Int(shim_MGQUniqueChipID())
        return id != -1 ? id : nil
    }

    static func serialNumber() -> String? {
        return shim_MGQSerialNumber() as String?
    }

    static func securityDomain() -> Int? {
        let id =  Int(shim_MGQSecurityDomain())
        return id != -1 ? id : nil
    }

    static func boardID() -> Int? {
        let id =  Int(shim_MGQBoardID())
        return id != -1 ? id : nil
    }

    static func chipID() -> Int? {
        let id =  Int(shim_MGQChipID())
        return id != -1 ? id : nil
    }

    static func isComputeNode() -> Bool {
        return hasServiceNamed("manta-c")
    }

    static func isBMC() -> Bool {
        return hasServiceNamed("manta-b")
    }

    static func isVM() -> Bool {
        return MobileGestalt.current.isVirtualDevice
    }

    static func isDarwinCloud() -> Bool {
        return self.releaseType().contains("Darwin Cloud")
    }

    static func buildVersion() -> String? {
        return MobileGestalt.current.buildVersion
    }

    static func releaseType() -> String {
        return MobileGestalt.current.releaseType
    }

    private static func hasServiceNamed(_ name: String) -> Bool {
        var result = false
        let matchingDict = IOServiceNameMatching(name)
        let service = IOServiceGetMatchingService(kIOMainPortDefault, matchingDict)

        if service != IO_OBJECT_NULL {
            result = true
            IOObjectRelease(service)
        }

        return result
    }

    static func allowsInternalSecurityPolicies() -> Bool {
        return shim_allows_internal_security_policies()
    }

    static func lockCryptexes() throws {
        let e = shim_cryptex_lockdown()
        if e != 0 {
            throw Errno(rawValue: e)
        }
    }

    static func defaultUSRAction() -> DInitUSRPurpose? {
        let entry = IORegistryEntryFromPath(kIOMainPortDefault, kIODeviceTreePlane + ":/chosen")
        guard entry > 0 else {
            return nil
        }

        let key = "darwinos-security-environment" as CFString
        guard let property = IORegistryEntryCreateCFProperty(entry, key, kCFAllocatorDefault, 0)  else {
            return nil
        }

        let data = property.takeRetainedValue() as! Data
        let value = data.withUnsafeBytes { raw in
            raw.load(as: UInt32.self)
        }
        logger.log("\(key) is \(value)")

        switch value {
        case 6: // BOOT_ENVIRONMENT_EMBEDDED_DARWINOS
            return .rem
        default:
            return nil
        }
    }

    static func userspaceReboot(_ purposeString: DInitUSRPurpose) throws {
		let purpose: rb3_userreboot_purpose_t_t

		do {
            purpose = try DInitUSRPurpose.determineRebootPurpose(purposeString)
		} catch {
			throw DInitError.userspaceRebootFailed(error, description: "Failed to determine reboot purpose")
		}

        var timeout:mach_timespec_t = mach_timespec(tv_sec: 60, tv_nsec: 0)
        let kr = IOKitWaitQuiet(kIOMainPortDefault, &timeout)
        logger.log("IOKitWaitQuiet returned \(kr)")

        let rc = shim_usr(purpose)
        if rc != 0  {
			throw DInitError.userspaceRebootFailed(Errno(rawValue: rc))
        }
    }

    static func reboot() throws {
        guard 0 == shim_reboot3(RB2_FULLREBOOT) else {
			throw DInitError.rebootFailed(Errno.current)
        }
    }

    static func shutdown() throws {
#if os(macOS)
        if !Subprocess.run(shell: nil, command: "shutdown -h now") {
            throw DInitError.shutdownFailed
        }
#else
        logger.error("Shutdown only available on macOS")
        throw DInitError.shutdownFailed
#endif
    }
}
