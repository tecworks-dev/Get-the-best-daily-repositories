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
//  DInitConfigLoader.swift
//  darwin-init
//

import ArgumentParserInternal
import Foundation
import IOKit
import os
import System
import RemoteServiceDiscovery

enum DInitConfigLoader {
    /// Loads a config from the system or the provided source.
    static func load(from source: DInitConfigSource? = nil) async throws -> DInitLoadedConfig {
        do {
            if let source = source {
                let config = try await load(from: source)
                return DInitLoadedConfig(arguments: nil, config: config)
            } else {
                return try await loadSystem()
            }
        } catch {
            throw Error.unableToLoad(source, error)
        }
    }

    /// Loads a default set of config
    ///
    /// If anything that need to be done automatically, either on internal, or
    /// for any specific hardware, it should land in the default config. The
    /// default config will be picked up when running
    /// `darwin-init apply --system` and will be merged with system config when
    /// applying. The keys in system config take priority during a conflict.
    private static func loadDefaultConfig() -> DInitConfig? {
        var config: DInitConfig? = nil

        if Computer.isComputeNode() {
            logger.debug("Setting default compute name and local hostname for compute node")
            if let carrier = shim_MGQOceanComputeCarrierID() as? Int,
                let soc = shim_MGQOceanComputeCarrierSlot() as? Int {
                config = DInitConfig(
                    computerName: "soc-\(carrier)-\(soc)",
                    localHostName: "soc-\(carrier)-\(soc)")
            } else {
                logger.error("Failed to copy carrier / soc information")
            }
        }

        return config
    }

    private static func loadSystem() async throws -> DInitLoadedConfig {
        var nvramConfigRaw: DInitNVRAMConfig?
        // On internal variants of darwin, first attempt to load the system
        // configuration from NVRAM.
        if Computer.allowsInternalSecurityPolicies() || Computer.isVM() {
            logger.debug("Attempting to load configuration from NVRAM")
            nvramConfigRaw = try loadNVRAMConfig()
        }

        // If the device is a BMC and does not have a NVRAM configuration,
        // fall back to using the default data center config server.
        if Computer.isBMC() && nvramConfigRaw == nil {
            logger.log("Using default data center bootstrap server")
            nvramConfigRaw = DInitNVRAMConfig(
                source: .left(.network(kDInitDataCenterBootstrapServer)))
        }
        
        if Computer.isComputeNode() && nvramConfigRaw == nil {
            logger.log("Loading config from remote service")
            let deviceTypeDesc = String(cString: remote_device_type_get_description(REMOTE_DEVICE_TYPE_COMPUTE_CONTROLLER))
            nvramConfigRaw = DInitNVRAMConfig(source: .left(.remoteService(deviceTypeDesc)))
        }

        // If no system configuration was found, error and exit early.
        guard let nvramConfigRaw = nvramConfigRaw else {
            throw Error.missingSystemConfiguration
        }

        let nvramConfig: DInitConfig
        switch nvramConfigRaw.source {
        case let .left(source):
            nvramConfig = try await load(from: source)
        case let .right(_config):
            nvramConfig = _config
        }
        logger.log("Successfully loaded configuration")

        // Merge the default config and the NVRAM config. If there's a config
        // defined in both places, take the one from the NVRAM config.
        let config: DInitConfig
        if let defaultConfig = loadDefaultConfig() {
            logger.log("Found default config, merging with NVRAM config...")
            config = try nvramConfig.merging(defaultConfig, uniquingKeysWith: { first, _ in first })
        } else {
            config = nvramConfig
        }

        return DInitLoadedConfig(
            arguments: nvramConfigRaw.arguments,
            config: config)
    }

    private static func load(from source: DInitConfigSource) async throws -> DInitConfig {
        logger.debug("Loading configuration from \(source)")
        let data: Data
        switch source {
        case .standardInput:
            data = try FileHandle.standardInput.readToEnd() ?? Data()
        case let .json(json):
            data = json.data(using: .utf8) ?? Data()
        case let .network(url):
            // Retry every thirty seconds for three minutes to ensure
            // slow network initialization doesn't become
            // a fatal error
            data = try await loadNetworkConfig(from: url, retries: 6, backoff: 30)
        case let .fileSystem(path):
            data = try path.loadData()
        case let .ec2imds(key):
            data = try await EC2MetadataService.fetchUserData(for: key)
        case let .remoteService(deviceTypeDesc):
            data = try loadRemoteServiceConfig(from: deviceTypeDesc)
        }

        let config = try decode(DInitConfig.self, from: data, source: source)
        logger.log("Successfully loaded configuration from \(source)")
        return config
    }
    
    private static func loadNetworkConfig(from url: URL, retries: Int, backoff: UInt32) async throws -> Data {
        let identity = DInitDeviceIdentity.shared
        logger.info("Requesting darwin-init configuration for \(identity) from \(url).")
        return try await Network.post(identity, to: url, attempts: retries, backoff: .fixed(.seconds(backoff)))
    }

    /// Loads a DarwinInitNVRAMConfig from NVRAM.
    ///
    /// Configurations in NVRAM can come from multiple variables in a variety of
    /// formats. `loadNVRAMConfig()` unifies this loading process.
    /// `loadNVRAMConfig()` first attempts to read data from the `darwin-init`
    /// variable, however if no data is found will fall back to reading data
    /// from the legacy `darwininit` variable. After successfully reading from
    /// either variable, `loadNVRAMConfig()` base64 decodes the data if needed.
    /// At this point the data is expected to be valid UTF8 JSON. Finally,
    /// `loadNVRAMConfig()` attempts to deserialize the data into a
    /// `DInitNVRAMConfig`. If this fails, `loadNVRAMConfig()` falls back to
    /// deserializing the data into a `DInitConfig`.
    ///
    ///  - Important:
    /// The fall back paths are purely included for compatibility with existing
    /// systems and will be removed in the future.
    private static func loadNVRAMConfig() throws -> DInitNVRAMConfig? {
        let nvram = try NVRAM()

        var key = kDInitNVRAMConfigKey
        var data: Data
        do {
            data = try nvram.getData(forKey: key)
        } catch NVRAM.Error.unableToGetDataForKey(_, shim_kIOReturnNoResources()) {
            logger.warning("No data contained in the \(key) NVRAM variable")

            do {
                // For compatibility with existing systems, attempt to load the
                // configuration from the legacy NVRAM key.
                key = kDInitNVRAMConfigLegacyKey
                data = try nvram.getData(forKey: key)
                logger.warning("Use of the \(key) NVRAM variable is deprecated and will be removed in the future, please use \(kDInitNVRAMConfigKey) instead")
            } catch NVRAM.Error.unableToGetDataForKey(_, shim_kIOReturnNoResources()) {
                // No value was set in both kDInitNVRAMConfigKey and
                // kDInitNVRAMConfigLegacyKey. Return nil to indicate there is
                // no NVRAM configuration to load.
                logger.debug("No configuration found in NVRAM")
                return nil
            } catch {
                // Throw the original error
                throw NVRAM.Error.unableToGetDataForKey(kDInitNVRAMConfigKey, shim_kIOReturnNoResources())
            }
        }

        if let _data = Data(base64Encoded: data) {
            logger.debug("Data from \(key) NVRAM variable was recognized as base64")
            data = _data
        }

        logger.debug("Loaded raw config from NVRAM: \(data))")

        do {
            return try decode(DInitNVRAMConfig.self, from: data, source: nil)
        } catch let err {
            do {
                logger.warning("Data from \(key) NVRAM variable is not a valid `DInitNVRAMConfig`, \(err.localizedDescription)")
                let config = try decode(DInitConfig.self, from: data, source: nil)
                logger.warning("Data from \(key) NVRAM variable is encoded as `DInitConfig`, this use is deprecated and will be removed in the future, please use the `DInitNVRAMConfig` instead")
                return DInitNVRAMConfig(arguments: nil, source: .right(config))
            } catch { throw err }
        }
    }
    
    private static func loadRemoteServiceConfig(from deviceTypeDesc: String) throws -> Data {
        let queue = DispatchQueue(label: "remote_device_browse")
        let semaphore = DispatchSemaphore(value: 0)
        var configData: Data = Data()
        var _device: remote_device_t? = nil
        
        let deviceType = remote_device_type_parse(deviceTypeDesc)
        guard deviceType != REMOTE_DEVICE_TYPE_INVALID_OR_UNKNOWN else {
            throw RemoteDeviceTypeError.invalidDeviceTypeDesc(deviceTypeDesc)
        }

        var flag = atomic_flag()

        let browser = remote_device_start_browsing(deviceType, queue) { device, canceling in
            if canceling {
                return
            }

            remote_device_set_connected_callback(device!, queue) { device in
                if atomic_flag_test_and_set_explicit(&flag, memory_order_relaxed) {
                    // timed out, bail
                    return
                }
                _device = device
                semaphore.signal()
            }
        }
        defer {
            remote_device_browser_cancel(browser)
        }

        switch semaphore.wait(timeout: DispatchTime.now() + .seconds(120)) {
        case .success:
            let deviceName = String(cString: remote_device_get_name(_device!))
            logger.info("remote device found: \(deviceName)")
        case .timedOut:
            if atomic_flag_test_and_set_explicit(&flag, memory_order_relaxed) {
                // the callback has been fired but not signaled yet.
                // it's safe to wait for that forever here
                semaphore.wait()
            } else {
                logger.error("timeout waiting for remote device to show up")
                throw Error.loadTimeout(.remoteService(deviceTypeDesc))
            }
        }

        configData = try DInitRemoteService.fetchConfig(from: _device!)

        logger.debug("Raw data got from remote service: \(configData)")
        return configData
    }

    private static func decode<T>(_ : T.Type = T.self, from data: Data, source: DInitConfigSource?) throws -> T where T: Decodable {
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch let error as DecodingError {
            throw Error.unableToDecodeData(source, error)
        }
    }
}

extension DInitConfigLoader {
    indirect enum Error: Swift.Error {
        case missingSystemConfiguration
        case unableToLoad(DInitConfigSource?, Swift.Error)
        case unableToDecodeData(DInitConfigSource?, DecodingError)
        case configRejected(String, Int32)
        case loadTimeout(DInitConfigSource)
    }
}

extension DInitConfigLoader.Error: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .missingSystemConfiguration:
            return "No system configuration found."
        case let .unableToLoad(source, error):
            return "Unable to load configuration data from \(source?.description ?? "system"): \(error.localizedDescription)"
        case let .unableToDecodeData(source, error):
            var description = "Unable to decode configuration data from \(source?.description ?? "system"): "
            switch error {
            case let .typeMismatch(_, context),
                let .valueNotFound(_, context),
                let .keyNotFound(_, context),
                let .dataCorrupted(context):
                description += context.debugDescription
            @unknown default:
                description += error.localizedDescription
            }
            return description
        case let .configRejected(source, errorCode):
            return "libsecureconfig rejected configuration data from \(source): \(errorCode)."
        case let .loadTimeout(source):
            return "Timeout loading configuration data from \(source)"
        }
    }
}
