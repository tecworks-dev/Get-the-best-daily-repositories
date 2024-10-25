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

//  Copyright © 2024 Apple, Inc. All rights reserved.
//

import ArgumentParserInternal
import CoreAnalytics
import Foundation
import System

extension CLI.InstanceCmd {
    struct InstanceCreateCmd: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "create",
            abstract: "Create a new Virtual Research Environment instance.",
            discussion: """
            The standard flow for this command uses the configuration and assets provided
            within release metadata (using the --release option). After completing successfully,
            a VM has been created and restored with the configured OS image and left stopped.
            It's also possible to create a VRE instance using --osimage if --release is not used.

            Use the instance 'configure' command to manage individual cryptexes and replace
            darwin-init configuration.
            """
        )

        @OptionGroup var globalOptions: CLI.globalOptions
        @OptionGroup var instanceOptions: CLI.InstanceCmd.options

        @Option(name: [.customLong("name"), .customShort("N")],
                help: "VRE instance name.",
                transform: { try CLI.validateVREName($0) })
        var vreName: String

        @Option(name: [.customLong("release"), .customShort("R")],
                help: "SW Release Log index, or path to release .json or .protobuf file.",
                completion: .file())
        var release: String?

        // SW releases are selected by <release> indexnum for given <logEnvironment>
        @Option(name: [.customLong("environment"), .customShort("E")],
                help: ArgumentHelp("SW Transparency Log environment.",
                                   visibility: .customerHidden))
        var logEnvironment: TransparencyLog.Environment = CLIDefaults.ktEnvironment

        @Option(name: [.customLong("osimage"), .customShort("O")],
                help: "Alternate Private Cloud Compute OS image path.",
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var osImage: String? // restore image

        @Option(name: [.customLong("variant")],
                help: """
                Specify variant for OS installation. (values: \(CLI.osVariants.joined(separator: ", ")).
                default: \(CLI.defaultOSVariant))
                """,
                transform: { try CLI.validateOSVariant($0) })
        var osVariant: String?

        @Option(name: [.customLong("variant-name")],
                help: ArgumentHelp("Specify variant-name for OS installation.", visibility: .customerHidden))
        var osVariantName: String?

        @Option(name: [.customLong("fusing")],
                help: ArgumentHelp("Specify VRE instance fusing.", visibility: .customerHidden),
                transform: { try CLI.validateFusing($0) })
        var fusing: String?

        @Option(name: [.customLong("macaddr"), .customLong("mac")],
                help: "Specify network MAC address for VRE [xx:xx:xx:xx:xx:xx].",
                transform: { try CLI.validateMACAddress($0) })
        var macAddr: String?

        @Option(name: [.customLong("boot-args"), .customShort("B")],
                help: "Specify VRE boot-args (research variant only).",
                transform: { try CLI.validateNVRAMArgs($0) })
        var bootArgs: VRE.NVRAMArgs?

        @Option(name: [.customLong("nvram")],
                help: ArgumentHelp("Specify VRE nvram args.", visibility: .customerHidden),
                transform: { try CLI.validateNVRAMArgs($0) })
        var nvramArgs: VRE.NVRAMArgs?

        @Option(name: [.customLong("rom")],
                help: ArgumentHelp("Path to iBoot ROM image for VRE.", visibility: .customerHidden),
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var romImage: String?

        @Option(name: [.customLong("vseprom")],
                help: ArgumentHelp("Path to vSEP ROM image for VRE.", visibility: .customerHidden),
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var vsepImage: String?

        @Option(name: [.customLong("http-endpoint")],
                help: "Bind built-in HTTP service to <addr>[:<port>] or 'none' (default: automatic)",
                transform: { try CLI.validateHTTPService($0) })
        var httpService: VRE.HTTPServiceDef = .init(enabled: true)

        @Option(name: [.customShort("K"), .customLong("kernelcache")],
                help: "Custom kernel cache for VRE.",
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var kernelCache: String?

        @Option(name: [.customShort("S"), .customLong("sptm")],
                help: "Custom SPTM for VRE.",
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var sptm: String?

        @Option(name: [.customShort("M"), .customLong("txm")],
                help: "Custom TXM for VRE.",
                completion: .file(),
                transform: { try CLI.validateFilePath($0) })
        var txm: String?

        func validate() throws {
            if osVariant != nil && osVariantName != nil {
                throw ValidationError("Only one of --variant or --variant-name can be specified.")
            }
        }

        func run() async throws {
            CLI.setupDebugStderr(debugEnable: globalOptions.debugEnable)
            CLI.logger.log("create VRE \(vreName, privacy: .public)")

            Main.printHardwareRecommendationWarningIfApplicable()

            guard !VRE.exists(vreName) else {
                throw CLIError("VRE '\(vreName)' already exists")
            }

            // Synthesize all of the inputs to create a recipe for this instance.
            // Any cryptexes specified in the darwin-init must be accessible locally
            // prior to creating the instance (we will link/copy these to the instance
            // directory).

            var darwinInit = DarwinInitHelper() // start with "empty" darwin-init
            var releaseAssets: [CryptexSpec] = []
            var releaseID = "-" // glean from release metadata (if available)

            if let release { // --release argument specified
                var releaseMetadata: SWReleaseMetadata
                var altAssetsDir: FilePath? = nil

                if let relIndex = UInt64(release) { // index number: look for release.json file (from "releases download")
                    CLI.logger.log("lookup release index \(relIndex, privacy: .public)")
                    if logEnvironment != .production {
                        CLI.logger.log("using KT environment: \(logEnvironment.rawValue)")
                    }

                    var assetHelper: AssetHelper
                    do {
                        assetHelper = try AssetHelper(directory: CLIDefaults.assetsDirectory.path)
                    } catch {
                        throw CLIError("\(CLIDefaults.assetsDirectory.path): \(error)")
                    }

                    guard let relMD = assetHelper.loadRelease(
                        index: relIndex,
                        logEnvironment: logEnvironment
                    ) else {
                        throw CLIError("SW Release info not found for index \(relIndex): downloaded?")
                    }

                    releaseMetadata = relMD
                } else if release.hasSuffix(".json") { // external release.json file
                    do {
                        releaseMetadata = try SWReleaseMetadata(from: FileManager.fileURL(release))
                    } catch {
                        throw CLIError("cannot load \(release): \(error)")
                    }

                    altAssetsDir = FilePath(release).removingLastComponent()
                } else if release.hasSuffix(".pb") || release.hasSuffix(".protobuf") { // external protobuf file
                    do {
                        let pbufdata = try Data(contentsOf: FileManager.fileURL(release))
                        releaseMetadata = try SWReleaseMetadata(data: pbufdata)
                    } catch {
                        throw CLIError("cannot load \(release): \(error)")
                    }
                } else {
                    throw ValidationError("must provide release index or release.json file pathname")
                }

                releaseAssets = try extractAssetsFromReleaseMetadata(releaseMetadata,
                                                                     altAssetSourceDir: altAssetsDir)

                guard let releaseDarwinInit = releaseMetadata.darwinInit else {
                    throw CLIError("unable to parse darwin-init config from release info")
                }

                darwinInit = releaseDarwinInit
                if let relid = releaseMetadata.releaseHash {
                    releaseID = relid.hexString
                }
            }

            var osVariant = self.osVariant
            if let osImage {
                if osVariant == nil || osVariant!.isEmpty, osVariantName == nil || osVariantName!.isEmpty {
                    osVariant = CLI.defaultOSVariant
                    print("Using default OS variant: \(osVariant!)")
                    CLI.logger.log("using default OS variant: \(osVariant!, privacy: .public)")
                }

                if releaseAssets.count > 0 {
                    // update ASSET_TYPE_OS with image/variant ultimately used
                    try releaseAssets.append(CryptexSpec(
                        path: osImage,
                        variant: osVariantName ?? osVariant ?? "Unknown",
                        assetType: SWReleaseMetadata.assetTypeName(.os)
                    ))
                }
            }

            var vre = try VRE(
                name: vreName,
                releaseID: releaseID,
                httpService: httpService,
                vrevmPath: instanceOptions.vrevmPath
            )

            var fusing = self.fusing
            if fusing == nil {
                fusing = CLI.internalBuild ? "dev" : "prod"
            }
            CLI.logger.log("VM fusing: \(fusing!, privacy: .public)")

            let vmConfig = VRE.VMConfig(
                osImage: osImage,
                osVariant: osVariant,
                osVariantName: osVariantName,
                fusing: fusing!,
                macAddr: macAddr,
                bootArgs: bootArgs,
                nvramArgs: nvramArgs,
                romImagePath: romImage,
                vsepImagePath: vsepImage,
                kernelCachePath: kernelCache,
                sptmPath: sptm,
                txmPath: txm
            )

            // create the VRE instance (and underlying VM) -- a restore is also performed on the
            //  new VM using the recipe we've crafted above from the input.
            do {
                try vre.create(
                    vmConfig: vmConfig,
                    darwinInit: darwinInit,
                    instanceAssets: releaseAssets
                )
            } catch {
                // if VRE/VM creation failed, attempt to save a copy of the vrevm logs
                let createErr = error
                if let savedLogs = try? CLI.copyVMLogs(vre: vre) {
                    print("\nCopy of the VRE VM logs stored under: \(savedLogs.path)")
                }

                try? vre.remove()
                throw createErr
            }

            AnalyticsSendEventLazy("com.apple.securityresearch.pccvre.restored") {
                var eventReleaseID = releaseID
                if eventReleaseID.isEmpty || eventReleaseID == "-" {
                    eventReleaseID = "unknown"
                }

                return [
                    "version": eventReleaseID as NSString,
                ]
            }
        }

        // extractAssetsFromReleaseMetadata parses release metadata (json file) to obtain list of
        //  assets (os image, cryptexes, and types) along with their qualified pathname (relative to
        //  the json file), verified to exist
        private func extractAssetsFromReleaseMetadata(
            _ releaseMetadata: SWReleaseMetadata,
            altAssetSourceDir: FilePath? = nil
        ) throws -> [CryptexSpec] {
            // Extract the assets specified by the release metadata. For each asset, validate
            // that we can find it and determine the local path where it can be found.
            var releaseAssets: [CryptexSpec] = []
            if let osAsset = releaseMetadata.osAsset() {
                do {
                    let assetPath = try CLI.expandAssetPath(osAsset,
                                                            altAssetSourceDir: altAssetSourceDir)
                    try releaseAssets.append(CryptexSpec(
                        path: assetPath.string,
                        variant: osAsset.variant,
                        assetType: SWReleaseMetadata.assetTypeName(osAsset.type)
                    ))

                    CLI.logger.log("OS release asset: \(assetPath.string, privacy: .public)")
                } catch {
                    throw CLIError("asset '\(osAsset.url)' in release: \(error)")
                }
            }

            if let cryptexAssets = releaseMetadata.cryptexAssets() {
                for asset in cryptexAssets.values {
                    do {
                        let assetPath = try CLI.expandAssetPath(asset,
                                                                altAssetSourceDir: altAssetSourceDir)
                        try releaseAssets.append(CryptexSpec(
                            path: assetPath.string,
                            variant: asset.variant,
                            assetType: SWReleaseMetadata.assetTypeName(asset.type)
                        ))

                        CLI.logger.log("cryptex release asset: \(assetPath.string, privacy: .public)")
                    } catch {
                        throw CLIError("asset '\(asset.url)' in release: \(error)")
                    }
                }
            }

            if let toolsAsset = releaseMetadata.hostToolsAsset() {
                do {
                    let assetPath = try CLI.expandAssetPath(toolsAsset,
                                                            altAssetSourceDir: altAssetSourceDir)
                    try releaseAssets.append(CryptexSpec(
                        path: assetPath.string,
                        variant: toolsAsset.variant,
                        assetType: SWReleaseMetadata.assetTypeName(toolsAsset.type)
                    ))

                    CLI.logger.log("host tools release image: \(assetPath.string, privacy: .public)")
                } catch {
                    throw CLIError("asset '\(toolsAsset.url)' in release: \(error)")
                }
            }

            return releaseAssets
        }
    }
}
