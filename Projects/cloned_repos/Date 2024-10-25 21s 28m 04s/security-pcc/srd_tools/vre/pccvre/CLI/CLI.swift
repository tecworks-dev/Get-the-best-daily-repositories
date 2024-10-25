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
import Foundation
import Network
import OSPrivate_os_log
import System

struct CLI: AsyncParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: commandName,
        abstract: "Private Cloud Compute Virtual Research Environment tool.",
        subcommands: [
            LicenseCmd.self,
            ReleaseCmd.self,
            InstanceCmd.self,
            Image4Cmd.self,
            CryptexCmd.self,
            TransparencyLogCmd.self,
            AttestationCmd.self,
        ]
    )

    // commandDir returns directory containing this executable (or ".")
    static var commandDir: FilePath {
        let argv0 = FilePath(Bundle.main.executablePath!).removingLastComponent()
        if argv0.isEmpty {
            return FilePath(".")
        }

        return argv0
    }

    static let logger = os.Logger(subsystem: applicationName, category: "CLI")

    static var internalBuild: Bool { os_variant_allows_internal_security_policies(applicationName) }
    static let defaultOSVariant: String = "customer"

    static var osVariants: [String] {
        let publicVariants: [String] = [
            defaultOSVariant,
            "research",
        ]
        let internalVariants: [String] = [
            "internal-development",
            "internal-debug",
        ]

        return CLI.internalBuild ? publicVariants + internalVariants : publicVariants
    }

    struct globalOptions: ParsableArguments {
        @Flag(name: [.customLong("debug"), .customShort("d")], help: "Enable debugging.")
        var debugEnable: Bool = false
    }

    // setupDebugLogger configures log.debug messages to write to stderr when --debug enabled
    static func setupDebugStderr(debugEnable: Bool = false) {
        guard debugEnable else {
            return
        }

        var previous_hook: os_log_hook_t?
        previous_hook = os_log_set_hook(OSLogType.debug) { level, msg in
            // let msgCStr = os_log_copy_formatted_message(msg)
            if let subsystemCStr = msg?.pointee.subsystem,
               String(cString: subsystemCStr) == applicationName,
               let msgCStr = os_log_copy_decorated_message(level, msg)
            {
                fputs(String(cString: msgCStr), stderr)
                free(msgCStr)
                fflush(stderr)
            }

            previous_hook?(level, msg)
        }
    }

    static func parseURL(_ arg: String) throws -> URL {
        guard let url = URL.normalized(string: arg) else {
            throw ValidationError("invalid url")
        }

        return url
    }

    static func validateFilePath(_ arg: String) throws -> String {
        guard FileManager.isExist(arg, resolve: true) else {
            throw ValidationError("\(arg): not found")
        }

        guard FileManager.isRegularFile(arg, resolve: true) else {
            throw ValidationError("\(arg): not a file")
        }

        return arg
    }

    static func validateCryptexSpec(_ arg: String, relativeDir: String? = nil) throws -> CryptexSpec {
        let parg = arg.split(separator: ":", maxSplits: 2)
        guard parg.count == 2 else {
            throw ValidationError("invalid image spec; must be <variant>:<path>")
        }

        let pathURL: URL
        do {
            pathURL = try FileManager.fullyQualified(String(parg[1]), relative: relativeDir)
        } catch {
            throw ValidationError("\(parg[1]): \(error)")
        }

        return try CryptexSpec(path: pathURL.path, variant: String(parg[0]))
    }

    // validateVREName checks whether arg is a valid VRE instance name (no whitespace)
    static func validateVREName(_ arg: String) throws -> String {
        guard !arg.isEmpty, arg.rangeOfCharacter(from: .whitespaces) == nil else {
            throw ValidationError("invalid VRE name")
        }

        return arg
    }

    static func validateDirectoryPath(_ arg: String) throws -> FilePath {
        guard FileManager.isDirectory(arg) else {
            throw ValidationError("\(arg): not found or not a directory")
        }

        return FilePath(arg)
    }

    static func validateCryptexVariantName(_ arg: String) throws -> String {
        if arg.isEmpty {
            throw ValidationError("invalid variant name provided")
        }

        if arg.count > FILENAME_MAX {
            throw ValidationError("provided variant name exceeds max \(FILENAME_MAX)")
        }
        return arg
    }

    // validateMACAddress checks whether arg in the form of [hh:hh:hh:hh:hh:hh] and not all 00's or ff's
    static func validateMACAddress(_ arg: String) throws -> String {
        let octs = arg.split(separator: ":", maxSplits: 5)
        guard octs.count == 6 else {
            throw ValidationError("invalid MAC address")
        }

        var msum: UInt64 = 0
        for oct in octs {
            guard let oval = UInt8(oct, radix: 16) else {
                throw ValidationError("invalid MAC address")
            }

            msum += UInt64(oval)
        }

        // ensure not all 00's or ff's
        guard msum > 0 && msum < 1530 else {
            throw ValidationError("invalid MAC address")
        }

        return arg.lowercased()
    }

    static func validateFusing(_ arg: String) throws -> String {
        switch arg {
        case "prod": break
        case "dev":
            guard CLI.internalBuild else {
                throw ValidationError("dev fusing not supported")
            }
        default:
            throw ValidationError("specified fusing: \(arg) not supported")
        }

        return arg
    }

    // validateNVramArgs parses arg as a set of whitespace separate NVram args (each of which
    // may be in the form of "key=value" or as simply "key"); returns VRE.nvramArgs map
    static func validateNVRAMArgs(_ arg: String) throws -> VRE.NVRAMArgs {
        var bootArgs: [String: String] = [:]
        for p in arg.components(separatedBy: .whitespacesAndNewlines) {
            let kv = p.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)

            if kv.count > 1 { // typical key=value
                bootArgs[String(kv[0])] = String(kv[1])
            } else { // otherwise, add bare token to args
                bootArgs[String(kv[0])] = ""
            }
        }

        return bootArgs
    }

    // validateHTTPService parses arg as VRE.httpService spec; forms:
    //   'none', '<ipaddr>', '<ipaddr>:<port>' or ':<port>'
    static func validateHTTPService(_ arg: String) throws -> VRE.HTTPServiceDef {
        if arg == "none" {
            return VRE.HTTPServiceDef(enabled: false)
        }

        let httpBind = arg.split(separator: ":", maxSplits: 1)
        switch httpBind.count {
        case 1:
            // parse as IP addr (no port)
            guard let _ = try? validateIPAddress(String(httpBind[0])) else {
                throw ValidationError("invalid http service addr")
            }

            return VRE.HTTPServiceDef(enabled: true, address: String(httpBind[0]))

        case 2:
            // <ip>:<port>
            let addr = httpBind[0] == "" ? nil : String(httpBind[0])
            if let addr {
                guard let _ = try? validateIPAddress(addr) else {
                    throw ValidationError("invalid http service addr")
                }
            }

            // or :<port>
            guard let port = UInt16(String(httpBind[1])), port > 0 else {
                throw ValidationError("invalid http service port")
            }

            return VRE.HTTPServiceDef(enabled: true, address: addr, port: port)

        default:
            throw ValidationError("invalid http service spec")
        }
    }

    // validateIPAddress parses arg as either an IPv4 or IPv6 address
    static func validateIPAddress(_ arg: String) throws -> IPAddress {
        if let ip = IPv4Address(arg) {
            return ip
        }

        if let ip = IPv6Address(arg) {
            return ip
        }

        throw ValidationError("invalid IP address")
    }

    static func validateOSVariant(_ arg: String) throws -> String {
        if !CLI.osVariants.contains(arg) {
            throw ValidationError("invalid variant specified")
        }

        return arg
    }

    // confirmYN outputs "<prompt> (y/n) " and returns true if input starts with 'Y' or 'y'; else false
    static func confirmYN(prompt: String) -> Bool {
        print(prompt, terminator: " (y/n) ")
        if let yn = readLine(strippingNewline: true)?.trimmingCharacters(in: .whitespaces).uppercased() {
            return yn.hasPrefix("Y")
        }

        return false
    }

    // expandAssetPath searches for a SWReleaseMetadata.Asset (representing a release asset) in various
    //  locations - ensuring exists as a regular file - and returns the qualified pathname; if assetURL:
    //  - resemble full/partial pathname ("file" scheme containing a "/") relative to CWD
    //  - whose last component name exists under either altAssetSourceDir or CLIDefaults.assetsDirectory
    static func expandAssetPath(
        _ asset: SWReleaseMetadata.Asset,
        altAssetSourceDir: FilePath? = nil
    ) throws -> FilePath {
        let assetURL = URL(string: asset.url) ?? FileManager.fileURL(asset.url)

        // if resembles a file pathname (full or partial), use in situ (relative to CWD)
        if assetURL.scheme == "file", asset.url.contains("/") {
            do {
                let assetPath = try FileManager.fullyQualified(assetURL,
                                                               relative: FileManager.default.currentDirectoryPath,
                                                               resolve: true)
                guard FileManager.isRegularFile(assetPath) else {
                    throw CLIError("not a file")
                }

                return FilePath(assetPath.path)
            } catch {
                throw CLIError("\(error)")
            }
        }

        let assetName = assetURL.lastPathComponent

        // otherwise, check under altAssetSourceDir
        if let altAssetSourceDir,
           let assetPath = try? FileManager.fullyQualified(
               assetName,
               relative: altAssetSourceDir.string,
               resolve: true
           ),
           FileManager.isRegularFile(assetPath)
        {
            return FilePath(assetPath.path)
        }

        // .. or in pccvre assets folder
        do {
            let assetPath = try FileManager.fullyQualified(
                assetName,
                relative: CLIDefaults.assetsDirectory.path,
                resolve: true
            )

            guard FileManager.isRegularFile(assetPath) else {
                throw CLIError("not a file")
            }

            return FilePath(assetPath.path)
        } catch {
            throw CLIError("\(error)")
        }
    }

    // mountPCHostTools attempts to mount dmgFile (if provided) or the ".hostTools" asset (associated
    //  with a SW Release), expected to contain "tie-vre-cli" and "cloudremotediagctl". A URL of the
    //  mounted set of tools along with a callback pointer used to clean up mounted image(s).
    static func mountPCHostTools(
        vre: VRE,
        dmgFile: String? = nil
    ) throws -> (URL, () -> Void) {
        var toolsDMGPath: URL
        if let dmgFile {
            toolsDMGPath = FileManager.fileURL(dmgFile)
        } else {
            // if no tools DMG patch explicitly provided, look for HOST_TOOLS release asset
            guard let toolsAsset = vre.config.lookupAssetType(type: .hostTools) else {
                throw CLIError("no Host Tools DMG available (and no release asset found)")
            }

            toolsDMGPath = vre.cryptexFile(toolsAsset.file)
        }

        if !FileManager.isRegularFile(toolsDMGPath) {
            throw CLIError("\(toolsDMGPath): file not found")
        }

        CLI.logger.debug("mountPCHostTools: \(toolsDMGPath, privacy: .public)")

        var toolsDMGs: [DMGHelper] = []
        // callback to tidy up mounts/temp dirs
        let unmountCallback = {
            for var dmg in toolsDMGs {
                try? dmg.eject()
            }
        }

        toolsDMGs = try VRE.mountPCHostTools(dmgPath: toolsDMGPath.path)
        guard let toolsMountDir = toolsDMGs.first?.mountPoint else {
            throw CLIError("unable to obtain Host Tools mountpoint")
        }

        CLI.logger.log("mountPCHostTools: mounted on \(toolsMountDir, privacy: .public)")
        return (toolsMountDir, unmountCallback)
    }

    // unpackPCTools mounts either the dmgFile or the ".hostTools" asset (from a SW Release)
    //  and copies the contents (use/ and System/ subdirs) into the "PCTools/" folder of the instance dir.
    //  A path URL to the fully-qualified PCTools/ folder is returned. If "PCTools/" already exists, it is
    //  assumed to already be unpacked and no further action taken.
    static func unpackPCTools(
        vre: VRE,
        dmgFile: String? = nil
    ) throws -> URL {
        if !vre.pcToolsUnpacked {
            CLI.logger.debug("unpackPCTools: not already unpacked; attempting to mount")
            let (toolsMountDir, unmountCallback) = try CLI.mountPCHostTools(vre: vre, dmgFile: dmgFile)

            CLI.logger.log("unpackPCTools: mounted on \(toolsMountDir, privacy: .public); copy into place")
            fputs("Unpacking PCC Host Tools...\n", stderr)
            do {
                try vre.copyPCHostTools(mountPoint: toolsMountDir)
            } catch {
                throw CLIError("unable to copy in Host Tools for instance")
            }

            // done copying: clean up mounts
            CLI.logger.debug("unpackPCTools: running unmountCallback")
            unmountCallback()
        }

        CLI.logger.debug("unpackPCTools: using \(vre.pcToolsDir, privacy: .public)")
        return vre.pcToolsDir
    }

    // copyVMLogs attempts to copy any collected logs for a vrevm VM instance (logs/) to a tempDirectory
    //  (or destDir/) -- typically called prior to wiping VM after a failed "instance create" command;
    //  the destination path is returned if successful
    static func copyVMLogs(
        vre: VRE,
        destDir altDest: String? = nil
    ) throws -> URL {
        guard let vminfo = try? vre.status(),
              let vmBundleDir = vminfo.bundlepath
        else {
            throw CLIError("copyVMLogs: no bundle dir available for VM")
        }

        // obtain list of <vre.name>/logs/subdirs (if any)
        let vmLogsDir = FileManager.fileURL(vmBundleDir).appendingPathComponent("logs")
        let vmLogsSubs: [String]
        do {
            vmLogsSubs = try FileManager.default.contentsOfDirectory(atPath: vmLogsDir.path)
        } catch {
            throw CLIError("\(vmLogsDir.path): not found")
        }

        // setup destination folder (either provided or temp folder)
        var logDest: URL
        if let altDest {
            try FileManager.default.createDirectory(atPath: altDest, withIntermediateDirectories: true)
            logDest = FileManager.fileURL(altDest)
        } else {
            // .../com.apple.security-research.pccvre/logs/<vre.name>/...
            logDest = try FileManager.tempDirectory(subPath: applicationName, "logs", vre.name)
        }

        // copy each log subdir separately (as previous copies likely to be around)
        for logSubDir in vmLogsSubs {
            let logSubDir = vmLogsDir.appendingPathComponent(logSubDir)
            let logMsg = "\(logSubDir.path) -> \(logDest.path)"
            do {
                try FileManager.copyFile(logSubDir, logDest)
                CLI.logger.log("copyVMLogs: \(logMsg, privacy: .public)")
            } catch {
                CLI.logger.error("\(logMsg, privacy: .public): \(error)")
                throw CLIError("failed to copy VM logs")
            }
        }

        return logDest
    }
}

// Defaults provides global defaults from envvars or presets
//   CMDNAME_DEBUG:      Enable debugging
//   CMDNAME_ENV:        SW Transparency Log "environment" (internal only)
//   CMDNAME_ASSETS_DIR: Location to store downloaded release assets
//
private let envPrefix = commandName.uppercased()

enum CLIDefaults {
    static var debugEnable: Bool {
        if let debugEnv = ProcessInfo().environment["\(envPrefix)_DEBUG"] {
            // false if starts with "n(o)", "f(alse)", "0", else true (if set)
            return !debugEnv.lowercased().starts(with: ["n", "f", "0"])
        }

        return false
    }

    static var ktEnvironment: TransparencyLog.Environment {
        if CLI.internalBuild {
            if let envEnv = ProcessInfo().environment["\(envPrefix)_ENV"] {
                if let env = TransparencyLog.Environment(rawValue: envEnv) {
                    return env
                }
            }
        }

        return .production
    }

    static var assetsDirectory: URL {
        if let assetsDirEnv = ProcessInfo().environment["\(envPrefix)_ASSETS_DIR"] {
            return FileManager.fileURL(assetsDirEnv)
        }

        
        if let assetDir = try? FileManager.tempDirectory(subPath: applicationName, "assets") {
            return assetDir
        } else {
            return URL.applicationSupportDirectory
                .appendingPathComponent(applicationName)
                .appendingPathComponent("assets")
        }
    }
}

// CLIError provides general error encapsulation for errors encountered within CLI layer
struct CLIError: Error, CustomStringConvertible {
    var message: String
    var description: String { message }

    init(_ message: String) {
        CLI.logger.error("\(message, privacy: .public)")
        self.message = message
    }
}
