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

// VRE provides implementation of instances specific to Transparency Log releases.
//  This sits atop VMs provided through the "vrevm" utility.

import Foundation
import os
import System

struct VRE {
    typealias NVRAMArgs = [String: String]
    typealias VMConfig = VRE.VM.Config
    typealias VMStatus = VRE.VM.Status

    let name: String
    var config: VRE.Config
    let vm: VRE.VM

    static let logger = os.Logger(subsystem: applicationName, category: "VRE")

    // security policy keys that must be suppressed when enabling certain researcher features (such as ssh)
    private static let securityPolicyDarwinInitKeys = ["config-security-policy",
                                                       "config-security-policy-version"]

    // applicationDir contains the full path of the "~/Library/Application Support/" dir for this utility
    static var applicationDir: URL {
        URL.applicationSupportDirectory.appendingPathComponent(applicationName)
    }

    // instancesDir contains the full path of the VRE instances catalog
    static var instancesDir: URL {
        VRE.applicationDir.appending(path: "instances")
    }

    // instanceDir contains the full path of the named instance bundle:
    //  - instance.plist config
    //  - darwin-init.json
    //  - disk images/cryptexes used to (re)create the instance VM
    var instanceDir: URL { VRE.instanceDir(name) }
    static func instanceDir(_ name: String) -> URL {
        VRE.instancesDir.appending(path: name)
    }

    // instanceFile contains the full path of the named instance config plist
    static let instanceConfigFilename = "instance.plist"
    var instanceConfigFile: URL { VRE.instanceConfigFile(name) }
    static func instanceConfigFile(_ name: String) -> URL {
        VRE.instanceDir(name).appending(path: instanceConfigFilename)
    }

    // darwinInit contains the full path of the working darwin-init.json file;
    //  it is passed into the VM upon starting
    static let darwinInitFilename = "darwin-init.json"
    var darwinInitFile: URL { VRE.darwinInitFile(name) }
    static func darwinInitFile(_ name: String) -> URL {
        VRE.instanceDir(name).appending(path: darwinInitFilename)
    }

    // cryptexFile returns a pathname of a cryptex image file relative to the VRE instance
    //  (only the last component of path is used)
    func cryptexFile(_ path: String) -> URL { VRE.cryptexFile(name, path: path) }
    static func cryptexFile(_ name: String, path: String) -> URL {
        return VRE.instanceDir(name).appending(path: FileManager.fileURL(path).lastPathComponent)
    }

    // pcToolsDir is the designated directory containing an unpacked Private Cloud (Host) Tools,
    //  used for making inference requests (tie-vre-cli)
    static let pcToolsDirname = "PCTools"
    var pcToolsDir: URL { VRE.pcToolsDir(name) }
    static func pcToolsDir(_ name: String) -> URL {
        VRE.instanceDir(name).appending(path: pcToolsDirname)
    }

    // pcToolsUnpacked is true if PCTools/ exists (implying TIE inference tools unpacked)
    var pcToolsUnpacked: Bool { FileManager.isDirectory(pcToolsDir) }

    // protectedInstanceFiles is list of entries to guard against overwriting (with cryptex images)
    static let protectedInstanceFiles = [instanceConfigFilename, darwinInitFilename, pcToolsDirname]

    // exists returns true if the instanceDir exists
    static func exists(_ name: String) -> Bool {
        FileManager.isExist(VRE.instanceDir(name))
    }

    // initialize new VRE instance
    init(
        name: String,
        releaseID: String,
        httpService: VRE.HTTPServiceDef,
        vrevmPath: String? = nil
    ) throws {
        self.name = name
        config = VRE.Config(
            name: name,
            releaseID: releaseID,
            httpService: httpService
        )

        do {
            if let vrevmPath {
                vm = try VRE.VM(name: name, vrevmPath: vrevmPath)
            } else {
                vm = try VRE.VM(name: name)
            }
        } catch {
            throw VREError("VRE instance: \(error)")
        }
    }

    // load VRE instance from existing instance.plist file
    init(
        name: String,
        vrevmPath: String? = nil
    ) throws {
        self.name = name
        config = try VRE.Config(
            contentsOf: VRE.instanceConfigFile(name)
        )

        if let vrevmPath {
            vm = try VRE.VM(name: name, vrevmPath: vrevmPath)
        } else {
            vm = try VRE.VM(name: name)
        }
    }

    // create writes a new VRE instance on the file system and creates the underlying VM.
    //  - name instance folder created (removed upon any errors)
    //  - cryptexes enumerated in instanceAssets are hard-linked (or copied) in
    //  - create (VRE+Config) instance.plist file
    //  - create darwin-init.json file (from darwinInit parameter)
    //  - create & restore VM (vrevm create) from vmConfig
    mutating func create(
        vmConfig: VMConfig, // base configuration with which to create a VM
        darwinInit: DarwinInitHelper = DarwinInitHelper(),
        instanceAssets: [CryptexSpec] // items to copy in
    ) throws {
        if VRE.exists(name) {
            throw VREError("instance folder already exists")
        }

        if let _ = try? vm.status() {
            throw VREError("VRE VM already exists")
        }

        do {
            try FileManager.default.createDirectory(at: instanceDir, withIntermediateDirectories: true)
        } catch {
            throw VREError("cannot create instance folder: \(error)")
        }

        // for all cryptex assets (from release metadata):
        // - save reference to those with an ASSET_TYPE_XX in the config
        // - link/copy into instance directory
        // - copiedAssets contains instanceAssets with updated pathnames copied into instance area
        var copiedAssets: [CryptexSpec] = []
        for asset in instanceAssets {
            do {
                let dstCryptexPath = try copyInImage(asset.path.string)
                if let assetType = asset.assetType {
                    config.addReleaseAsset(
                        type: assetType,
                        file: dstCryptexPath.path,
                        variant: asset.variant
                    )
                }

                try copiedAssets.append(
                    CryptexSpec(path: dstCryptexPath.path,
                                variant: asset.variant,
                                assetType: asset.assetType))
            } catch {
                throw VREError("copy asset into VRE area: \(error)")
            }
        }

        var vmConfig = vmConfig
        // set osImage/variant from release info (unless passed in by caller)
        if vmConfig.osImage == nil {
            guard let osAsset = config.lookupAssetType(type: .os) else {
                throw VREError("no OS restore image defined")
            }

            vmConfig.osImage = cryptexFile(osAsset.file).path
            if vmConfig.osVariantName == nil {
                vmConfig.osVariantName = osAsset.variant
            }
        }

        do {
            try config.write(to: instanceConfigFile)
        } catch {
            throw VREError("save config into VRE area: \(error)")
        }

        var darwinInit = darwinInit
        try darwinInit.populateReleaseCryptexes(assets: copiedAssets)

        vmConfig.darwinInitPath = darwinInitFile.path
        try darwinInit.save(toFile: vmConfig.darwinInitPath)

        do {
            try vm.create(config: vmConfig)
        } catch {
            throw VREError("VM creation failed: \(error)")
        }
    }

    // remove deletes associated instance VM and instance config
    func remove() throws {
        VRE.logger.log("remove VRE instance: \(name, privacy: .public)")
        do {
            do {
                try vm.remove()
            } catch {
                VRE.logger.error("remove VM: \(error, privacy: .public)")
                // fall through
            }

            try FileManager.default.removeItem(at: instanceDir)
        } catch {
            throw VREError("remove VRE instance folder: \(error)")
        }
    }

    // status returns information from the underlying VM (from vrevm --list --json)
    func status() throws -> VMStatus {
        if let status = try? vm.status() {
            return status
        }

        return VMStatus(name: name, state: "invalid")
    }

    // start launches instance VM (in the foreground);
    // under the instances/<vrename>/
    // - ensure darwin-init.json present
    // - configure & start http service
    // - update an ephemeral copy of darwin-init
    // - start VM (via vrevm), passing in updated (ephemeral) darwin-init
    // - block while VM running
    func start(
        quietMode: Bool = false // pass --quiet to vrevm
    ) async throws {
        VRE.logger.log("start VRE instance: \(name, privacy: .public)")
        guard let vmStatus = try? vm.status(), let vmState = vmStatus.state else {
            throw VREError("VRE VM not found")
        }

        guard vmState != "running" else {
            throw VREError("VRE VM currently running")
        }

        var darwinInitPath = darwinInitFile.path
        guard let darwinInit = try? DarwinInitHelper(fromFile: darwinInitPath) else {
            throw VREError("invalid darwin-init.json file")
        }

        var httpService: HTTPServer?
        var tmpDarwinInitPath: String?
        if let httpConfig = config.httpService, httpConfig.enabled {
            httpService = try await HTTPServer(
                docDir: instanceDir.path,
                bindAddr: httpConfig.address,
                bindPort: httpConfig.port
            )

            do {
                try httpService!.start()
            } catch {
                throw VREError("start http service: \(error)")
            }

            let bindAddr = String(describing: httpService!.bindAddr)
            print("HTTP service started: \(bindAddr):\(httpService!.bindPort!)")

            // update darwinInit cryptex entries with httpService endpoint -- if updates made,
            //  save to temporary file to pass into vm.start()
            if let newDarwinInit = updateDarwinInitLocalURLs(
                darwinInit: darwinInit,
                httpServer: httpService!
            ) {
                do {
                    tmpDarwinInitPath = try FileManager.tempDirectory(
                        subPath: applicationName, UUID().uuidString
                    ).appendingPathComponent("darwin-init.json").path
                    VRE.logger.debug("temp darwin-init: \(tmpDarwinInitPath!, privacy: .public)")
                } catch {
                    throw VREError("create temp dir: \(error)")
                }

                do {
                    try newDarwinInit.save(toFile: tmpDarwinInitPath)
                } catch {
                    throw VREError("temp copy of darwin-init: \(error)")
                }
                darwinInitPath = tmpDarwinInitPath!
            }
        }

        defer {
            if var httpService {
                do {
                    try httpService.shutdown()
                } catch {
                    VRE.logger.error("shutdown http service: \(error, privacy: .public)")
                    // fall through
                }
            }

            if let tmpDarwinInitPath {
                do {
                    try FileManager.default.removeItem(atPath: tmpDarwinInitPath)
                } catch {
                    VRE.logger.error("remove temp file \(tmpDarwinInitPath, privacy: .public): \(error, privacy: .public)")
                    // fall through
                }
            }
        }

        // blocks while running
        do {
            try vm.start(darwinInit: darwinInitPath, quietMode: quietMode)
        } catch {
            throw VREError("start VM: \(error)")
        }
    }

    // configureSSH adds configuration to darwin-init to enable SSH access:
    //  if enabled == true
    //    - add cryptex containing SSH service (either specified shellCryptex or the "DEBUG_SHELL"
    //        asset from release metadata info if available)
    //      - skip adding the "os" asset if variant name contains " internal "
    //    - add user{root:0:0} stanza containing publicKey (only one supported)
    //    - set "ssh: true"
    //    - can be "enabled" again to update public key
    //  if enabled == false
    //    - remove shellCryptex (if known)
    //    - set "ssh: false"
    //    - remove user{} stanza
    func configureSSH(
        enabled: Bool = true,
        publicKey: String? = nil,
        shellCryptex: CryptexSpec? = nil
    ) throws {
        VRE.logger.log("configure SSH for VRE \(name) (enabled: \(enabled, format: .answer, privacy: .public))")

        // prefix to use for disabling/reenabling darwin-init security policy keys
        let secPolicyDisabledPrefix = "SSH_DISABLED-"

        guard var darwinInit = try? DarwinInitHelper(fromFile: darwinInitFile.path) else {
            throw VREError("invalid darwin-init.json file")
        }

        if !enabled {
            darwinInit.sshConfig = DarwinInitHelper.SSH(enabled: false)
            if let shellAsset = config.lookupAssetType(type: .debugShell) {
                darwinInit.removeCryptex(variant: shellAsset.variant)
            }

            // re-enable (possibly previously disabled) security policy settings
            reenableSecurityPolicy(&darwinInit, prefix: secPolicyDisabledPrefix)

            try darwinInit.save()
            return
        }

        guard let publicKey else {
            throw VREError("public key not provided")
        }

        // try to determine whether we started with an " internal " variant
        var internalVariant = false
        if let osAsset = config.lookupAssetType(type: .os) {
            internalVariant = osAsset.variant.lowercased().contains(" internal ")
        }

        if !internalVariant { // skip adding "shell" cryptex for "internal" variants
            if let shellCryptex {
                // if cryptex specified by caller, use it
                do {
                    let dstShellCryptex = try copyInImage(shellCryptex.path.string)
                    darwinInit.addCryptex(
                        DarwinInitHelper.Cryptex(
                            url: dstShellCryptex.lastPathComponent,
                            variant: shellCryptex.variant
                        )
                    )
                } catch {
                    throw VREError("copy cryptex to VRE instance directory: \(error)")
                }
            } else if let shellAsset = config.lookupAssetType(type: .debugShell) {
                // otherwise check if AssetType.debugShell available to the instance (for non-Internal builds)
                darwinInit.addCryptex(DarwinInitHelper.Cryptex(
                    url: shellAsset.file,
                    variant: shellAsset.variant
                ))
            } else {
                VRE.logger.log("no shell cryptex provided - ssh may not be available")
            }
        }

        darwinInit.sshConfig = DarwinInitHelper.SSH(
            enabled: true,
            user: DarwinInitHelper.SSH.User(sshPubKey: publicKey)
        )

        // must disable security policy settings to allow ssh
        disableSecurityPolicy(&darwinInit, prefix: secPolicyDisabledPrefix)
        try darwinInit.save()
    }

    // copyInImage checks src exists (as a regular file), does not overwrite any "protected" files,
    //  and (if overwrite enable) ensures existing destination (if present) is removed prior to
    //  either hard-linking or copying in. A filename extension is appended as needed based on the
    //  file type (dmg, ipsw, aar). The final destination URL is returned upon success.
    func copyInImage(
        _ src: String,
        dstName: String? = nil,
        overwrite: Bool = false
    ) throws -> URL {
        let srcURL = FileManager.fileURL(src)
        var dstName = dstName ?? srcURL.lastPathComponent

        guard FileManager.isExist(src, resolve: true) else {
            throw VREError("\(src): does not exist")
        }
        guard FileManager.isRegularFile(src, resolve: true) else {
            throw VREError("\(src): not a file")
        }

        // don't allow clobbering of instance.plist, darwin-init.json, etc
        guard !VRE.protectedInstanceFiles.contains(dstName) else {
            throw VREError("cannot overwrite \(dstName)")
        }

        // determine image type (based on header)
        let imageFileType: AssetHelper.FileType
        do {
            imageFileType = try AssetHelper.fileType(srcURL)
        } catch {
            throw VREError("\(src): image type: \(error)")
        }

        // .. and append an appropriate .extension as needed
        let srcExt = srcURL.pathExtension
        let addExt = switch imageFileType {
        case .gz:
            ![imageFileType.ext, "tgz"].contains(srcExt.lowercased())
        default:
            !imageFileType.ext.isEmpty && srcExt.lowercased() != imageFileType.ext
        }

        if addExt {
            dstName += "." + imageFileType.ext
        }

        // derive destination path
        let dst = cryptexFile(dstName)
        if FileManager.isExist(dst, resolve: true) {
            if overwrite {
                try FileManager.default.removeItem(at: dst)
            } else {
                return dst
            }
        }

        do {
            try FileManager.linkFile(srcURL, dst)
            VRE.logger.debug("copy/link \(src, privacy: .public) -> \(dst.path, privacy: .public)")
        } catch {
            throw VREError("copy/link file: \(error)")
        }

        return dst
    }

    // copyPCHostTools copies list of subPaths (relative to mountPoint, which is expected to be the
    //  mount point of the PCC Host Tools) into the PCTools/ folder of the VRE instance -- the
    //  copying is recursive, so files may contain directory names. A staging directory is used
    //  during the process and moved into place when completed (previous PCTools/ is removed)
    func copyPCHostTools(
        mountPoint: URL,
        subPaths: [String] = ["usr", "System"]
    ) throws {
        VRE.logger.log("copy private cloud host-side tools")
        VRE.logger.debug("pc tools mounted at: \(mountPoint.path, privacy: .public)")

        // unpack into staging area (limit leaving partial results)
        let tmpPCToolsDir = FileManager.fileURL(pcToolsDir.path + "-unpack")
        try? FileManager.default.removeItem(at: tmpPCToolsDir)

        do {
            try FileManager.default.createDirectory(
                at: tmpPCToolsDir,
                withIntermediateDirectories: false
            )
        } catch {
            throw VREError("mkdir: \(tmpPCToolsDir.path): \(error)")
        }

        for sub in subPaths {
            let src = mountPoint.appending(path: sub)
            do {
                VRE.logger.log("copy \(src.path, privacy: .public) -> \(tmpPCToolsDir.path, privacy: .public)")
                try FileManager.default.copyItem(at: src, to: tmpPCToolsDir.appending(path: sub))
            } catch {
                throw VREError("copy \(src.path) -> \(tmpPCToolsDir.path): \(error)")
            }
        }

        // unpack completed - now move staging folder into place
        VRE.logger.debug("remove original VRE pctools folder \(pcToolsDir, privacy: .public)")
        try? FileManager.default.removeItem(at: pcToolsDir)
        do {
            try FileManager.default.moveItem(at: tmpPCToolsDir, to: pcToolsDir)
        } catch {
            throw VREError("move \(tmpPCToolsDir.path) -> \(pcToolsDir): \(error)")
        }
    }

    // disableSecurityPolicy disables config-security-policy keys in provided darwin-init.
    //  Returns true of update applied, false otherwise (key wasn't found).
    @discardableResult
    func disableSecurityPolicy(
        _ darwinInit: inout DarwinInitHelper,
        prefix: String = "DISABLED-"
    ) -> Bool {
        var updated = false
        for spkey in VRE.securityPolicyDarwinInitKeys where darwinInit.disableKey(spkey, prefix: prefix) {
            updated = true
            break
        }

        return updated
    }

    // reenableSecurityPolicy re-enables config-security-policy keys in provided darwin-init
    //   previously moved aside with prefix. Returns true of update applied, false otherwise
    //   (key wasn't found).
    @discardableResult
    func reenableSecurityPolicy(
        _ darwinInit: inout DarwinInitHelper,
        prefix: String = "DISABLED-"
    ) -> Bool {
        var updated = false
        for spkey in VRE.securityPolicyDarwinInitKeys where darwinInit.enableKey(spkey, prefix: prefix) {
            updated = true
            break
        }

        return updated
    }

    // mountPCHostTools mounts dmgPath and checks if expected directories (checkToolsDirs)
    //  are available. Publishing needs may require an inner DMG (containing the actual tools),
    //  therefore, check for .dmg file in the root folder and mount it (only taken 1 level)
    //  and check for expected dirs.
    // Returns 1 or 2 DMGHandles (with first handle referencing the one containing the tools);
    //  these should be ejected in order -- they are otherwise ejected automatically when
    //  out of scope
    // Error is thrown if expected pathnames containing the tools are not found in any image
    //   (among other reasons)
    static func mountPCHostTools(
        dmgPath: String,
        checkToolsDirs: [String] = ["usr/local/bin", "System/Library"]
    ) throws -> [DMGHelper] {
        VRE.logger.log("mountPCHostTools: \(dmgPath, privacy: .public)")
        var dmgHandles: [DMGHelper] = [] // must hold these while mounted

        func _doMount(_ dp: String) throws -> URL {
            do {
                var dmg = try DMGHelper(path: dp)
                try dmg.mount()
                dmgHandles.append(dmg)

                guard let mnt = dmg.mountPoint else {
                    throw VREError("unable to obtain mountpoint")
                }

                return mnt
            } catch {
                throw VREError("\(dmgPath): \(error)")
            }
        }

        var toolsMountPoint: URL
        do {
            toolsMountPoint = try _doMount(dmgPath)
        } catch {
            throw VREError("mount private cloud host tools: \(error)")
        }

        // check if first DMG contains another DMG in root dir
        var innerDMGPath: String?
        do {
            let dirls = try FileManager.default.contentsOfDirectory(atPath: toolsMountPoint.path)
            innerDMGPath = dirls.filter { $0.hasSuffix(".dmg") }.first
        } catch {
            throw VREError("unable to get contents of \(dmgPath)")
        }

        if var innerDMGPath {
            innerDMGPath = toolsMountPoint.appending(path: innerDMGPath).path
            VRE.logger.log("mountPCHostTools: mount inner DMG: \(innerDMGPath, privacy: .public)")
            do {
                toolsMountPoint = try _doMount(innerDMGPath)
            } catch {
                throw VREError("mount private cloud host tools: \(error)")
            }
        }

        // check if expected pathnames are available
        for sub in checkToolsDirs {
            let searchPath = toolsMountPoint.appending(path: sub)
            if !FileManager.isDirectory(searchPath) {
                VRE.logger.error("mountPCHostTools: \(searchPath, privacy: .public)/ not found")
                throw VREError("host tools not found")
            }

            VRE.logger.debug("mountPCHostTools: found \(sub, privacy: .public)/")
        }

        return dmgHandles.reversed() // innermost disk image handle first
    }

    // instances returns a set of available VRE instances
    static func instances(vrevmPath: String? = nil) -> [VRE]? {
        guard let instanceNames = try? FileManager.default.contentsOfDirectory(
            atPath: VRE.instancesDir.path)
            .filter({ FileManager.isDirectory(VRE.instanceDir($0)) })
        else {
            return nil
        }

        VRE.logger.debug("list of VRE instance: \(instanceNames, privacy: .public)")

        var vres: [VRE] = []
        for iname in instanceNames {
            do {
                try vres.append(VRE(name: iname, vrevmPath: vrevmPath))
            } catch {
                VRE.logger.error("could not load VRE instance: \(iname, privacy: .public)")
                continue
            }
        }

        return vres.count > 0 ? vres : nil
    }

    // updateDarwinInitLocalURLs updates cryptex locations (url) of darwin-init with httpServer
    // endpoint (bindAddr/bindPort) -- only entries that are not already in URL form are updated.
    // If any changes applied, updated copy of darwinInit is returned, otherwise nil (no changes).
    //
    // Example:
    //      "cryptex": [
    //          {
    //              "url": "PlatinumLining3A501_PrivateCloud_Support.aar",
    //              "variant": "PrivateCloud Support"
    //          }, ...
    //       ]
    //   is updated to:
    //      "cryptex": [
    //          {
    //              "url": "http://192.168.64.1:53423/PlatinumLining3A501_PrivateCloud_Support.tar.gz",
    //              "variant": "PrivateCloud Support"
    //          }, ...
    //       ]
    //   (where httpServer.bindAddr == 192.168.64.1 / .bindPort == 53423)
    //
    private func updateDarwinInitLocalURLs(
        darwinInit: DarwinInitHelper,
        httpServer: HTTPServer
    ) -> DarwinInitHelper? {
        var updated = false
        var newDarwinInit = darwinInit

        for cryptex in darwinInit.cryptexes {
            if let qurl = httpServer.makeURL(path: cryptex.url),
               qurl.absoluteString != cryptex.url
            {
                var newcryptex = cryptex
                newcryptex.url = qurl.absoluteString

                newDarwinInit.addCryptex(newcryptex)
                updated = true
                VRE.logger.log("update darwin-init cryptex: \(cryptex.url, privacy: .public) -> \(newcryptex.url, privacy: .public)")
            }
        }

        // only return new copy if updated
        return updated ? newDarwinInit : nil
    }
}

// VREError provides general error encapsulation for errors encountered while handling VRE instances
struct VREError: Error, CustomStringConvertible {
    var message: String
    var description: String { message }

    init(_ message: String) {
        VRE.logger.error("\(message, privacy: .public)")
        self.message = message
    }
}
