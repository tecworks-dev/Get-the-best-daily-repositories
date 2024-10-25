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

import Foundation
import os
import System

// DarwinInitHelper provides a high-level interface to a darwin-init.json configuration file
// Selected "providers" available for common settings made for a VRE (such as cryptex spec and SSH)

struct DarwinInitHelper {
    let sourcePath: String?
    var loaded: [String: Any]

    static let logger = os.Logger(subsystem: applicationName, category: "DarwinInitHelper")

    // retrieve and store ["cryptex"] block as [DarwinInit.Cryptex]
    var cryptexes: [DarwinInitHelper.Cryptex] {
        get {
            var _cryptexes: [DarwinInitHelper.Cryptex] = []
            if let cryptexArray = loaded["cryptex"] as? [[String: Any]] {
                for c in cryptexArray {
                    guard let cryptex = Cryptex(c) else {
                        continue
                    }

                    _cryptexes.append(cryptex)
                }
            }

            return _cryptexes
        }

        set(newValue) {
            var _new: [[String: Any]] = []
            for c in newValue {
                _new.append(c.asDictionary())
            }

            setKey("cryptex", value: _new)
        }
    }

    // retrieve and store various settings related to [the] ssh user
    var sshConfig: DarwinInitHelper.SSH {
        get {
            var sshUser: DarwinInitHelper.SSH.User
            if let user = loaded["user"] as? [String: Any] {
                sshUser = DarwinInitHelper.SSH.User(
                    uid: UInt(user["uid"] as? String ?? "0") ?? 0,
                    gid: UInt(user["gid"] as? String ?? "0") ?? 0,
                    name: user["root"] as? String ?? "root",
                    sshPubKey: SSH.validateSSHPubKey(user["ssh_authorized_key"] as? String ?? "") ?? ""
                )
            } else {
                sshUser = DarwinInitHelper.SSH.User()
            }

            return DarwinInitHelper.SSH(
                enabled: loaded["ssh"] as? Bool ?? false,
                pwAuthEnabled: loaded["ssh_pwauth"] as? Bool ?? false,
                user: sshUser
            )
        }

        set(newValue) {
            setKey("ssh", value: newValue.enabled)

            if newValue.sshPWAuth {
                setKey("ssh_pwauth", value: newValue.sshPWAuth)
            } else {
                removeKey("ssh_pwauth") // remove if false
            }

            if let userDef = newValue.userDef {
                setKey("user", value: [
                    "uid": userDef.uid,
                    "gid": userDef.gid,
                    "name": userDef.name,
                    "ssh_authorized_key": userDef.sshPubKey,
                ])
            } else {
                removeKey("user")
            }
        }
    }

    init(fromFile: String) throws {
        do {
            let src = try NSData(contentsOfFile: fromFile) as Data
            self.loaded = try JSONSerialization.jsonObject(with: src, options: []) as! [String: Any]
        } catch {
            throw DarwinInitHelperError("darwin-init load from \(fromFile): \(error)")
        }

        self.sourcePath = fromFile
    }

    init(data: Data) throws {
        do {
            self.loaded = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
        } catch {
            throw DarwinInitHelperError("darwin-init create from provided data: \(error)")
        }

        self.sourcePath = nil
    }

    init() {
        self.sourcePath = nil
        self.loaded = [:]
    }

    // encode returns json representation as base64 blob
    func encode() -> String {
        return json(pretty: false).data(using: .utf8)!.base64EncodedString()
    }

    // json returns string representing darwin-init in json form; pretty sets whether to
    // include newlines and indentation for readability or compact form
    func json(pretty: Bool = false) -> String {
        var jsonOpts: JSONSerialization.WritingOptions = [.fragmentsAllowed, .withoutEscapingSlashes]
        if pretty {
            jsonOpts = [.fragmentsAllowed, .prettyPrinted, .withoutEscapingSlashes, .sortedKeys]
        }

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: loaded as NSDictionary, options: jsonOpts)
            let jsonString = String(decoding: jsonData, as: UTF8.self)
            return jsonString
        } catch {
            return "{}"
        }
    }

    // save writes darwin-init to named path in "pretty" form
    func save(toFile: String? = nil) throws {
        guard let toFile = toFile ?? sourcePath else {
            throw DarwinInitHelperError("darwin-init write: no destination provided")
        }

        var jsonStr = json(pretty: true)
        do {
            try jsonStr.write(toFile: toFile, atomically: true, encoding: .utf8)
        } catch {
            throw DarwinInitHelperError("darwin-init write to \(toFile): \(error)")
        }

        jsonStr = json(pretty: false)
        DarwinInitHelper.logger.debug("wrote darwin-init to: \(toFile, privacy: .public)")
        DarwinInitHelper.logger.debug("darwin-init: \(jsonStr, privacy: .public)")
    }

    // getKey returns value from in-core darwin-init or nil if not found; value type may otherwise be
    //  String, Integer, Double, Bool, or another dictionary - for bare tokens/assertions, an empty
    //  String ("") is returned
    func getKey(_ key: String) -> Any? {
        return loaded[key]
    }

    // removeKey deletes item from in-core darwin-init
    mutating func removeKey(_ key: String) {
        loaded[key] = nil
    }

    // setKey adds/replaces key to in-core darwin-init
    mutating func setKey(_ key: String, value: Any) {
        loaded.updateValue(value, forKey: key)
    }

    // disableKey renames a key within in-core darwin-init using a prefix (effectively disabling it)
    //  such that it allows it to be easily "re-enabled". Returns true if update applied, false if
    //  original key isn't found.
    @discardableResult
    mutating func disableKey(_ key: String, prefix: String = "DISABLED-") -> Bool {
        if let val = getKey(key) {
            removeKey(key)
            setKey(prefix+key, value: val)
            return true
        }

        return false
    }

    // enableKey renames a key within in-core darwin-init using a prepended prefix (such as by
    //  disableKey) back to the (original) key to enable it again - existing key (if present) is
    //  overwritten. Returns true if update applied, false if original key isn't found.
    @discardableResult
    mutating func enableKey(_ key: String, prefix: String = "DISABLED-") -> Bool {
        if let val = getKey(prefix+key) {
            removeKey(prefix+key)
            setKey(key, value: val)
            return true
        }

        return false
    }

    // lookupCryptex returns cryptex entry matching variant (nil if not found)
    func lookupCryptex(variant: String) -> Cryptex? {
        for c in cryptexes where c.variant == variant {
            return c
        }

        return nil
    }

    // addCryptex replaces existing (with matching variant) or otherwise introduces new cryptex entry
    mutating func addCryptex(_ new: DarwinInitHelper.Cryptex) {
        removeCryptex(variant: new.variant)
        cryptexes.append(new)
        DarwinInitHelper.logger.log("add cryptex '\(new.variant, privacy: .public)' (\(new.url, privacy: .public))")
    }

    // removeCryptex deletes existing cryptex entry with matching variant; returns true if updated
    @discardableResult
    mutating func removeCryptex(variant: String) -> Bool {
        var updated = false
        var loaded = cryptexes

        loaded = loaded.filter {
            if $0.variant == variant {
                DarwinInitHelper.logger.log("remove cryptex variant '\(variant, privacy: .public)'")
                updated = true
                return false
            }

            return true
        }

        if updated {
            cryptexes = loaded
        }

        return updated
    }

    //  populateReleaseCryptexes fills in cryptex entries of darwinInit from releaseAssets provided.
    //  Cryptex entries in darwinInit whose -variant- name matches AssetType (ASSET_TYPE_XX) in
    //  releaseMeta are substituted in.
    //
    //  The path location inserted is just the last component (base filename) which will later be
    //  qualified to reference the local HTTP service when the VRE is started.
    //
    //  Example:
    //    darwin-init contains the following stanza:
    //      "cryptex": [
    //          {
    //            "variant": "ASSET_TYPE_PCS",
    //            "url": "/"
    //          },
    //          {
    //            "variant": "ASSET_TYPE_MODEL",
    //            "url": "/"
    //          },
    //          ...
    //      ]
    //
    //   Assets associated with ASSET_TYPE_PCS and ASSET_TYPE_MODEL in the metadata will be filled
    //   into the corresponding entry (variant and url), as well as link/copied into the instance
    //   folder alongside the final darwin-init.json file.
    //
    mutating func populateReleaseCryptexes(
        assets: [CryptexSpec]
    ) throws {
        for diCryptex in cryptexes {
            // match cryptex variant tags in darwin-init ("ASSET_TYPE_XX") to entries in release assets
            if let assetType = SWReleaseMetadata.assetTypeByName(diCryptex.variant) {
                let assetsOfType = assets.filter {
                    SWReleaseMetadata.assetTypeName(assetType) == $0.assetType
                }
                guard assetsOfType.count == 1 else {
                    if assetsOfType.count == 0 {
                        AssetHelper.logger.error("'\(diCryptex.variant, privacy: .public)' from darwin-init in release assets not found")
                    } else {
                        let assetType = SWReleaseMetadata.assetTypeName(assetType)
                        AssetHelper.logger.error("count of \(assetType, privacy: .public) in darwin-init != 1 (\(assetsOfType.count, privacy: .public))")
                    }

                    continue
                }

                let asset = assetsOfType[0]
                let assetVariant = asset.variant

                removeCryptex(variant: diCryptex.variant)
                guard let assetBasename = asset.path.lastComponent?.string else {
                    throw VREError("derive basename from asset path: \(asset.path)")
                }

                addCryptex(DarwinInitHelper.Cryptex(
                    url: assetBasename,
                    variant: assetVariant
                ))
            }
        }
    }
}

// DarwinInitHelperError provides general error encapsulation for errors encountered within DarwinInitHelper
struct DarwinInitHelperError: Error, CustomStringConvertible {
    var message: String
    var description: String { message }

    init(_ message: String) {
        DarwinInitHelper.logger.error("\(message, privacy: .public)")
        self.message = message
    }
}
