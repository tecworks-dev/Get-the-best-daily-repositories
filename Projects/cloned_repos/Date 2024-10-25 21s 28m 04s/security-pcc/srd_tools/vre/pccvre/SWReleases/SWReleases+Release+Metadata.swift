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

import CryptoKit
import Foundation

typealias SWReleaseMetadata = SWReleases.Release.Metadata

public extension SWReleases.Release {
    struct Metadata {
        typealias Asset = TxPB_ReleaseMetadata.Asset
        typealias AssetType = TxPB_ReleaseMetadata.AssetType
        typealias AssetVerifier = (URL) -> Bool

        let sourcePath: String? // set if loaded from a (json) file
        let metadata: TxPB_ReleaseMetadata?

        var timestamp: Date? { return self.metadata?.timestamp.date }
        var releaseHash: Data? { return self.metadata?.releaseHash }
        var assets: [Asset]? { return self.metadata?.assets }

        // darwinInit contains DarwinInitHelper object containing attached darwin-init config
        var darwinInit: DarwinInitHelper? {
            if let darwinInitJson = darwinInitString {
                return try? DarwinInitHelper(data: darwinInitJson.data(using: .utf8)!)
            }

            return nil
        }

        // darwinInitString contains attached darwin-init JSON document as a String
        var darwinInitString: String? { return try? self.metadata?.darwinInit.jsonString() }

        init(data: Data,
             releaseHash: Data? = nil // add releaseHash if unset (ideally already set)
        ) throws {
            guard !data.isEmpty else {
                throw TransparencyLogError("no metadata payload")
            }

            guard var metadata = try? TxPB_ReleaseMetadata(serializedData: data) else {
                throw TransparencyLogError("failed to parse metadata payload")
            }

            if metadata.releaseHash.isEmpty, let releaseHash {
                metadata.releaseHash = releaseHash
            }

            self.sourcePath = nil
            self.metadata = metadata

            if let mdjson = try? self.jsonString() {
                SWReleases.logger.debug("metadata payload: \(mdjson, privacy: .public)")
            }
        }

        init(from: URL) throws {
            let jsonBlob = try Data(contentsOf: from)
            guard let metadata = try? TxPB_ReleaseMetadata(jsonUTF8Data: jsonBlob) else {
                throw TransparencyLogError("failed to parse metadata json")
            }

            self.sourcePath = from.path
            self.metadata = metadata
        }

        func jsonString() throws -> String {
            return try self.metadata?.jsonString() ?? "{ }"
        }

        // assetsByType returns the set of Assets matching assetType
        func assetsByType(_ assetType: AssetType) -> [Asset]? {
            guard let mdAssets = self.assets else {
                return nil
            }

            let assets = mdAssets.filter { $0.type == assetType }
            return assets.count > 0 ? assets : nil
        }

        // osAsset returns -first- Asset of type AssetType.os
        func osAsset() -> Asset? {
            guard let osAssets = assetsByType(.os) else {
                return nil
            }

            return osAssets[0]
        }

        // cryptexAssets returns a map of AssetType to an Asset struct
        //   (excludes OS and host tools) -- at most, one of each type is supported
        func cryptexAssets() -> [AssetType: Asset]? {
            var assetsMap: [AssetType: Asset] = [:]
            for at in AssetType.allCases {
                if [AssetType.os, AssetType.hostTools].contains(at) {
                    continue
                }

                if let assets = assetsByType(at) {
                    assetsMap[at] = assets[0]
                }
            }

            return assetsMap.count > 0 ? assetsMap : nil
        }

        // hostToolAssets returns -first- Asset of type AssetType.hostTools
        func hostToolsAsset() -> Asset? {
            guard let hostTools = assetsByType(.hostTools) else {
                return nil
            }

            return hostTools[0]
        }

        // assetURL returns a parsed URL for the asset
        func assetURL(_ asset: Asset) -> URL? {
            return URL(string: asset.url)
        }

        // assetTypeByName maps asset type JSON string to enum type
        static func assetTypeByName(_ assetType: String) -> AssetType? {
            
            return switch assetType {
            case "ASSET_TYPE_UNSPECIFIED": .unspecified
            case "ASSET_TYPE_OS": .os
            case "ASSET_TYPE_PCS": .pcs
            case "ASSET_TYPE_MODEL": .model
            case "ASSET_TYPE_HOST_TOOLS": .hostTools
            case "ASSET_TYPE_DEBUG_SHELL": .debugShell
            default: nil
            }
        }

        // assetTypeName returns asset type enum to JSON string
        static func assetTypeName(_ assetType: AssetType) -> String {
            
            return switch assetType {
            case .unspecified: "ASSET_TYPE_UNSPECIFIED"
            case .os: "ASSET_TYPE_OS"
            case .pcs: "ASSET_TYPE_PCS"
            case .model: "ASSET_TYPE_MODEL"
            case .hostTools: "ASSET_TYPE_HOST_TOOLS"
            case .debugShell: "ASSET_TYPE_DEBUG_SHELL"
            default: "ASSET_TYPE_UNKNOWN"
            }
        }

        // assetVariant returns the variant field of asset
        func assetVariant(_ asset: Asset) -> String {
            return asset.variant
        }

        // assetTicket returns the ticket field of asset (if available)
        func assetTicket(_ asset: Asset) -> Data? {
            return asset.ticket.isEmpty ? nil : asset.ticket
        }

        // assetVerifier returns a callback function to verify digest of retrieved asset
        //  against the one published in the metadata record
        static func assetVerifier(_ asset: Asset) throws -> AssetVerifier {
            guard asset.hasDigest else {
                throw TransparencyLogError("asset doesn't contain digest")
            }

            let assetDigest = asset.digest

            let hashFunc: any HashFunction.Type = switch assetDigest.digestAlg {
            case .sha256: SHA256.self
            case .sha384: SHA384.self
            default: throw TransparencyLogError("asset has unknown digest type (\(assetDigest.digestAlg.rawValue))")
            }

            return { url in
                guard let compDigest = try? computeDigest(at: url, using: hashFunc) else {
                    return false
                }

                let expectedHash = assetDigest.value
                return expectedHash.elementsEqual(compDigest)
            }
        }
    }
}
