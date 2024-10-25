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
//  LibCryptex.swift
//  darwin-init
//

import Foundation
import os
import System
import CryptoKit

// FIXME: Replace with libcryptex api calls instead of subprocesses

enum LibCryptex {
    static let logger = Logger.libcryptex
    private static let cryptexctl = URL(fileURLWithPath: "/usr/bin/cryptexctl")

    static func setUpExtractedPath() -> FilePath? {
        let extractedPath = FilePath("/var/tmp/darwin-init/cryptex/").appending(UUID().uuidString)
        try? extractedPath.remove()
        do {
            try extractedPath.createDirectory(intermediateDirectories: true)
        } catch {
            logger.error("Unable to create temp directory for cryptex extraction: \(error.localizedDescription, privacy: .public)")
            return nil
        }
        return extractedPath
    }

    static func extractCryptex(at archivePath: FilePath, to extractedPath: FilePath, url: URL?, dawToken: String?, wgUsername: String?, wgToken: String?, altCDN: String?, retries: UInt?, aeaDecryptionParams: DInitAEADecryptionParams?) async -> Bool {
        logger.log("Extracting \(archivePath) to \(extractedPath)")
        
        let archiveMagic: FilePath.ArchiveType?
        
        do {
            archiveMagic = try archivePath.readArchiveMagic()
        } catch {
            logger.error("Failed to read archive magic for \(archivePath): \(error)")
            return false
        }
        
        do {
            switch archiveMagic {
            case .UncompressedAppleArchive:
                try archivePath.extractUncompressedAppleArchive(to: extractedPath)
            case .AppleEncryptedArchive:
                guard let components = await DInitCryptexConfig.resolveDecryptionComponents(url: url, dawToken: dawToken, wgUsername: wgUsername, wgToken: wgToken, altCDN: altCDN, retries: retries, aeaDecryptionParams: aeaDecryptionParams) else {
                    logger.error("Failed to resolve AEA asset decryption components. Unable to extract cryptex.")
                    return false
                }
                try archivePath.extractAppleEncryptedArchive(to: extractedPath, using: components.key, expectingArchiveIdentifier: components.archiveID)
            default:
                try archivePath.extract(to: extractedPath)
            }
        } catch {
            logger.log("Unable to extract cryptex: \(error.localizedDescription, privacy: .public)")
            return false
        }
        
        guard (try? extractedPath.directoryExists()) ?? false else {
            logger.error("Extraction didn't yield cryptex at \(extractedPath, privacy: .public)")
            return false
        }
        
        logger.log("Successfully extracted to \(extractedPath)")
        
        do {
            let extractedContents = try extractedPath.performDeepEnumerationOfFiles()
            logger.info("Extracted \(archivePath) with contents: \(extractedContents)")
        } catch {
            logger.error("Failed to enumerate the contents of \(extractedPath): \(error)")
        }
        
        return true
    }

    static func extractCryptex(at archivePath: FilePath, url: URL?, dawToken: String?, wgUsername: String?, wgToken: String?, altCDN: String?, retries: UInt?, aeaDecryptionParams: DInitAEADecryptionParams?) async -> FilePath? {
        // Set up path to extract cryptex to
        guard let extractedPath = setUpExtractedPath() else {
            return nil
        }

        return await extractCryptex(at: archivePath, to: extractedPath, url: url, dawToken: dawToken, wgUsername: wgUsername, wgToken: wgToken, altCDN: altCDN, retries: retries, aeaDecryptionParams: aeaDecryptionParams) ? extractedPath : nil
    }

    // e.g.: cryptexctl personalize -H -V "Release" com.apple.cryptex-nginx.cxbd
    static func personalizeCryptex(at cryptex: FilePath, withVariant variant: String?, usingAuthorizationService authorizationService: DInitAuthorizationService?, locatedAt serverURL: String?, usingAppleConnect: Bool) -> FilePath? {
        do {
            let outputPath = cryptex.removingLastComponent()
            logger.log("Personalized output Directory: \(outputPath)")

            var arguments = [CustomStringConvertible]()
            if let serverURL = serverURL {
                arguments += ["--signing-url", serverURL]
            }
            if let authorizationService = authorizationService {
                arguments += ["--signing-environment", authorizationService.rawValue]
            }
            if usingAppleConnect {
                arguments += ["--signing-sso"]
            }
            arguments += ["personalize"]

            if let variant = variant {
                arguments += ["--variant", variant]
            }
            arguments += [
                "--replace",
                "--host-identity",
                "--output-directory", outputPath,
                cryptex
            ]
            _ = try Subprocess.run(
                executable: cryptexctl,
                arguments: arguments)
        } catch {
            logger.error("Unable to personalize cryptex, \(error.localizedDescription)")
            return nil
        }

        let personalizedPath = cryptex.appending(extension: "signed")
        guard (try? personalizedPath.directoryExists()) ?? false else {
            logger.error("Personalization didn't produce signed cryptex at \(personalizedPath)")
            return nil
        }

        logger.log("Successfully personalized to \(personalizedPath)")
        return personalizedPath
    }

    // e.g.: cryptexctl install -p -V "Release" com.apple.cryptex-nginx.cxbd.signed
    static func installCryptex(at cryptexPath: FilePath, withVariant variant: String?, usingAuthorizationService authorizationService: DInitAuthorizationService?, locatedAt serverURL: String?, limitLoadToREM rem: Bool?) -> Bool {
        do {
            var arguments = [CustomStringConvertible]()
            if let serverURL = serverURL {
                arguments += ["--signing-url", serverURL]
            }
            if let authorizationService = authorizationService {
                arguments += ["--signing-environment", authorizationService.rawValue]
            }
            arguments += ["install"]
            
            if let variant = variant {
                arguments += ["--variant", variant]
            }

            if let rem = rem {
                if (rem) {
                    arguments += ["--limit-load-to-rem"]
                }
            }
            arguments += [
                "--print-info",
                cryptexPath
            ]
            _ = try Subprocess.run(
                executable: cryptexctl,
                arguments: arguments)
        } catch {
            logger.error("Unable to install cryptex, \(error.localizedDescription)")
            return false
        }
        return true
    }

    static func trust(rootCertificate: Data, usingAppleConnect: Bool, signingURL: String) -> Bool {
        let temporaryPath = FilePath.newTemporaryPath()
        do {
            try temporaryPath.save(rootCertificate)
        } catch {
            logger.error("Failed to save rootCertificate to temporary path \(temporaryPath): \(error.localizedDescription)")
            return false
        }
        defer { try? temporaryPath.remove() }
        do {
            var arguments = [CustomStringConvertible]()
            if usingAppleConnect {
                arguments.append("--signing-sso")
            }
            arguments.append(contentsOf: [
                "trust",
                "-u",
                signingURL,
                temporaryPath.string
            ])
            _ = try Subprocess.run(
                executable: cryptexctl,
                arguments: arguments)
        } catch let error as NonZeroExit where error.exitCode == Errno.alreadyInProcess.rawValue {
            logger.error("Already trusted authorization service with root certificate \(rootCertificate)")
        } catch {
            logger.error("Failed to trust authorization service with root certificate \(rootCertificate): \(error.localizedDescription)")
            return false
        }
        return true
    }
}
