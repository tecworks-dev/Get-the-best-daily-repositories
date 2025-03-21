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

extension CLI.InstanceCmd.InstanceConfigureCmd {
    struct SshCmd: AsyncParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "ssh",
            abstract: "Configure ssh for a VRE instance."
        )

        @OptionGroup var globalOptions: CLI.globalOptions
        @OptionGroup var configureOptions: CLI.InstanceCmd.InstanceConfigureCmd.options
        @OptionGroup var instanceOptions: CLI.InstanceCmd.options

        @Option(name: [.customLong("cryptex"), .customShort("C")],
                help: "Alternate cryptex image to support ssh (<variant>:<path>).",
                transform: { try CLI.validateCryptexSpec($0) })
        var cryptex: CryptexSpec?

        @Flag(name: [.customLong("disable")],
              help: "Disable SSH service (rather than enable).")
        var disable: Bool = false

        @Option(name: [.customLong("public-key-path"), .customShort("p")],
                help: "Public key path to use for ssh (~/.ssh/id_ed25519.pub for example)",
                completion: .file())
        var publicKeyPath: String?

        func run() async throws {
            CLI.setupDebugStderr(debugEnable: globalOptions.debugEnable)

            let vreName = configureOptions.instanceName
            CLI.logger.log("\(disable ? "disable" : "configure", privacy: .public) SSH for \(vreName, privacy: .public)")

            guard VRE.exists(vreName) else {
                throw CLIError("VRE '\(vreName)' not found")
            }

            let vre = try VRE(
                name: vreName,
                vrevmPath: instanceOptions.vrevmPath
            )

            if disable {
                do {
                    try vre.configureSSH(enabled: false)
                } catch {
                    throw CLIError("updating VRE configuration: \(error)")
                }

                print("Disabled SSH service for VRE instance.")
                return
            }

            guard let publicKeyPath else {
                throw ValidationError("must specify public-key-path")
            }

            // retrieve 1st public key found in provided path
            let sshPubKey = try loadSSHPubKey(from: publicKeyPath)

            try vre.configureSSH(publicKey: sshPubKey, shellCryptex: cryptex)
            print("Enabled SSH service for VRE instance.")

            // if ipAddr available for the VRE, print access info
            do {
                let vmStatus = try vre.vm.status()

                if let ipAddr = vmStatus.ipaddr {
                    print("  Access the VRE instance via 'ssh root@\(ipAddr)' using specified pubkey")
                    if vmStatus.isRunning {
                        print("    (upon next restart)")
                    }
                }
            } catch {}
        }

        // loadSSHPubKey retrieves the first valid SSH pubkey in the path 'from' -- ssh public key
        // string is returned (first 2 fields: <keytype> <base64key>) or an error is thrown
        private func loadSSHPubKey(from: String) throws -> String {
            let pubKeyContents: String
            do {
                pubKeyContents = try String(contentsOf: FileManager.fileURL(from), encoding: .utf8)
            } catch {
                throw CLIError("Unable to load \(from): \(error)")
            }

            for pubKey in pubKeyContents.components(separatedBy: .newlines) {
                if let sshPubKey = DarwinInitHelper.SSH.validateSSHPubKey(pubKey.trimmingCharacters(in: .whitespaces)) {
                    return sshPubKey
                }
            }

            throw CLIError("Unable to find a public key in \(from)")
        }
    }
}
