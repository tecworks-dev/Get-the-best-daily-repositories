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

extension CLI.InstanceCmd {
    struct InstanceInferenceRequestCmd: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "inference-request",
            abstract: "Execute an LLM inference request in a VRE instance."
        )

        @OptionGroup var globalOptions: CLI.globalOptions
        @OptionGroup var instanceOptions: CLI.InstanceCmd.options

        @Option(name: [.customLong("name"), .customShort("N")], help: "VRE name.",
                transform: { try CLI.validateVREName($0) })
        var vreName: String

        @Option(name: [.customLong("tools"), .customShort("T")],
                help: "Path to PrivateCloudTools dmg file.",
                completion: .file())
        var toolsDMGPath: String?

        @Option(name: [.customLong("prompt"), .customShort("P")],
                help: "LLM inference prompt, commonly a question in English language.")
        var prompt: String

        @Option(name: [.customLong("max-tokens")],
                help: ArgumentHelp("""
                Finish inference after generating specified number of tokens.
                This controls the duration of the request and amount of produced output.
                Takes effect only when com.apple.tie.internalRequestOptionsAllowed = true.
                """, visibility: .hidden))
        var maxTokens: Int = 100

        func run() async throws {
            CLI.setupDebugStderr(debugEnable: globalOptions.debugEnable)
            CLI.logger.log("make inference request to VRE \(vreName, privacy: .public)")

            Main.printHardwareRecommendationWarningIfApplicable()

            guard VRE.exists(vreName) else {
                throw CLIError("VRE '\(vreName)' not found.")
            }

            let vre = try VRE(
                name: vreName,
                vrevmPath: instanceOptions.vrevmPath
            )

            var toolsDir: URL
            var unmountCallback: (() -> Void)?

            if let toolsDMGPath {
                // caller-provided image: mount and use in place (don't unpack)
                CLI.logger.log("caller-provided tools DMG: \(toolsDMGPath, privacy: .public)")
                (toolsDir, unmountCallback) = try CLI.mountPCHostTools(vre: vre, dmgFile: toolsDMGPath)
            } else {
                toolsDir = try CLI.unpackPCTools(vre: vre)
            }

            defer {
                if let unmountCallback {
                    unmountCallback()
                }
            }

            guard vre.vm.isRunning else {
                throw CLIError("VRE \(vreName) is not running")
            }

            guard let vreIP = try vre.status().ipaddr else {
                throw CLIError("unable to determine IP address of VRE \(vreName)")
            }

            let tiePayload = try tiePayload()

            do {
                try performInference(
                    toolsDir: toolsDir,
                    hostname: vreIP,
                    tiePayload: tiePayload
                )
            } catch {
                throw CLIError("inference call failed: \(error)")
            }

            CLI.logger.log("inference call completed without error")
        }

        private func performInference(
            toolsDir: URL,
            hostname: String,
            tiePayload: String
        ) throws {
            let envvars = [
                "DYLD_FRAMEWORK_PATH": "\(toolsDir.path)/System/Library/PrivateFrameworks/"
            ]

            let tieCMD = "tie-vre-cli"
            let tieCLI = "\(toolsDir.path)/usr/local/bin/\(tieCMD)"
            guard FileManager.isRegularFile(tieCLI) else {
                throw CLIError("Unable to find inference tool (\(tieCMD))")
            }

            let commandLine = [
                tieCLI,
                "--hostname=\(hostname)",
                "--payload",
                tiePayload
            ]
            let logMsg = "TIE CLI call: [env: \(envvars)] \(commandLine.joined(separator: " "))"
            CLI.logger.log("\(logMsg, privacy: .public)")

            print("Executing inference:")
            let (exitCode, _, errOut) = try ExecCommand(commandLine, envvars: envvars).run(
                outputMode: .tee,
                queue: DispatchQueue(label: "\(applicationName).ExecCommand")
            )

            guard exitCode == 0 else {
                if !errOut.isEmpty {
                    CLI.logger.error("tie-cli error: \(errOut, privacy: .public)")
                }

                throw CLIError("exitCode=\(exitCode)")
            }
        }

        private func tiePayload() throws -> String {
            guard let escapedPrompt = try String(data: JSONEncoder().encode(prompt), encoding: .utf8) else {
                throw CLIError("Unable to escape the prompt for JSON")
            }

            return #"""
{
    "prompt_template": {
        "prompt_template_v1": {
            "prompt_template_id": "com.apple.gm.instruct.genericChat",
            "prompt_template_variable_bindings": [
                {
                    "name": "userPrompt",
                    "value": \#(escapedPrompt)
                }
            ]
        }
    },
    "model_config": {
        "model_name": "com.apple.fm.language.research.base",
        "model_adaptor_name": "com.apple.fm.language.research.adapter",
        "tokenizer_name": "com.apple.fm.language.research.tokenizer",
        "options": {
            "max_tokens": \#(maxTokens)
        }
    }
}
"""#
        }
    }
}
