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
//  ensemblectl.swift
//  ensemblectl
//
//  Created by Alex T Newman on 1/5/24.
//

import ArgumentParserInternal
import CryptoKit
import Foundation

// Import debug functions like activate and reconfigure
@_spi(Debug) import Ensemble

struct EnsembleCtl: ParsableCommand {
	static var configuration = CommandConfiguration(
		abstract: "CLI front-end to ensembled",
		subcommands: [
			GetStatus.self,
			GetDraining.self,
			GetNodeInfo.self,
			Activate.self,
			ReloadConfiguration.self,
			SendMessage.self,
			GetHealth.self,
			GetCableDiags.self,
			Encrypt.self,
			Decrypt.self,
			RotateSharedKey.self,
			GetMaxBuffersPerKey.self,
			GetMaxSecsKey.self,
			GetEnsembleID.self,
            RunServer.self,
            RunClient.self
		]
	)
}

extension EnsembleCtl {
	struct GetStatus: ParsableCommand {
		static let configuration = CommandConfiguration(abstract: "Prints status of ensemble.")
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let status = try ensembler.getStatus()
			print("Received status: \(String(describing: status))")
		}
	}

	struct GetDraining: ParsableCommand {
		static let configuration = CommandConfiguration(abstract: "Prints node's draining state.")
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let draining = try ensembler.isDraining()
			print("Received node's draining state: \(draining)")
		}
	}

	struct GetMaxBuffersPerKey: ParsableCommand {
		static let configuration = CommandConfiguration(abstract: "Gets the max buffers per key.")
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let maxBuffersPerKey = try ensembler.getMaxBuffersPerKey()
			print("Received Max buffers per key: \(maxBuffersPerKey)")
		}
	}

	struct GetMaxSecsKey: ParsableCommand {
		static let configuration = CommandConfiguration(abstract: "Gets the max seconds per key.")
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let maxSecsPerKey = try ensembler.getMaxSecsPerKey()
			print("Received Max seconds per key: \(maxSecsPerKey)")
		}
	}

	// TODO: format this less shittily
	struct GetNodeInfo: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Prints info about the nodes in the ensemble."
		)
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let nodeInfo = try ensembler.getNodeInfo()
			let leaderInfo = try ensembler.getLeaderInfo()
			let peerInfo = try ensembler.getPeerInfo()
			let leader = try ensembler.isLeader()

			print(
				"""
				Current Node Info:
				isLeader: \(leader)
				Node: \(nodeInfo)

				Leader Info:
				Node: \(leaderInfo)

				Peer Info:
				Peers: \(String(describing: peerInfo))
				"""
			)
		}
	}

	struct Activate: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Activate the ensemble manually. (Requires Root)"
		)
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let result = try ensembler.activate()
			if result == true {
				print("Activation succeeded.")
			} else {
				print("Activation failed.")
			}
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError("Activating the ensemble requires Root privileges")
			}
		}
	}

	struct RotateSharedKey: ParsableCommand {
		static let configuration =
			CommandConfiguration(abstract: "Rotate the shared key manually. (Requires Root)")
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let result = try ensembler.rotateSharedKey()
			if result == true {
				print("Rotate key succeeded.")
			} else {
				print("Rotate key failed.")
			}
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError("Rotating the key requires Root privileges")
			}
		}
	}

	struct Encrypt: ParsableCommand {
		static let configuration =
			CommandConfiguration(abstract: "Encrypt the message with sharedKey. (Requires Root)")
		@Argument(help: "The text to encrypt.")
		var text: String
		mutating func run() throws {
			let ensembler = try EnsemblerSession()

			guard let data = text.data(using: .utf8) else {
				print("Error converting text")
				return
			}
			guard let result = try ensembler.encryptData(data: data) else {
				print("Could not encrypt the data")
				return
			}

			print("Encrypted data (base64encoded) is \(result.base64EncodedString())")
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Encrypting the message using shared key requires Root privileges"
				)
			}
		}
	}

    struct RunServer: ParsableCommand {
        static let configuration =
            CommandConfiguration(abstract: "Gets the TLS options, and runs a server using the TLSOptions. (Requires Root)")
    
        @Argument(help: "The port to stand the server.")
        var port: UInt16
        
        mutating func run() throws {
            let ensembler = try EnsemblerSession()
            let options = try ensembler.getTlsOptions()
           
            print("Obtained TLS options \(options.securityProtocolOptions.hash)")
            
            runServer(port: port, tlsOptions: options)
        }

        mutating func validate() throws {
            guard geteuid() == 0 else {
                throw ValidationError(
                    "Encrypting the message using shared key requires Root privileges"
                )
            }
        }
    }
    
    struct RunClient: ParsableCommand {
        static let configuration =
            CommandConfiguration(abstract: "Gets the TLS options, and runs a server using the TLSOptions. (Requires Root)")
    
        @Argument(help: "The port on which the server listens.")
        var port: UInt16
        
        @Argument(help: "The server to connect to.")
        var server: String
        
        mutating func run() throws {
            let ensembler = try EnsemblerSession()
            let options = try ensembler.getTlsOptions()
           
            print("Obtained TLS options \(options.securityProtocolOptions.hash)")
            
            runClient(server: server, port: port, tlsOptions: options)
        }

        mutating func validate() throws {
            guard geteuid() == 0 else {
                throw ValidationError(
                    "Encrypting the message using shared key requires Root privileges"
                )
            }
        }
    }
    
	struct Decrypt: ParsableCommand {
		static let configuration =
			CommandConfiguration(
				abstract: "Decrypt the encrypted message with sharedKey. (Requires Root)"
			)
		@Argument(help: "The base64encoded text to decrypt.")
		var text: String
		mutating func run() throws {
			let ensembler = try EnsemblerSession()

			guard let data = Data(base64Encoded: text) else {
				print("Error converting text")
				return
			}

			guard let result = try ensembler.decryptData(data: data) else {
				print("Could not decrypt the data")
				return
			}

			print("Decrypted text: \(String(data: result, encoding: .utf8)!)")
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Decrypting the message using shared key requires Root privileges"
				)
			}
		}
	}

	struct ReloadConfiguration: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Force ensembled to re-read and reload configuration. (Requires Root)"
		)
		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let result = try ensembler.reloadConfiguration()
			if result == true {
				print("Configuration reloaded")
			} else {
				print("Configuration reload failed")
			}
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Reloading the ensemble configuration requires Root privileges"
				)
			}
		}
	}

	struct SendMessage: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Send a test message to given node. (Requires Root)"
		)

		@Argument(help: "the Node to message (by rank)")
		var destination: Int

		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let result = try ensembler.sendTestMessage(destination: self.destination)
			if result == true {
				print("Message sent")
			} else {
				print("Message failed to send.")
			}
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Reloading the ensemble configuration requires Root privileges"
				)
			}
		}
	}

	struct GetHealth: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Get ensemble health. (Requires Root)"
		)

		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let health = try ensembler.getHealth()
			print("Got health state: \(health.healthState)")
			print("Got health metadata: \(health.metadata)")
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Getting ensemble health requires Root privileges"
				)
			}
		}
	}

	struct GetCableDiags: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Run cable mesh diagnostics. 8 node ensembles only. (Requires Root)"
		)

		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let diags = try ensembler.getCableDiagnostics()
			print("Got \(diags.count) cable diagnostics")
			for diag in diags {
				print("  - \(diag)")
			}
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Getting cable diagnostics requires Root privileges"
				)
			}
		}
	}

	struct GetEnsembleID: ParsableCommand {
		static let configuration = CommandConfiguration(
			abstract: "Get ensemble ID. (Requires Root)"
		)

		mutating func run() throws {
			let ensembler = try EnsemblerSession()
			let ensembleID = try ensembler.getEnsembleID()
			print("Got \(String(describing: ensembleID))")
		}

		mutating func validate() throws {
			guard geteuid() == 0 else {
				throw ValidationError(
					"Getting cable diagnostics requires Root privileges"
				)
			}
		}
	}
}

@main
public enum EnsembleCtlMain {
	public static func main() throws {
		EnsembleCtl.main()
	}
}
