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

struct MachO64 {
    fileprivate static let headerSize = MemoryLayout<MachO64Header_t>.stride

    enum ParseError: Error, CustomStringConvertible {
        case fileSize(String)
        case fileMagic(String)
        case missingCmd(String, String)
        case duplicateCmd(String, String)
        case segmentInvalid(String)
        case segmentCoverage(String)
        case segmentContiguous(String)
        case segmentUnknown(String, String)
        case unixThreadUnsupported(String)
        case unixThreadFlavor(String, UInt32)

        var description: String {
            switch self {
            case .fileSize(let fileName):
                return "MachO64 | \(fileName) is too small to be a 64-bit MachO"
            case .fileMagic(let fileName):
                return "MachO64 | \(fileName) is not a 64-bit MachO"
            case .missingCmd(let fileName, let cmdName):
                return "MachO64 | \(fileName) does not have a LC_\(cmdName) command"
            case .duplicateCmd(let fileName, let cmdName):
                return "MachO64 | \(fileName) has multiple LC_\(cmdName) commands"
            case .segmentInvalid(let fileName):
                return "MachO64 | \(fileName) contains an invalid segment command"
            case .segmentCoverage(let fileName):
                return "MachO64 | \(fileName) doesn't contain enough segments to cover all bytes"
            case .segmentContiguous(let segName):
                return "MachO64 | \(segName) not contiguous with other regional segments"
            case .segmentUnknown(let filename, let segName):
                return "MachO64 | \(filename) contains an unknown segment: \(segName)"
            case .unixThreadUnsupported(let fileName):
                return "MachO64 | \(fileName) contains an unsupported unix thread"
            case .unixThreadFlavor(let fileName, let flavor):
                return "MachO64 | \(fileName) contains an unsupported unix thread: \(flavor)"
            }
        }
    }

    let fileName: String
    let fileData: Data
    let header: MachO64Header_t

    init (fileName: String, fileData: Data) throws {
        let header = try fileData.withUnsafeBytes { rawData in
            if rawData.count < MachO64.headerSize {
                throw ParseError.fileSize(fileName)
            }
            let rawPtr = OpaquePointer(rawData.baseAddress!)
            return UnsafePointer<MachO64Header_t>(rawPtr).pointee
        }

        /* Verify the magic number */
        if header.magic != kMachO64Magic {
            throw ParseError.fileMagic(fileName)
        }

        self.header = header
        self.fileName = fileName
        self.fileData = fileData
    }

    func find(loadCommand: UInt32) -> [Data] {
        var matchedCmds = [Data]()

        self.fileData.withUnsafeBytes { rawData in
            let rawPtr = rawData.baseAddress!.advanced(by: MachO64.headerSize)
            var startPtr = rawPtr.assumingMemoryBound(to: UInt8.self)

            for _ in 0..<self.header.numCmds {
                let cmdPtr = UnsafePointer<MachOLoadCommand_t>(OpaquePointer(startPtr))
                let loadCmd = cmdPtr.pointee

                if loadCmd.cmdCode == loadCommand {
                    let cmdBuf = UnsafeBufferPointer<UInt8>(
                        start: startPtr,
                        count: Int(loadCmd.cmdSize)
                    )
                    matchedCmds.append(Data(buffer: cmdBuf))
                }
                startPtr = startPtr.advanced(by: Int(cmdPtr.pointee.cmdSize))
            }
        }

        return matchedCmds
    }
}
