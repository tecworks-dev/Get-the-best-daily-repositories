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
//  ESClient.swift
//  SecurityMonitorLite
//

import EndpointSecurity
import Foundation
import os
#if os(macOS)
import Darwin.bsm.libbsm
#else
import DarwinPrivate.bsm.libbsm
#endif

class ESClient {
    var client: OpaquePointer?
    let eventTypes = [
        ES_EVENT_TYPE_NOTIFY_EXEC,
        ES_EVENT_TYPE_NOTIFY_EXIT,
        ES_EVENT_TYPE_NOTIFY_IOKIT_OPEN,
        ES_EVENT_TYPE_NOTIFY_OPENSSH_LOGIN,
        ES_EVENT_TYPE_NOTIFY_OPENSSH_LOGOUT
    ]

    init?() {
        let result = es_new_client(&self.client, esEventHandler)

        guard result == ES_NEW_CLIENT_RESULT_SUCCESS && self.client != nil else {
            SMLDaemon.daemonLog.error("Error creating es_client: \(result.rawValue)")
            return nil
        }
    }

    deinit {
        if self.client != nil {
            es_delete_client(self.client)
        }
    }

    func subscribeToEvents() -> es_return_t {
        return es_subscribe(self.client!, self.eventTypes, UInt32(self.eventTypes.count))
    }

    func esEventHandler(client: OpaquePointer, message: UnsafePointer<es_message_t>) {
        let esMessage = message.pointee
        switch esMessage.event_type {
        case ES_EVENT_TYPE_NOTIFY_EXEC:
            handleExecEvent(esMessage: esMessage)
        case ES_EVENT_TYPE_NOTIFY_EXIT:
            handleExitEvent(esMessage: esMessage)
        case ES_EVENT_TYPE_NOTIFY_IOKIT_OPEN:
            handleIOKitEvent(esMessage: esMessage)
        case ES_EVENT_TYPE_NOTIFY_OPENSSH_LOGIN:
            handleSSHLoginEvent(esMessage: esMessage)
        case ES_EVENT_TYPE_NOTIFY_OPENSSH_LOGOUT:
            handleSSHLogoutEvent(esMessage: esMessage)
        default:
            SMLDaemon.daemonLog.warning("Warning: Unexpected event type: \(esMessage.event_type.rawValue)")
        }
    }

    func handleExecEvent(esMessage: es_message_t) {
        // Grab arg string
        var argString = ""
        withUnsafePointer(to: esMessage.event.exec) {
            $0.withMemoryRebound(to: es_event_exec_t.self, capacity: 1) {
                let argCount = es_exec_arg_count($0)
                for i in 0 ..< argCount {
                    let arg = es_exec_arg($0, i)
                    argString.append(String(cString: arg.data))
                    if i != argCount - 1 {
                        argString.append(" ")
                    }
                }
            }
        }

        // Populate various fields
        let execProcess = esMessage.event.exec.target.pointee
        var tty = SMLDaemon.eventInfoNone
        if execProcess.tty != nil {
            tty = String(cString: (execProcess.tty!.pointee.path.data))
        }
        var script = SMLDaemon.eventInfoNone
        if esMessage.event.exec.script != nil {
            script = String(cString: esMessage.event.exec.script!.pointee.path.data)
        }
        let parentProcess = esMessage.process.pointee

        ProcessExecEvent(
            startTime: Double(execProcess.start_time.tv_sec) + (Double(execProcess.start_time.tv_usec) / Double(USEC_PER_SEC)),
            cwd: String(cString: esMessage.event.exec.cwd.pointee.path.data),
            args: argString,
            executable: String(cString: execProcess.executable.pointee.path.data),
            signingID: String(cString: execProcess.signing_id.data),
            platformBinary: String(describing: execProcess.is_platform_binary),
            tty: tty,
            script: script,
            pid: UInt32(audit_token_to_pid(execProcess.audit_token)),
            euid: audit_token_to_euid(execProcess.audit_token),
            egid: audit_token_to_egid(execProcess.audit_token),
            ppid: execProcess.ppid,
            parentExecutable: String(cString: parentProcess.executable.pointee.path.data),
            parentSigningID: String(cString: parentProcess.signing_id.data),
            parentIsPlatformBinary: String(describing: parentProcess.is_platform_binary)
        ).export()
    }

    func handleExitEvent(esMessage: es_message_t) {
        let exitStatus = esMessage.event.exit.stat
        // Only record non-zero exit
        guard exitStatus != 0 else {
            return
        }

        ProcessExitEvent(
            timestamp: self.esMessageTimeToDouble(esMessage: esMessage),
            pid: UInt32(audit_token_to_pid(esMessage.process.pointee.audit_token)),
            executable: String(cString: esMessage.process.pointee.executable.pointee.path.data),
            exitStatus: exitStatus
        ).export()
    }

    func handleIOKitEvent(esMessage: es_message_t) {
        let iokitEvent = esMessage.event.iokit_open
        let userClientClass = String(cString: iokitEvent.user_client_class.data)
        let executablePath = String(cString: esMessage.process.pointee.executable.pointee.path.data)
        let binName = URL(fileURLWithPath: executablePath).lastPathComponent

        // Only record events for flagged classes that aren't coming from expected sources
        let flaggedClasses: Set = ["AGXDeviceUserClient", "IOGPUDeviceUserClient"]
        let allowedExecutables: Set = ["tie-inference", "tie-model-owner"]
        guard flaggedClasses.contains(userClientClass) && !allowedExecutables.contains(binName) else {
            return
        }

        IOKitOpenEvent(
            timestamp: self.esMessageTimeToDouble(esMessage: esMessage),
            pid: UInt32(audit_token_to_pid(esMessage.process.pointee.audit_token)),
            executable: executablePath,
            userClientClass: userClientClass,
            userClientType: iokitEvent.user_client_type
        ).export()
    }

    func handleSSHLoginEvent(esMessage: es_message_t) {
        let sshEvent = esMessage.event.openssh_login.pointee

        SSHLoginEvent(
            timestamp: self.esMessageTimeToDouble(esMessage: esMessage),
            srcAddr: String(cString: sshEvent.source_address.data),
            username: String(cString: sshEvent.username.data),
            success: String(describing: sshEvent.success),
            uid: sshEvent.has_uid == true ? String(describing: sshEvent.uid.uid) : SMLDaemon.eventInfoNone
        ).export()
    }

    func handleSSHLogoutEvent(esMessage: es_message_t) {
        let sshEvent = esMessage.event.openssh_logout.pointee

        SSHLogoutEvent(
            timestamp: self.esMessageTimeToDouble(esMessage: esMessage),
            srcAddr: String(cString: sshEvent.source_address.data),
            username: String(cString: sshEvent.username.data),
            uid: sshEvent.uid
        ).export()
    }

    private func esMessageTimeToDouble(esMessage: es_message_t) -> Double {
        return Double(esMessage.time.tv_sec) + (Double(esMessage.time.tv_nsec) / Double(NSEC_PER_SEC))
    }

}
