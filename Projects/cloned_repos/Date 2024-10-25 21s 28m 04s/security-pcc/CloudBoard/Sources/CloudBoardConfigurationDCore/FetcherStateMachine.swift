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

// Copyright © 2023 Apple. All rights reserved.

import Foundation

/// A state machine that tracks the state of the fetcher.
///
/// It controls the state transitions and both keeps track of the current
/// state, such as the applied configuration package, and also verifies
/// that state transitions are accompanied with valid side effects.
///
/// For a diagram, check out `Docs/fetcher-state-machine.md`.
struct FetcherStateMachine {
    /// The current state of the fetcher state machine.
    enum State: Hashable {
        /// The initial state, before the first fetch has been started.
        case initial

        /// The final state if things go wrong, but instead of crashing we just
        /// stop the loop and mark ourselves unhealthy, for logs/metrics to pick up.
        case terminalUnhealthy

        /// When we've fetched an initial config, but haven't applied it yet.
        case pendingFirstConfig(NodeConfigurationPackage)

        /// When we failed to apply the first config.
        case failedApplyingFirstConfig(NodeConfigurationPackage)

        /// When we successfully applied a config and are periodically checking for new ones.
        /// But even when the re-fetch fails, we continue to serve the existing config.
        /// This is a looping state, unless we get canceled or an invalid state transition occurs.
        case appliedConfig(NodeConfigurationPackage)

        /// When the initial fetch fails, we apply fallback instead and keep trying to
        /// fetch config.
        case appliedFallback

        /// When we were in fallback and successfully fetched (but not yet applied) a config.
        case pendingFromFallback(NodeConfigurationPackage)

        /// When we failed to a new fetched config while still in fallback.
        case failedApplyingFromFallback(NodeConfigurationPackage)
    }

    /// The current state of the state machine.
    private(set) var state: State = .initial

    /// The reason the state machine transitioned into the terminal unhealthy state.
    enum UnhealthyReason: Hashable, CustomStringConvertible {
        /// Used when a method is called on the state machine, but it's already in the terminal unhealthy state.
        case alreadyUnhealthy

        /// Used when the state machine tried to perform an illegal transition.
        ///
        /// Instead of crashing, we emit the error and stay in the terminal unhealthy state forever.
        case invalidStateTransition(String, String)

        var description: String {
            switch self {
            case .alreadyUnhealthy:
                return "Already unhealthy."
            case .invalidStateTransition(let state, let action):
                return "Invalid state transition from \(state) using action \(action)."
            }
        }
    }

    /// A stringified error.
    struct StringError: Error, LocalizedError, CustomStringConvertible, Hashable {
        /// The message of the error.
        let message: String

        var description: String {
            self.message
        }

        var errorDescription: String? { self.description }
    }

    /// The actions returned by the methods of this state machine.
    enum Action: Hashable {
        /// Fetch config from upstream and call the ``fetchedConfig`` with the result.
        case fetchConfig(currentRevisionIdentifier: String?)

        /// Apply the config and call the ``appliedConfig`` method with the result.
        case applyConfig(NodeConfigurationPackage)

        /// Schedule a call to ``tick`` based on the polling interval.
        case scheduleTick

        /// Apply the fallback configuration, and schedule a call to ``tick``.
        case applyFallbackAndScheduleTick

        /// Report the first config, and schedule a call to ``tick``.
        case reportFirstConfigAndScheduleTick(NodeConfigurationPackage)

        /// Report that a new config replaced an old config, and schedule a call to ``tick``.
        case reportUpdatedConfigAndScheduleTick(old: NodeConfigurationPackage, new: NodeConfigurationPackage)

        /// Report that a config was loaded from upstream after previously failing the initial fetch, and schedule a
        /// call to ``tick``.
        case reportConfigRecoveredFromFallbackAndScheduleTick(NodeConfigurationPackage)

        /// Report the error and apply the config and call the ``appliedConfig`` method with the result.
        case reportErrorAndApplyConfig(StringError, NodeConfigurationPackage)

        /// Report the error and schedule a call to ``tick``.
        case reportErrorAndScheduleTick(StringError)

        /// Stop the loop and report that the state machine has terminated in an unhealthy state.
        ///
        /// Every method will return an unhealthy action from this point on.
        case markUnhealthyAndStopLoop(UnhealthyReason)
    }

    /// Call from the main loop.
    ///
    /// Call once at the start, and then after being scheduled explicitly by an action.
    mutating func tick() -> Action {
        switch self.state {
        case .initial:
            return .fetchConfig(currentRevisionIdentifier: nil)
        case .terminalUnhealthy:
            return .markUnhealthyAndStopLoop(.alreadyUnhealthy)
        case .pendingFirstConfig:
            return .markUnhealthyAndStopLoop(.invalidStateTransition("pendingFirstConfig", "tick"))
        case .failedApplyingFirstConfig:
            return .fetchConfig(currentRevisionIdentifier: nil)
        case .appliedConfig(let package):
            return .fetchConfig(currentRevisionIdentifier: package.revisionIdentifier)
        case .appliedFallback:
            return .fetchConfig(currentRevisionIdentifier: nil)
        case .pendingFromFallback(let package):
            return .applyConfig(package)
        case .failedApplyingFromFallback:
            return .fetchConfig(currentRevisionIdentifier: nil)
        }
    }

    /// Call after being instructed to `fetchConfig` from an action, with the result.
    mutating func fetchedConfig(_ result: Result<FetchLatestResult, StringError>) -> Action {
        switch self.state {
        case .initial:
            switch result {
            case .success(let value):
                switch value {
                case .upToDate:
                    // Initial config can never be "up to date", bail.
                    self.state = .terminalUnhealthy
                    return .markUnhealthyAndStopLoop(.invalidStateTransition(
                        "initial",
                        "fetchedInitialConfig with upToDate"
                    ))
                case .newAvailable(let package):
                    self.state = .pendingFirstConfig(package)
                    return .applyConfig(package)
                }
            case .failure:
                self.state = .appliedFallback
                return .applyFallbackAndScheduleTick
            }
        case .pendingFirstConfig:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition("pendingFirstConfig", "fetchedInitialConfig"))
        case .terminalUnhealthy:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.alreadyUnhealthy)
        case .failedApplyingFirstConfig(let package):
            switch result {
            case .success(let value):
                switch value {
                case .upToDate:
                    // Config can never be "up to date" if we haven't applied first config yet, bail.
                    self.state = .terminalUnhealthy
                    return .markUnhealthyAndStopLoop(.invalidStateTransition(
                        "failedApplyingFirstConfig",
                        "fetchedConfig with upToDate"
                    ))
                case .newAvailable(let package):
                    self.state = .pendingFirstConfig(package)
                    return .applyConfig(package)
                }
            case .failure(let failure):
                self.state = .pendingFirstConfig(package)
                return .reportErrorAndApplyConfig(failure, package)
            }
        case .appliedConfig:
            switch result {
            case .success(let value):
                switch value {
                case .upToDate:
                    // Still up-to-date, no need to do anything.
                    return .scheduleTick
                case .newAvailable(let newConfig):
                    return .applyConfig(newConfig)
                }
            case .failure:
                // Upstream is unhealthy, but that's okay because we already have an applied config,
                // so try again later.
                return .scheduleTick
            }
        case .appliedFallback:
            switch result {
            case .success(let value):
                switch value {
                case .upToDate:
                    // Config can never be "up to date" if we haven't applied first (non-fallback) config yet, bail.
                    self.state = .terminalUnhealthy
                    return .markUnhealthyAndStopLoop(.invalidStateTransition(
                        "appliedFallback",
                        "fetchedConfig with upToDate"
                    ))
                case .newAvailable(let package):
                    self.state = .pendingFromFallback(package)
                    return .applyConfig(package)
                }
            case .failure:
                // Upstream is unhealthy, but that's okay because we're already on fallback,
                // so try again later.
                return .scheduleTick
            }
        case .pendingFromFallback:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition("pendingFromFallback", "fetchedConfig"))
        case .failedApplyingFromFallback(let package):
            switch result {
            case .success(let value):
                switch value {
                case .upToDate:
                    // Config can never be "up to date" if we haven't applied first (non-fallback) config yet, bail.
                    self.state = .terminalUnhealthy
                    return .markUnhealthyAndStopLoop(.invalidStateTransition(
                        "failedApplyingFromFallback",
                        "fetchedConfig with upToDate"
                    ))
                case .newAvailable(let package):
                    self.state = .pendingFromFallback(package)
                    return .applyConfig(package)
                }
            case .failure:
                self.state = .pendingFromFallback(package)
                // Upstream is unhealthy after we failed to apply, just try again later.
                return .scheduleTick
            }
        }
    }

    /// Call after being instructed to ``applyConfig`` from an action, with the result.
    mutating func attemptedToApplyConfig(_ result: Result<NodeConfigurationPackage, StringError>) -> Action {
        switch self.state {
        case .initial:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition("initial", "attemptedToApplyConfig"))
        case .terminalUnhealthy:
            return .markUnhealthyAndStopLoop(.alreadyUnhealthy)
        case .pendingFirstConfig(let package):
            switch result {
            case .success:
                self.state = .appliedConfig(package)
                return .reportFirstConfigAndScheduleTick(package)
            case .failure(let failure):
                self.state = .failedApplyingFirstConfig(package)
                return .reportErrorAndScheduleTick(failure)
            }
        case .failedApplyingFirstConfig:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition(
                "failedApplyingFirstConfig",
                "attemptedToApplyConfig"
            ))
        case .appliedConfig(let currentConfig):
            switch result {
            case .success(let newConfig):
                // We just successfully applied a new config.
                self.state = .appliedConfig(newConfig)
                return .reportUpdatedConfigAndScheduleTick(old: currentConfig, new: newConfig)
            case .failure(let failure):
                // We failed to apply a new config, we stay on the old config.
                return .reportErrorAndScheduleTick(failure)
            }
        case .appliedFallback:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition("appliedFallback", "attemptedToApplyConfig"))
        case .pendingFromFallback(let package):
            switch result {
            case .success:
                // We just successfully applied a new config after recovering from a fallback.
                self.state = .appliedConfig(package)
                return .reportConfigRecoveredFromFallbackAndScheduleTick(package)
            case .failure(let failure):
                self.state = .failedApplyingFromFallback(package)
                return .reportErrorAndScheduleTick(failure)
            }
        case .failedApplyingFromFallback:
            self.state = .terminalUnhealthy
            return .markUnhealthyAndStopLoop(.invalidStateTransition(
                "failedApplyingFromFallback",
                "attemptedToApplyConfig"
            ))
        }
    }

    /// Call when the loop around the state machine is canceled.
    mutating func cancel() {
        self.state = .terminalUnhealthy
    }
}

extension FetcherStateMachine: CustomStringConvertible {
    var description: String {
        self.state.description
    }
}

extension FetcherStateMachine.State: CustomStringConvertible {
    var description: String {
        switch self {
        case .initial:
            return "initial"
        case .terminalUnhealthy:
            return "terminalUnhealthy"
        case .pendingFirstConfig(let configurationPackage):
            return "pendingFirstConfig (\(configurationPackage))"
        case .failedApplyingFirstConfig(let configurationPackage):
            return "failedApplyingFirstConfig (\(configurationPackage))"
        case .appliedConfig(let configurationPackage):
            return "appliedConfig (\(configurationPackage))"
        case .appliedFallback:
            return "appliedFallback"
        case .pendingFromFallback(let configurationPackage):
            return "pendingFromFallback (\(configurationPackage))"
        case .failedApplyingFromFallback(let configurationPackage):
            return "failedApplyingFromFallback (\(configurationPackage))"
        }
    }
}
