//
//  LocalizedStringKey+.swift
//
//  Created by James Sedlacek on 2/1/25.
//

import SwiftUICore

@MainActor
extension LocalizedStringKey {
    static let giveARating = LocalizedStringKey("Give a Rating")
    static let helpUsGrow = LocalizedStringKey("Help Us Grow!")
    static let maybeLater = LocalizedStringKey("Maybe Later")
    static let networkError = LocalizedStringKey("Network Error!")
    static let noRatingsYet = LocalizedStringKey("Be the first to rate us!")
    static let noReviewsYet = LocalizedStringKey("No Reviews Yet")
    static let tryAgain = LocalizedStringKey("Try Again")

    static func ratings(_ count: Int) -> LocalizedStringKey {
        .init("\(count) ratings")
    }
}
