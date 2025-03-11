//
//  RatingRequestScreen+View.swift
//
//  Created by James Sedlacek on 3/8/25.
//

import SwiftUI

extension RatingRequestScreen: View {
    public var body: some View {
        VStack(spacing: 16) {
            headerSection
            reviewList
                .safeAreaInset(
                    edge: .bottom,
                    content: callToActionSection
                )
        }
        .padding(.vertical)
        .overlay(content: errorStateView)
        .task(fetchData)
    }
}

// MARK: Header Section
extension RatingRequestScreen {
    private var headerSection: some View {
        VStack(alignment: .center, spacing: 12) {
            titleView
            averageRatingView
            totalRatingsView
        }
    }

    private var titleView: some View {
        Text(.helpUsGrow)
            .font(.largeTitle.bold())
    }

    private var averageRatingView: some View {
        RatingView(rating: averageRating)
            .redacted(when: state.isLoading)
    }

    @ViewBuilder
    private var totalRatingsView: some View {
        if isShowingNoRatings {
            Text(.noRatingsYet)
        } else {
            HStack {
                MemojisStack()
                Text(.ratings(totalRatings))
                    .font(.body.weight(.medium))
                    .redacted(when: state.isLoading)
            }
        }
    }
}

// MARK: Review List
extension RatingRequestScreen {
    private var reviewList: some View {
        List {
            if state.isLoading {
                ForEach(0..<5, id: \.self) { _ in
                    ReviewCard(review: .mock(), memoji: Memoji[0])
                        .redacted(reason: .placeholder)
                        .listRowSeparator(.hidden)
                }
            } else {
                ForEach(reviews.indices, id: \.self) { index in
                    ReviewCard(
                        review: reviews[index],
                        memoji: Memoji[index]
                    )
                    .listRowSeparator(.hidden)
                }
            }
        }
        .scrollContentBackground(.hidden)
        .listSectionSeparator(.hidden)
        .listSectionSpacing(8)
        .listStyle(.plain)
        .overlay(noReviewsView)
    }

    @ViewBuilder
    private var noReviewsView: some View {
        if isShowingNoReviews {
            ContentUnavailableView(.noReviewsYet, symbol: .squareAndPencil)
                .aspectRatio(1, contentMode: .fit)
                .background(
                    .background.secondary,
                    in: .rect(cornerRadius: 20)
                )
                .padding(20)
        }
    }
}

// MARK: Call to Action Section
extension RatingRequestScreen {
    @ViewBuilder
    private func callToActionSection() -> some View {
        if !state.isLoading {
            VStack(spacing: .zero) {
                ratingRequestButton
                maybeLaterButton
            }
            .background(.background)
            .transition(.move(edge: .bottom))
        }
    }

    private var ratingRequestButton: some View {
        Button(.giveARating, action: ratingRequestAction)
            .buttonStyle(.filled)
            .padding()
    }

    @ViewBuilder
    private var maybeLaterButton: some View {
        if let maybeLaterAction {
            Button(.maybeLater, action: maybeLaterAction)
                .font(.subheadline)
        }
    }
}

// MARK: Error State
extension RatingRequestScreen {
    @ViewBuilder
    private func errorStateView() -> some View {
        if let errorMessage = state.errorMessage {
            ContentUnavailableView {
                Label(.networkError, symbol: .exclamationmarkTriangle)
            } description: {
                Text(errorMessage)
            } actions: {
                Button(.tryAgain, action: tryAgainAction)
                    .buttonStyle(.filled)
            }
            .background(.background)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}
