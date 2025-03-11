//
//  MemojisStack.swift
//
//  Created by James Sedlacek on 3/7/25.
//

import SwiftUI

struct MemojisStack: View {
    var body: some View {
        HStack(alignment: .center, spacing: -10) {
            memoji(at: 7)
                .zIndex(3)

            memoji(at: 6)
                .zIndex(2)

            memoji(at: 5)
                .zIndex(1)
        }
    }

    private func memoji(at index: Int) -> some View {
        Memoji[index].image
            .resizable()
            .frame(width: 40, height: 40)
            .background(.background.secondary)
            .clipShape(.circle)
            .background(
                Circle()
                    .stroke(.background, lineWidth: 4)
            )
    }
}

#Preview {
    MemojisStack()
        .padding()
}
