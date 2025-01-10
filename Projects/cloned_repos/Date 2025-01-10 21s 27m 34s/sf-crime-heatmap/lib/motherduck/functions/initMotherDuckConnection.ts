"use client"

import type { MDConnection } from "@motherduck/wasm-client"

// Create a connection to MotherDuck to be used in the frontend throughout a session.
export default async function initMotherDuckConnection(mdToken: string, database?: string): Promise<MDConnection | undefined> {
    // Only run in browser environment
    if (typeof window === 'undefined') {
        console.warn("MotherDuck connection can only be initialized in browser environment")
        return
    }

    try {
        // Check for WebAssembly support
        if (typeof WebAssembly === 'undefined') {
            throw new Error('WebAssembly is not supported in your browser. This application requires WebAssembly support to function properly.')
        }

        // Dynamically import MDConnection
        const { MDConnection } = await import("@motherduck/wasm-client")

        const connection = await MDConnection.create({ mdToken })

        if (database) {
            await connection.evaluateQuery(`USE ${database}`)
        }

        return connection
    } catch (error) {
        console.error("Failed to create MotherDuck connection", error)
        if (error instanceof Error) {
            // Add Safari-specific messaging
            if (error.message.includes('WebAssembly')) {
                throw new Error(`Browser compatibility issue: ${error.message}. If you're using Safari, please ensure you have WebAssembly enabled in your settings.`)
            }
            throw error
        }
        throw new Error('An unexpected error occurred while initializing the database connection')
    }
}
