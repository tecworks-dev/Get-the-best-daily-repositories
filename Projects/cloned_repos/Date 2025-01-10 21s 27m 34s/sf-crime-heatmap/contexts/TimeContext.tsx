'use client'

import { createContext, useContext, useState, ReactNode, useEffect } from 'react'

const TIME_STORAGE_KEY = 'sf-crime-map:time'

interface TimeContextType {
  selectedWeek: number
  setSelectedWeek: (week: number) => void
}

const TimeContext = createContext<TimeContextType | undefined>(undefined)

export function TimeProvider({ children }: { children: ReactNode }) {
  const [selectedWeek, setSelectedWeek] = useState(() => {
    // Try to load from localStorage during initialization
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(TIME_STORAGE_KEY)
      return saved ? parseInt(saved, 10) : 0
    }
    return 0
  })

  // Save to localStorage whenever selectedWeek changes
  useEffect(() => {
    localStorage.setItem(TIME_STORAGE_KEY, selectedWeek.toString())
  }, [selectedWeek])

  return (
    <TimeContext.Provider value={{ selectedWeek, setSelectedWeek }}>
      {children}
    </TimeContext.Provider>
  )
}

export function useTime() {
  const context = useContext(TimeContext)
  if (context === undefined) {
    throw new Error('useTime must be used within a TimeProvider')
  }
  return context
} 