'use client'

import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { useTime } from "@/contexts/TimeContext"
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"

export const START_DATE = new Date('2018-01-01')
export const END_DATE = new Date('2025-01-07')
export const TOTAL_WEEKS = Math.floor((END_DATE.getTime() - START_DATE.getTime()) / (7 * 24 * 60 * 60 * 1000))

const SPEED_STORAGE_KEY = 'sf-crime-map:playback-speed'

export default function TimeSlider() {
  const { selectedWeek, setSelectedWeek } = useTime()
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(() => {
    // Try to load from localStorage during initialization
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(SPEED_STORAGE_KEY)
      return saved ? parseInt(saved, 10) : 1
    }
    return 1
  })
  const [smoothValue, setSmoothValue] = useState(selectedWeek)
  const animationFrameRef = useRef<number>()
  const lastTimeRef = useRef<number>()

  // Save playback speed to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem(SPEED_STORAGE_KEY, playbackSpeed.toString())
  }, [playbackSpeed])

  useEffect(() => {
    // Update smooth value when selectedWeek changes manually
    if (!isPlaying) {
      setSmoothValue(selectedWeek)
    }
  }, [selectedWeek, isPlaying])

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      lastTimeRef.current = undefined
      return
    }

    const animate = (currentTime: number) => {
      if (!lastTimeRef.current) {
        lastTimeRef.current = currentTime
      }

      const deltaTime = (currentTime - lastTimeRef.current) / 1000 // Convert to seconds
      const weekIncrement = playbackSpeed * deltaTime
      const nextValue = smoothValue + weekIncrement

      if (nextValue >= TOTAL_WEEKS) {
        setIsPlaying(false)
        setSelectedWeek(TOTAL_WEEKS - 1)
        setSmoothValue(TOTAL_WEEKS - 1)
        return
      }

      setSmoothValue(nextValue)
      // Only update the actual week when crossing integer boundaries
      if (Math.floor(nextValue) !== selectedWeek) {
        setSelectedWeek(Math.floor(nextValue))
      }

      lastTimeRef.current = currentTime
      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isPlaying, playbackSpeed, smoothValue, selectedWeek, setSelectedWeek])

  const handleSliderChange = (value: number[]) => {
    setSelectedWeek(value[0])
    setSmoothValue(value[0])
  }

  const getCurrentDate = () => {
    // Use smoothValue for display during animation
    const date = new Date(START_DATE.getTime() + Math.floor(smoothValue) * 7 * 24 * 60 * 60 * 1000)
    return date.toISOString().split('T')[0]
  }

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleSpeedChange = () => {
    // Cycle through speeds: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 1
    setPlaybackSpeed(prev => prev === 32 ? 1 : prev * 2)
  }

  return (
    <div className="bg-background/80 backdrop-blur-sm p-2 rounded-lg shadow">
      <div className="flex items-center justify-between mb-2">
        <Label htmlFor="time-slider" className="text-sm font-medium">
          Select Week:
        </Label>
        <span className="text-sm font-medium">{getCurrentDate()}</span>
      </div>
      <div className="flex items-center gap-2 mb-2">
        <Button
          variant="outline"
          size="icon"
          onClick={togglePlayPause}
          className="h-8 w-8"
        >
          {isPlaying ? (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><rect width="4" height="16" x="6" y="4"></rect><rect width="4" height="16" x="14" y="4"></rect></svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          )}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSpeedChange}
          className="h-8 w-14 text-xs"
        >
          {playbackSpeed}x
        </Button>
      </div>
      <Slider
        id="time-slider"
        max={TOTAL_WEEKS}
        step={1}
        value={[isPlaying ? Math.floor(smoothValue) : selectedWeek]}
        onValueChange={handleSliderChange}
      />
    </div>
  )
}

