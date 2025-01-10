import { useTime } from '@/contexts/TimeContext'
import { START_DATE } from '@/components/TimeSlider'
import { useMemo } from 'react'
import { useMotherDuckClientState } from '@/lib/motherduck/context/motherduckClientContext'
import { Incident } from '@/types/incidents'
import { useState, useEffect } from 'react'
import type { DuckDBRow } from '@motherduck/wasm-client'
/*
all columns, used for llm dont delete
"Incident Datetime",
"Incident Date",
"Incident Time",
"Incident Year",
"Incident Day of Week",
"Report Datetime",
"Row ID",
"Incident ID",
"Incident Number",
"CAD Number",
"Report Type Code",
"Report Type Description",
"Filed Online",
"Incident Code",
"Incident Category",
"Incident Subcategory",
"Incident Description",
Resolution,
Intersection,
CNN,
"Police District",
"Analysis Neighborhood",
"Supervisor District",
"Supervisor District 2012",
Latitude,
Longitude,
Point,
Neighborhoods,
"ESNCAG - Boundary File",
"Central Market/Tenderloin Boundary Polygon - Updated",
"Civic Center Harm Reduction Project Boundary",
"HSOC Zones as of 2018-06-05",
"Invest In Neighborhoods (IIN) Areas",
"Current Supervisor Districts",
"Current Police Districts"
*/
const SQL_QUERY = `
SELECT 
  "Incident Datetime" as incident_datetime,
  "Incident Category" as incident_category,
  "Incident Description" as incident_description,
  Latitude as latitude,
  Longitude as longitude
FROM 
  sf_crime_stats.data
WHERE 
  Latitude IS NOT NULL 
  AND Longitude IS NOT NULL
  AND "Incident Category" != 'Non-Criminal'
  AND "Incident Datetime" >= '2018-01-01'
  AND "Incident Datetime" <= '2025-12-31'
ORDER BY "Incident Datetime" DESC;
`

type WeeklyIncidents = { [weekIndex: number]: Incident[] }

export function useIncidents() {
  const { safeEvaluateQuery } = useMotherDuckClientState()
  const { selectedWeek } = useTime()
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [rawData, setRawData] = useState<Incident[]>([])

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await safeEvaluateQuery(SQL_QUERY)
        if (result.status === "success") {
          const incidents = result.result.data.toRows().map((row: DuckDBRow) => ({
            incident_datetime: String(row.incident_datetime),
            incident_category: String(row.incident_category),
            incident_description: String(row.incident_description),
            latitude: Number(row.latitude) || 0,
            longitude: Number(row.longitude) || 0,
          }))
          setRawData(incidents)
          setError(null)
        } else {
          setError(new Error(result.err.message))
        }
      } catch (err) {
        setError(err instanceof Error ? err : new Error('An error occurred'))
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [safeEvaluateQuery])

  // Pre-process data into weekly chunks when rawData changes
  const weeklyData = useMemo(() => {
    if (!rawData) return {}

    const weeklyChunks: WeeklyIncidents = {}
    
    rawData.forEach(incident => {
      const incidentDate = new Date(incident.incident_datetime)
      const weekIndex = Math.floor((incidentDate.getTime() - START_DATE.getTime()) / (7 * 24 * 60 * 60 * 1000))
      
      if (!weeklyChunks[weekIndex]) {
        weeklyChunks[weekIndex] = []
      }
      weeklyChunks[weekIndex].push(incident)
    })

    return weeklyChunks
  }, [rawData])

  // Get data for selected week
  const data = useMemo(() => {
    return weeklyData[selectedWeek] || []
  }, [weeklyData, selectedWeek])

  return {
    data,
    isLoading,
    error
  }
}

