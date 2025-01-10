import { useMotherDuckClientState } from '@/lib/motherduck/context/motherduckClientContext'
import { useState, useEffect } from 'react'
/*
all columns
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
export interface MonthlyStats {
  date: string
  total: number
  larceny_theft: number
  motor_vehicle_theft: number
  other_miscellaneous: number
  assault: number
  malicious_mischief: number
}

const SQL_QUERY = `
WITH parsed_dates AS (
  SELECT 
    CASE 
      WHEN "Incident Datetime" LIKE '%/%' THEN strptime("Incident Datetime", '%Y/%m/%d %I:%M:%S %p')
      WHEN "Incident Datetime" LIKE '%-%-% %:%:%' THEN strptime("Incident Datetime", '%Y-%m-%d %H:%M:%S')
      ELSE NULL
    END as parsed_datetime,
    "Incident Category",
    COUNT(*) as count
  FROM 
    sf_crime_stats.data
  WHERE 
    "Incident Datetime" >= '2018-01-01'
    AND "Incident Datetime" <= '2025-12-31'
    AND "Incident Category" != 'Non-Criminal'
  GROUP BY 
    "Incident Datetime",
    "Incident Category"
),
monthly_stats AS (
  SELECT 
    DATE_TRUNC('month', parsed_datetime) as month,
    "Incident Category",
    SUM(count) as count
  FROM 
    parsed_dates
  WHERE 
    parsed_datetime IS NOT NULL
  GROUP BY 
    DATE_TRUNC('month', parsed_datetime),
    "Incident Category"
)
SELECT 
  month::VARCHAR as date,
  SUM(count) as total,
  SUM(CASE WHEN "Incident Category" = 'Larceny Theft' THEN count ELSE 0 END) as larceny_theft,
  SUM(CASE WHEN "Incident Category" = 'Motor Vehicle Theft' THEN count ELSE 0 END) as motor_vehicle_theft,
  SUM(CASE WHEN "Incident Category" = 'Other Miscellaneous' THEN count ELSE 0 END) as other_miscellaneous,
  SUM(CASE WHEN "Incident Category" = 'Assault' THEN count ELSE 0 END) as assault,
  SUM(CASE WHEN "Incident Category" = 'Malicious Mischief' THEN count ELSE 0 END) as malicious_mischief
FROM 
  monthly_stats
GROUP BY 
  month
ORDER BY 
  month;
`

export function useSidebarStats() {
  const { safeEvaluateQuery } = useMotherDuckClientState()
  const [data, setData] = useState<MonthlyStats[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await safeEvaluateQuery(SQL_QUERY)
        if (result.status === "success") {
          const stats = result.result.data.toRows().map((row: any) => ({
            date: row.date?.toString() ?? '',
            total: Number(row.total) || 0,
            larceny_theft: Number(row.larceny_theft) || 0,
            motor_vehicle_theft: Number(row.motor_vehicle_theft) || 0,
            other_miscellaneous: Number(row.other_miscellaneous) || 0,
            assault: Number(row.assault) || 0,
            malicious_mischief: Number(row.malicious_mischief) || 0,
          }))
          setData(stats)
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

  return {
    data,
    isLoading,
    error
  }
} 