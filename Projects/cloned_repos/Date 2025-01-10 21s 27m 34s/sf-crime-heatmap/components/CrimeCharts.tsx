"use client"

import { TrendingUp } from "lucide-react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, XAxis, ReferenceLine } from "recharts"
import { useAllIncidents } from "@/hooks/useAllIncidents"
import { useTime } from "@/contexts/TimeContext"
import { useMemo } from "react"
import { START_DATE } from "@/components/TimeSlider"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

const chartConfig = {
  incidents: {
    label: "Incidents",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

export default function CrimeCharts() {
  const { data, isLoading } = useAllIncidents()
  const { selectedWeek } = useTime()
  
  if (isLoading) {
    return <div>Loading chart data...</div>
  }

  const chartData = useMemo(() => {
    if (!data.length) return []

    const yearData: { [year: string]: number } = {}
    const startYear = 2018
    const endYear = 2025
    
    // Initialize years
    for (let year = startYear; year <= endYear; year++) {
      for (let month = 0; month < 12; month++) {
        const date = `${year}-${String(month + 1).padStart(2, '0')}`
        yearData[date] = 0
      }
    }
    
    // Count incidents by month
    data.forEach(incident => {
      const date = new Date(incident.incident_datetime)
      const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
      yearData[key] = (yearData[key] || 0) + 1
    })
    
    return Object.entries(yearData)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, count]) => ({
        date,
        incidents: count
      }))
  }, [data])

  // Calculate current reference line position
  const currentDate = useMemo(() => {
    const date = new Date(START_DATE)
    date.setDate(date.getDate() + selectedWeek * 7)
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`
  }, [selectedWeek])


  const ChartComponents = useMemo(() => ({
    bar: (
      <BarChart data={chartData} height={300}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="date"
          tickLine={false}
          tickMargin={10}
          axisLine={false}
          interval={6}
        />
        <ChartTooltip />
        <Bar dataKey="incidents" fill="var(--color-desktop)" radius={4} />
        <ReferenceLine
          x={currentDate}
          stroke="red"
          strokeDasharray="3 3"
          label={{ value: 'Current', position: 'top' }}
        />
      </BarChart>
    ),
    area: (
      <AreaChart
        data={chartData}
        height={300}
        margin={{
          left: 12,
          right: 12,
        }}
      >
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          interval={6}
        />
        <ChartTooltip />
        <Area
          dataKey="incidents"
          type="monotone"
          fill="var(--color-desktop)"
          fillOpacity={0.4}
          stroke="var(--color-desktop)"
        />
        <ReferenceLine
          x={currentDate}
          stroke="red"
          strokeDasharray="3 3"
          label={{ value: 'Current', position: 'top' }}
        />
      </AreaChart>
    )
  }), [chartData, currentDate])

  if (isLoading) {
    return <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
      <Card>
        <CardHeader>
          <CardTitle>Monthly Crime Distribution</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px] animate-pulse bg-muted" />
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Crime Trend</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px] animate-pulse bg-muted" />
      </Card>
    </div>
  }


  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
      <Card>
        <CardHeader>
          <CardTitle>Monthly Crime Distribution</CardTitle>
          <CardDescription>Bar chart showing criminal incidents by month (2018-2025)</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig}>
            {ChartComponents.bar}
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Crime Trend</CardTitle>
          <CardDescription>Area chart showing criminal incident distribution over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig}>
            {ChartComponents.area}
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
} 