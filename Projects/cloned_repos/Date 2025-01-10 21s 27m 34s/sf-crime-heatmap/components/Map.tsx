'use client'

import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import 'leaflet.heat'
import { useIncidents } from '@/hooks/useIncidents'

declare module 'leaflet' {
  export function heatLayer(latlngs: [number, number, number][], options?: any): any
}

// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: '/marker-icon-2x.png',
  iconUrl: '/marker-icon.png',
  shadowUrl: '/marker-shadow.png',
})

function HeatmapLayer({ data }: { data: [number, number, number][] }) {
  const map = useMap()

  useEffect(() => {
    if (!map) return

    const heat = L.heatLayer(data, { radius: 25 }).addTo(map)

    return () => {
      map.removeLayer(heat)
    }
  }, [map, data])

  return null
}

export default function Map() {
  const [mapCenter] = useState<[number, number]>([37.7749, -122.4194])
  const { data: incidents, isLoading, error } = useIncidents()

  if (error) return <div>Error: {error.message}</div>

  const heatmapData: [number, number, number][] = incidents.map(incident => [
    incident.latitude,
    incident.longitude,
    1 // intensity
  ])

  return (
    <>
    <MapContainer center={mapCenter} zoom={13} style={{ height: '100%', width: '100%' }}>
      
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
      <HeatmapLayer data={heatmapData} />
    </MapContainer>
        </>
  )
}