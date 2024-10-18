'use client'
export default function Error({error, }: {
  error: Error
}) {
  console.error(error)
  return (
    <div>
      <h1>Error</h1>
      <div>{error.message}</div>
    </div>
  )
}