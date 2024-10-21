import { RouterProvider } from "react-router-dom"
import { router } from "./router"
import { useQuery } from "@tanstack/react-query"
import { tipcClient } from "./lib/tipc-client"
import { Setup } from "./components/setup"
import { lazy, Suspense } from "react"

const Updater = lazy(() => import("./components/updater"))

function App(): JSX.Element {
  const isAccessibilityGrantedQuery = useQuery({
    queryKey: ["isAccessibilityGranted"],
    queryFn: async () => {
      // return false
      return tipcClient.isAccessibilityGranted()
    },
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
  })

  return (
    <>
      {isAccessibilityGrantedQuery.data === false ? (
        <Setup />
      ) : (
        <RouterProvider router={router}></RouterProvider>
      )}

      <Suspense>
        <Updater />
      </Suspense>
    </>
  )
}

export default App
