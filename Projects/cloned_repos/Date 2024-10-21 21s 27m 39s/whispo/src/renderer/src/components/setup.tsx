import { useMicrphoneStatusQuery } from "@renderer/lib/query-client"
import { Button } from "./ui/button"
import { tipcClient } from "@renderer/lib/tipc-client"
import { useQuery } from "@tanstack/react-query"

export function Setup() {
  const microphoneStatusQuery = useMicrphoneStatusQuery()
  const isAccessibilityGrantedQuery = useQuery({
    queryKey: ["setup-isAccessibilityGranted"],
    queryFn: () => tipcClient.isAccessibilityGranted(),
  })

  return (
    <div className="app-drag-region flex h-dvh items-center justify-center p-10">
      <div className="-mt-20">
        <h1 className="text-center text-3xl font-extrabold">
          Welcome to {process.env.PRODUCT_NAME}
        </h1>
        <h2 className="mb-10 text-center text-neutral-500 dark:text-neutral-400">
          We need some system permissions before we can run the app
        </h2>
        <div className="mx-auto max-w-screen-md">
          <div className="grid divide-y rounded-lg border">
            {process.env.IS_MAC && (
              <PermissionBlock
                title="Accessibility Access"
                description={`We need Accessibility Access to capture keyboard events, so that you can hold Ctrl key to start recording, we don't log or store your keyboard events.`}
                actionText="Enable in System Settings"
                actionHandler={() => {
                  tipcClient.requestAccesssbilityAccess()
                }}
                enabled={isAccessibilityGrantedQuery.data}
              />
            )}

            <PermissionBlock
              title="Microphone Access"
              description={`We need Microphone Access to record your microphone, recordings are store locally on your computer only.`}
              actionText={
                microphoneStatusQuery.data === "denied"
                  ? "Enable in System Settings"
                  : "Request Access"
              }
              actionHandler={async () => {
                const granted = await tipcClient.requestMicrophoneAccess()
                if (!granted) {
                  tipcClient.openMicrophoneInSystemPreferences()
                }
              }}
              enabled={microphoneStatusQuery.data === "granted"}
            />
          </div>
        </div>

        <div className="mt-10 flex items-center justify-center">
          <Button
            variant="outline"
            className="gap-2"
            onClick={() => {
              tipcClient.restartApp()
            }}
          >
            <span className="i-mingcute-refresh-2-line"></span>
            <span>Restart App</span>
          </Button>
        </div>
      </div>
    </div>
  )
}

const PermissionBlock = ({
  title,
  description,
  actionHandler,
  actionText,
  enabled,
}: {
  title: React.ReactNode
  description: React.ReactNode
  actionText: string
  actionHandler: () => void
  enabled?: boolean
}) => {
  return (
    <div className="grid grid-cols-2 gap-5 p-3">
      <div>
        <div className="text-lg font-bold">{title}</div>
        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          {description}
        </div>
      </div>
      <div className="flex items-center justify-end">
        {enabled ? (
          <div className="inline-flex items-center gap-1 text-green-500">
            <span className="i-mingcute-check-fill"></span>
            <span>Granted</span>
          </div>
        ) : (
          <Button type="button" onClick={actionHandler}>
            {actionText}
          </Button>
        )}
      </div>
    </div>
  )
}
