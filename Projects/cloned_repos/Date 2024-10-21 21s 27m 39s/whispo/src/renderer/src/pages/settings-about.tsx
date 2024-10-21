export function Component() {
  return (
    <div className="flex items-center justify-center rounded-lg border border-neutral-200 bg-neutral-50 py-5 dark:border-border dark:bg-neutral-900">
      <h1>
        {process.env.PRODUCT_NAME} {process.env.APP_VERSION}
      </h1>
    </div>
  )
}
