import { Footer } from '@/components/layout/Footer'
import { Header } from '@/components/layout/Header'

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <div className="fixed inset-0 flex justify-center sm:px-8">
        <div className="flex w-full max-w-7xl lg:px-8">
          <div className="w-full shadow-xl dark:bg-muted" />
        </div>
      </div>
      <div className="relative flex w-full flex-col px-4 sm:px-0">
        <Header />
        <main className="flex-auto">{children}</main>
        <Footer />
      </div>
    </>
  )
}
