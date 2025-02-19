export default function AuthLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    // <html>
    //   <body>
        <main>{children}</main>
    //   </body>
    // </html>
  )
}