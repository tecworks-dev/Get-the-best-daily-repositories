import fs from "fs"

if (process.platform === "win32") {
  const pnpmPath = process.env.npm_execpath

  const content = fs.readFileSync(pnpmPath, "utf8")

  const fixedContent = content.replace(/^#.+/, "#!node")

  fs.writeFileSync(pnpmPath, fixedContent)
}
