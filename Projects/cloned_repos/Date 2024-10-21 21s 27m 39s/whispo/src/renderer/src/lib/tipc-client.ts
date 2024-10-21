import { createClient, createEventHandlers } from "@egoist/tipc/renderer"
import type { Router } from "../../../main/tipc"
import type { RendererHandlers } from "../../../main/renderer-handlers"

export const tipcClient = createClient<Router>({
  // pass ipcRenderer.invoke function to the client
  // you can expose it from preload.js in BrowserWindow
  ipcInvoke: window.electron.ipcRenderer.invoke,
})

export const rendererHandlers = createEventHandlers<RendererHandlers>({
  on: window.electron.ipcRenderer.on,

  send: window.electron.ipcRenderer.send,
})
