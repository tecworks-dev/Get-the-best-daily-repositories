# quansync

[![npm version][npm-version-src]][npm-version-href]
[![npm downloads][npm-downloads-src]][npm-downloads-href]
[![bundle][bundle-src]][bundle-href]
[![JSDocs][jsdocs-src]][jsdocs-href]
[![License][license-src]][license-href]

Create sync/async APIs with usable logic.

**Quan**tum + **Sync** - "_Superposition_" between `sync` and `async`.

- Typesafe
- ESM, modern JavaScript
- Zero dependencies

Heavily inspired by [`genasync`](https://github.com/loganfsmyth/gensync) by [@loganfsmyth](https://github.com/loganfsmyth).

## Usage

```bash
pnpm i quansync
```

```ts
import fs from 'node:fs'
import { quansync } from 'quansync'

// Create an quansync function by providing `sync` and `async` implementations
const readFile = quansync({
  sync: (path: string) => fs.readFileSync(path),
  async: (path: string) => fs.promises.readFile(path),
})

// Create an quansync function by providing a generator function
const myFunction = quansync(function* (filename) {
  // Use `yield*` to call another quansync function
  const code = yield * readFile(filename, 'utf8')

  return `// some custom prefix\n${code}`
})

// Use it as a sync function
const result = myFunction.sync('./some-file.js')

// Use it as an async function
const asyncResult = await myFunction.async('./some-file.js')
```

## Build-time Macro

You can use `quansync` as a build-time macro to generate sync and async functions in an async/await style.

```ts
import fs from 'node:fs'
import { quansyncMacro } from 'quansync'

// Create an quansync function by providing `sync` and `async` implementations
const readFile = quansyncMacro({
  sync: (path: string) => fs.readFileSync(path),
  async: (path: string) => fs.promises.readFile(path),
})

// Create an quansync function by providing a **async** function
const myFunction = quansyncMacro(async (filename) => {
  // Use `await` to call another quansync function
  const code = await readFile(filename, 'utf8')

  return `// some custom prefix\n${code}`
})

// Use it as a sync function
const result = myFunction.sync('./some-file.js')

// Use it as an async function
const asyncResult = await myFunction.async('./some-file.js')
```

For more details on usage, see [unplugin-quansync](https://github.com/unplugin/unplugin-quansync#usage).

## Sponsors

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/antfu/static/sponsors.svg">
    <img src='https://cdn.jsdelivr.net/gh/antfu/static/sponsors.svg'/>
  </a>
</p>

## License

[MIT](./LICENSE) License © [Anthony Fu](https://github.com/antfu)

<!-- Badges -->

[npm-version-src]: https://img.shields.io/npm/v/quansync?style=flat&colorA=080f12&colorB=1fa669
[npm-version-href]: https://npmjs.com/package/quansync
[npm-downloads-src]: https://img.shields.io/npm/dm/quansync?style=flat&colorA=080f12&colorB=1fa669
[npm-downloads-href]: https://npmjs.com/package/quansync
[bundle-src]: https://img.shields.io/bundlephobia/minzip/quansync?style=flat&colorA=080f12&colorB=1fa669&label=minzip
[bundle-href]: https://bundlephobia.com/result?p=quansync
[license-src]: https://img.shields.io/github/license/antfu/quansync.svg?style=flat&colorA=080f12&colorB=1fa669
[license-href]: https://github.com/antfu/quansync/blob/main/LICENSE
[jsdocs-src]: https://img.shields.io/badge/jsdocs-reference-080f12?style=flat&colorA=080f12&colorB=1fa669
[jsdocs-href]: https://www.jsdocs.io/package/quansync
