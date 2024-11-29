import createMarkdownIt from 'markdown-it'
import { codeToHtml, createHighlighter } from 'shiki'
import { expect, it } from 'vitest'
import createMarkdownItAsync from '../src/'

it('exported', async () => {
  const shiki = await createHighlighter({
    themes: ['vitesse-light'],
    langs: ['ts'],
  })

  const mds = createMarkdownIt({
    highlight(str, lang) {
      return shiki.codeToHtml(str, { lang, theme: 'vitesse-light' })
    },
  })
  const mda = createMarkdownItAsync({
    async highlight(str, lang) {
      return await codeToHtml(str, {
        lang,
        theme: 'vitesse-light',
      })
    },
  })

  const fixture = `
# Hello

Some code 

\`\`\`ts
console.log('Hello')
\`\`\`
`

  const result1 = mds.render(fixture)
  const result2 = await mda.renderAsync(fixture)

  expect(result1).toEqual(result2)
})
