import * as cheerio from 'cheerio'

export async function getHackerNewsTopStories(today: string) {
  const url = `https://news.ycombinator.com/front?day=${today}`
  console.info(`get top stories ${today} from ${url}`)

  const response = await fetch(url, {
    headers: {
      'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'accept-encoding': 'gzip, deflate, br, zstd',
      'accept-language': 'zh-CN,zh;q=0.9,zh-TW;q=0.8,zh-HK;q=0.7,en;q=0.6,en-US;q=0.5,en-GB;q=0.4',
      'cache-control': 'no-cache',
      'dnt': '1',
      'pragma': 'no-cache',
      'priority': 'u=0, i',
      'referer': 'https://www.google.com/',
      'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Microsoft Edge";v="132"',
      'sec-ch-ua-mobile': '?0',
      'sec-ch-ua-platform': '"macOS"',
      'sec-fetch-dest': 'document',
      'sec-fetch-mode': 'navigate',
      'sec-fetch-site': 'same-origin',
      'sec-fetch-user': '?1',
      'upgrade-insecure-requests': '1',
      'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0',
    },
  })
  const text = await response.text()
  const $ = cheerio.load(text)
  const stories: Story[] = $('.athing.submission').map((i, el) => ({
    id: $(el).attr('id'),
    title: $(el).find('.titleline > a').text(),
    url: $(el).find('.titleline > a').attr('href'),
    hackerNewsUrl: `https://news.ycombinator.com/item?id=${$(el).attr('id')}`,
  })).get()

  return stories.filter(story => story.id && story.url)
}

export async function getHackerNewsStory(story: Story, maxTokens: number, JINA_KEY?: string) {
  const headers: HeadersInit = {
    'X-Retain-Images': 'none',
  }

  if (JINA_KEY) {
    headers.Authorization = `Bearer ${JINA_KEY}`
  }

  const [article, comments] = await Promise.all([
    fetch(`https://r.jina.ai/${story.url}`, {
      headers,
    }).then((res) => {
      if (res.ok) {
        return res.text()
      }
      else {
        console.error(`get story failed: ${res.statusText}  ${story.url}`)
        return ''
      }
    }),
    fetch(`https://r.jina.ai/https://news.ycombinator.com/item?id=${story.id}`, {
      headers: {
        ...headers,
        'X-Remove-Selector': '.navs',
        'X-Target-Selector': '#pagespace + tr',
      },
    }).then((res) => {
      if (res.ok) {
        return res.text()
      }
      else {
        console.error(`get story comments failed: ${res.statusText} https://news.ycombinator.com/item?id=${story.id}`)
        return ''
      }
    }),
  ])
  return [
    story.title
      ? `
<title>
${story.title}
</title>
`
      : '',
    article
      ? `
<article>
${article.substring(0, maxTokens * 4)}
</article>
`
      : '',
    comments
      ? `
<comments>
${comments.substring(0, maxTokens * 4)}
</comments>
`
      : '',
  ].filter(Boolean).join('\n\n---\n\n')
}
