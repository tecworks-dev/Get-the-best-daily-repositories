
// personal info
export const name = 'Corey Chiu'
export const headline = 'Software engineer, Full-Stack web developer, and indie hacker.'
export const introduction = 'I’m Corey, a software engineer based in Shenzhen, China. I like coding, and building interesting things'
export const email = 'iamcoreychiu@gmail.com'
export const githubUsername = 'iamcorey'

// about page
export const aboutMeHeadline = "I'm Corey Chiu, a software engineer based in Shenzhen, China."
export const aboutParagraphs = [
  "I love coding. I learned programming when I in college. I wrote my first program in Java when I was 18.",
  "I have a lot of hobbies, such as travelling, photography, watching movies, music and so on.",
  "I'm working as a software develop engineer in Shenzhen, China now. And I'm building a lot of side projects in my spare time."
]


// projects
export const projectHeadLine = "What I've done and what I'm doing."
export const projectIntro = "I've worked on a variety of projects, from simple websites to complex web applications. And many of them are open-source. Here are a few of my favorites."


// blog
export const blogHeadLine = "What I've thinking about."
export const blogIntro = "I've written something about AI, programming and life."

// changelog
export const changelogHeadLine = "What's new about this site"
export const changelogIntro = "Check out the latest changes to this site."


// friends
export const friendsHeadLine = "Corey' friends"
export const friendsIntro = "Meet some interesting friends."

// social links
export type SocialLinkType = {
  name: string,
  ariaLabel?: string,
  icon: string,
  href: string
}

export const socialLinks: Array<SocialLinkType> = [
  {
    name: 'X',
    icon: 'x',
    href: 'https://x.com/realcoreychiu'
  },
  {
    name: 'Bsky',
    icon: 'bsky',
    href: 'https://bsky.app/profile/coreychiu.com'
  },
  {
    name: 'Jike',
    icon: 'pill',
    href: 'https://okjk.co/P7c1zU'
  },
  {
    name: 'Github',
    icon: 'github',
    href: 'https://github.com/iamcorey'
  },
  {
    name: 'Wechat',
    icon: 'wechat',
    href: 'https://mp.weixin.qq.com/s/DxnRgqNfgzXIhqj6w_x0dQ'
  },
  {
    name: 'Ko-fi',
    icon: 'coffee',
    href: 'https://ko-fi.com/coreychiu'
  }
]


// education 
export type EducationItemType = {
  school: string
  major: string
  image?: string
  logo: string
  start: string
  end: string
}



export const educationList: Array<EducationItemType> = [
  {
    school: 'Not a school',
    major: 'Computer Science and Technology',
    logo: 'college',
    start: '2016',
    end: '2020'
  },
]

// career
export type CareerItemType = {
  company: string
  title: string
  image?: string
  logo: string
  start: string
  end: string
}



export const careerList: Array<CareerItemType> = [
  {
    company: 'Not a corp',
    title: 'Software Engineer',
    logo: 'bank',
    start: '2020',
    end: 'Present'
  },
]

export type ProjectItemType = {
  name: string
  description: string
  link: { href: string, label: string }
  date?: string
  logo?: string,
  category?: string[],
  tags?: string[],
  image?: string,
  techStack?: string[],
  gitStars?: number,
  gitForks?: number
}

// projects 
export const projects: Array<ProjectItemType> = [
  {
    name: 'AI Best Tools',
    description:
      'Find the best AI tools in AIBest.tools',
    link: { href: 'aibest.tools', label: 'AI Best Tools' },
    logo: '/images/icon/aibesttools.png',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['AI', 'Tools Directory']
  },
  {
    name: 'Best Directories',
    description:
      'Your ultimate directory of directories.',
    link: { href: 'bestdirectories.org', label: 'Best Directories' },
    logo: '/images/icon/bestdirectories.png',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['Directory of Directories']
  },
  {
    name: 'User Growth',
    description:
      'Boost Your business growth with UserGrowth.link',
    link: { href: 'usergrowth.link', label: 'User Growth' },
    logo: '/images/icon/usergrowth.ico',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['User Growth', 'Marketing', 'SEO']
  },
  {
    name: 'Dev Toolset',
    description:
      'Open-source database-free tools directory.',
    link: { href: 'devtoolset.net', label: 'Dev Toolset' },
    logo: '/images/icon/devtoolset.png',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['Open Source', 'Database-Free', 'Tools Directory']
  },
  {
    name: 'Domain Score',
    description:
      'Ultimate AI-Powered tool for domain scoring and evaluation',
    link: { href: 'domainscore.ai', label: 'Domain Score' },
    logo: '/images/icon/domainscore.png',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['Domain', 'AI', 'SEO']
  },
  {
    name: 'MagicBox Tools',
    description:
      'Find the best AI tools in MagicBox.tools',
    link: { href: 'magicbox.tools', label: 'MagicBox Tools' },
    logo: '/images/icon/magicbox.png',
    category: ['Website'],
    techStack: ['Next.js', 'TailwindCSS', 'Shadcn/UI'],
    tags: ['AI', 'Tools Directory']
  },
]

export const githubProjects: Array<ProjectItemType> = [
  {
    name: 'Devtoolset',
    description: 'Open-source & database-free developer tools navigator / 开源无数据库配置的开发者工具导航站',
    link: { href: 'github.com/iAmCorey/devtoolset', label: 'Devtoolset' },
    gitStars: 146,
    gitForks: 47
  },
  {
    name: 'Chrome Extension Plasmo Template',
    description:
      'A chrome extension template using plasmo, tailwind css, shadcn/ui',
    link: { href: 'github.com/iAmCorey/chrome-extension-plasmo-template', label: 'Chrome Extension Plasmo Template' },
    gitStars: 52,
    gitForks: 10
  },
  {
    name: 'Awesome Indie Hacker Tools',
    description:
      '独立开发/出海开发相关技术栈及工具收录 / Find the best tools for indie hackers here',
    link: { href: 'github.com/iAmCorey/awesome-indie-hacker-tools', label: 'Awesome Indie Hacker Tools' },
    gitStars: 763,
    gitForks: 61
  },
  {
    name: 'Awesome AI Directory',
    description:
      'AI资源工具导航站收录 / Find all the best AI directories',
    link: { href: 'github.com/iAmCorey/awesome-ai-directory', label: 'Awesome AI Directory' },
    gitStars: 37,
    gitForks: 5
  },
  {
    name: 'Producthunt Daily Bot',
    description:
      'A bot getting product hunt daily top products',
    link: { href: 'github.com/iAmCorey/producthunt-daily-bot', label: 'Producthunt Daily Bot' },
  },
  {
    name: 'Cantonese Echoes',
    description:
      'Cantonese Echoes / 粵語殘片',
    link: { href: 'github.com/iAmCorey/Cantonese-Echoes', label: 'Cantonese Echoes' },
    gitStars: 1
  },
]

// changelog
export type ChangelogItemType = {
  date: string
  content: [{
    title: string
    description: string
  }]
}

export const changelogList: Array<ChangelogItemType> = [
  {
    date: '2024-11-18',
    content: [
      {
        title: '💡 Idea come up',
        description: 'I want to have a portfolio websit. The website doesn\'t need to be so fancy but should have a good design. It need to show all my projects and have a blog section to share my ideas and my development experience.'
      },
    ]
  },
  {
    date: '2024-11-22',
    content: [
      {
        title: '🛫 Project deployed',
        description: 'My portfolio website\'s first version finished building and deployed'
      },
    ]
  },
  {
    date: '2024-12-10',
    content: [
      {
        title: '👭 Add friends page',
        description: 'Add friends page to list my friends'
      },
    ]
  },
  {
    date: '2024-12-11',
    content: [
      {
        title: '🌿 Open source',
        description: 'Open source my portfolio website'
      },
    ]
  }
]

// https://simpleicons.org/
export const techIcons = [
  "typescript",
  "javascript",
  "supabase",
  "cloudflare",
  "java",
  "oracle",
  "mysql",
  "react",
  "nodedotjs",
  "nextdotjs",
  "prisma",
  "postgresql",
  "nginx",
  "vercel",
  "docker",
  "git",
  "github",
  "visualstudiocode",
  "androidstudio",
  "ios",
  "apple",
  "wechat"
];


// friends
export type FriendItemType = {
  name: string
  description?: string
  link: { href: string, label?: string }
  logo?: string
}

export const friends: Array<FriendItemType> = [
  {
    name: 'Corey Chiu',
    link: { href: 'https://coreychiu.com' },
  },
  {
    name: 'Axis Wang',
    link: { href: 'https://wangxs.cfd' },
  },
  {
    name: 'Greener-Dalii Studio',
    link: { href: 'https://greenerdalii.top' },
  },
  {
    name: 'Luolink',
    link: { href: 'https://luolink.com' },
  },
  {
    name: 'Frend guo',
    link: { href: 'https://frendguo.com' },
  },
  {
    name: 'Amon Xu',
    link: { href: 'https://amonxu.com' },
  },
  {
    name: 'Bill Fu',
    link: { href: 'https://billfu.com' },
  },
  {
    name: 'SavoKiss',
    link: { href: 'https://savokiss.com' },
  },
]


