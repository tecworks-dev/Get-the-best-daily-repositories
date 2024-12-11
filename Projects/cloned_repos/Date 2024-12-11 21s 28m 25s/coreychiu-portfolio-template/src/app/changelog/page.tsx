
import { SimpleLayout } from '@/components/layout/SimpleLayout'
import { changelogIntro, changelogHeadLine } from '@/config/infoConfig'

import ChangelogBlock from '@/components/changelog/Changelog'



export const metadata = {
  title: 'Changelog',
  description: changelogHeadLine,
}

export default function Changelog() {
  return (
    <SimpleLayout
      title={changelogHeadLine}
      intro={changelogIntro}
    >
      <div className="space-y-20">
        <ChangelogBlock />
      </div>
    </SimpleLayout>
  )
}
