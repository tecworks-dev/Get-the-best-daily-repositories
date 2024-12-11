"use client"


import { Student } from '@phosphor-icons/react'
import { EducationItemType, educationList } from '@/config/infoConfig'
import { CustomIcon } from '@/components/shared/CustomIcon'




function EducationItem({ educationItem }: { educationItem: EducationItemType }) {
  return (
    <li className="flex gap-4">
      <div className="relative mt-1 flex h-10 w-10 flex-none items-center justify-center rounded-full shadow-md border border-muted bg-background">
        <CustomIcon name={educationItem.logo} />
      </div>
      <dl className="flex flex-auto flex-wrap gap-x-2">
        <dt className="sr-only">Company</dt>
        <dd className="w-full flex-none text-sm font-medium">
          {educationItem.school}
        </dd>
        <dt className="sr-only">Title</dt>
        <dd className="text-xs text-muted-foreground">
          {educationItem.major}
        </dd>
        <dt className="sr-only">Date</dt>
        <dd
          className="ml-auto text-xs text-muted-foreground"
          aria-label={`${educationItem.start} until ${educationItem.end}`}
        >
          {educationItem.start} - {educationItem.end}
        </dd>
      </dl>
    </li>
  )
}

export default function Education() {


  return (
    <div className="rounded-2xl border border-muted shadow-sm p-6">
      <h2 className="flex text-sm font-semibold">
        <Student size={24} weight="duotone" />
        <span className="ml-3">Education</span>
      </h2>
      <ol className="mt-6 space-y-4">
        {educationList.map((educationItem, educationItemIndex) => (
          <EducationItem key={educationItemIndex} educationItem={educationItem} />
        ))}
      </ol>
    </div>
  )
}