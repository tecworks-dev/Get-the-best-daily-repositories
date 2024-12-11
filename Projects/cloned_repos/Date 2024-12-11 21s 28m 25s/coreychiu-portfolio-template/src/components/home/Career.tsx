"use client"


import { Briefcase } from '@phosphor-icons/react'
import { CareerItemType, careerList } from '@/config/infoConfig'
import { CustomIcon} from '@/components/shared/CustomIcon'




function CareerItem({ careerItem }: { careerItem: CareerItemType }) {
  return (
    <li className="flex gap-4">
      <div className="relative mt-1 flex h-10 w-10 flex-none items-center justify-center rounded-full shadow-md border border-muted bg-background">
        <CustomIcon name={careerItem.logo} />
      </div>
      <dl className="flex flex-auto flex-wrap gap-x-2">
        <dt className="sr-only">Company</dt>
        <dd className="w-full flex-none text-sm font-medium">
          {careerItem.company}
        </dd>
        <dt className="sr-only">Title</dt>
        <dd className="text-xs text-muted-foreground">
          {careerItem.title}
        </dd>
        <dt className="sr-only">Date</dt>
        <dd
          className="ml-auto text-xs text-muted-foreground"
          aria-label={`${careerItem.start} until ${careerItem.end}`}
        >
          {careerItem.start} - {careerItem.end}
        </dd>
      </dl>
    </li>
  )
}

export default function Career() {
  return (
    <div className="rounded-2xl border border-muted shadow-sm p-6">
      <h2 className="flex text-sm font-semibold">
        <Briefcase size={24} weight="duotone" />
        <span className="ml-3">Work</span>
      </h2>
      <ol className="mt-6 space-y-4">
        {careerList.map((careerItem, careerItemIndex) => (
          <CareerItem key={careerItemIndex} careerItem={careerItem} />
        ))}
      </ol>
    </div>
  )
}