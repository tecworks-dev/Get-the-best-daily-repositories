'use client'
import {t} from '@lingui/macro'
export default function Component() {
  return (
    <div className="flex justify-center py-12 bg-white">
      <div className="max-w-7xl space-y-10 md:space-y-0 md:space-x-6 md:flex md:justify-between">
        <div className="flex-grow">
          <div className="flex flex-col items-center text-center">
            <div className="p-4 bg-gray-100 rounded-full">
              <DollarSignIcon className="h-6 w-6 text-gray-600" />
            </div>
            <h2 className="mt-4 font-semibold text-gray-900">{t`Free AI Photo Extender`}</h2>
            <p className="mt-2 text-sm text-gray-500 max-w-[500px]">
              {t`Start for free with the AI extender. Expand your first 3 images at no charge.`}
            </p>
          </div>
        </div>

        <div className="flex flex-col items-center text-center">
          <div className="p-4 bg-gray-100 rounded-full">
            <BrainIcon className="h-6 w-6 text-gray-600" />
          </div>
          <h2 className="mt-4 font-semibold text-gray-900">{t`Context-aware AI`}</h2>
          <p className="mt-2 text-sm text-gray-500 max-w-[500px]">
            {t`The AI analyzes your image's context and extends it for consistently realistic outcomes.`}
          </p>
        </div>
        <div className="flex flex-col items-center text-center">
          <div className="p-4 bg-gray-100 rounded-full">
            <RatioIcon className="h-6 w-6 text-gray-600" />
          </div>
          <h2 className="mt-4 font-semibold text-gray-900">{t`Extend Again`}</h2>
          <p className="mt-2 text-sm text-gray-500 max-w-[500px]">
            {t`Extended results can be expanded repeatedly,allowing for continuous enhancements`}
          </p>
        </div>
      </div>
    </div>
  )
}

function BrainIcon(props:any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
      <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
      <path d="M6 18a4 4 0 0 1-1.967-.516" />
      <path d="M19.967 17.484A4 4 0 0 1 18 18" />
    </svg>
  )
}


function DollarSignIcon(props:any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" x2="12" y1="2" y2="22" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  )
}


function RatioIcon(props:any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="12" height="20" x="6" y="2" rx="2" />
      <rect width="20" height="12" x="2" y="6" rx="2" />
    </svg>
  )
}