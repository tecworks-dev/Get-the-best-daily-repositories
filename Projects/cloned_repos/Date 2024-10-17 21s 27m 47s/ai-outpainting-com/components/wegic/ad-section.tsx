import React from 'react'

export default function AdSection() {
  return (
    <section className="bg-white py-16 sm:py-20 dark:bg-slate-800">
      <div className="mx-auto max-w-7xl sm:px-6 lg:px-8">
        <div
          className="relative overflow-hidden bg-gray-900 px-6 py-20 shadow-xl sm:rounded-3xl sm:px-10 sm:py-24 md:px-12 lg:px-20 dark:bg-slate-900">
          <img className="IMAGE absolute inset-0 z-[1] opacity-30 h-full w-full object-cover"
               src="https://images.unsplash.com/photo-1640622300473-977435c38c04?crop=entropy&amp;cs=tinysrgb&amp;fit=crop&amp;fm=jpg&amp;h=2000&amp;ixid=MnwxfDB8MXxyYW5kb218MHx8YWksYXJ0aXN0fHx8fHx8MTcxNTkwOTgxMQ&amp;ixlib=rb-4.0.3&amp;q=80&amp;utm_campaign=api-credit&amp;utm_medium=referral&amp;utm_source=unsplash_source&amp;w=3250"
               alt="https://images.unsplash.com/photo-1640622300473-977435c38c04?crop=entropy&amp;cs=tinysrgb&amp;fit=crop&amp;fm=jpg&amp;h=2000&amp;ixid=MnwxfDB8MXxyYW5kb218MHx8YWksYXJ0aXN0fHx8fHx8MTcxNTkwOTgxMQ&amp;ixlib=rb-4.0.3&amp;q=80&amp;utm_campaign=api-credit&amp;utm_medium=referral&amp;utm_source=unsplash_source&amp;w=3250" />
          <div className="absolute -left-80 -top-56 transform-gpu blur-3xl" aria-hidden="true">
            <div
              className="aspect-[1097/845] w-[68.5625rem] bg-gradient-to-r from-[#4caf50] to-[#2196f3] opacity-[0.45]"
              style={{ clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)' }}></div>
          </div>
          <div className="hidden md:absolute md:bottom-16 md:left-[50rem] md:block md:transform-gpu md:blur-3xl"
               aria-hidden="true">
            <div
              className="aspect-[1097/845] w-[68.5625rem] bg-gradient-to-r from-[#4caf50] to-[#2196f3] opacity-25"
              style={{ clipPath: 'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)' }}></div>
          </div>
          <div className="relative mx-auto max-w-2xl lg:mx-0">
            <figure>
              <blockquote className="DESC text-lg font-semibold text-white/90 sm:text-xl sm:leading-8">
                <div className="_editable_jwu41_1 undefined"
                     data-link="link=&amp;target=_blank&amp;text=AI%20Outpainting%20has%20transformed%20my%20creative%20process.%20The%20ability%20to%20extend%20my%20artwork%20seamlessly%20has%20opened%20up%20new%20avenues%20for%20artistic%20expression.">AI
                  Outpainting has transformed my creative process. The ability to extend my artwork seamlessly has
                  opened up new avenues for artistic expression.
                </div>
              </blockquote>
              <figcaption className="mt-6 text-base text-white/90 ">
                <div className="font-semibold">
                  <div className="_editable_jwu41_1 undefined"
                       data-link="link=&amp;target=_blank&amp;text=Jordan%20Smith">Jordan Smith
                  </div>
                </div>
                <div className="mt-1">
                  <div className="_editable_jwu41_1 undefined"
                       data-link="link=&amp;target=_blank&amp;text=Digital%20Artist">Digital Artist
                  </div>
                </div>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>
    </section>
  )
}