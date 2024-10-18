'use client'
import { t } from '@lingui/macro'

type FAQItem = {
  question: string
  answer: string
}


const FaqItem = ({ faq }: { faq: FAQItem }) => (
  <div className="bg-gray-100 dark:bg-[#1E2735] mb-[1px]">
    <h5 className="p-4 lg:p-6 mb-0 w-full text-start flex justify-between items-center text-xl font-medium">
      <span>{faq.question}</span>
    </h5>
    <div className="px-3 lg:px-6 pb-2 lg:pb-6">
      <p className="opacity-50">{faq.answer}</p>
    </div>
  </div>
)


const FAQs = () => {
  const items: FAQItem[] = [
    {
      'question': t`What is AI Photo Extender?`,
      'answer': t`AI Photo Extender is an innovative online tool that uses advanced AI technology to expand and enhance user-uploaded images. This AI photo extender can seamlessly extend image content in all four directions while maintaining the original style. Additionally, our AI photo extender can increase image resolution and process up to four images simultaneously.`
    },
    {
      'question': t`How does AI Photo Extender work?`,
      'answer': t`AI Photo Extender utilizes state-of-the-art AI algorithms to analyze uploaded images and generate additional content. Our AI photo extender's deep learning model ensures that the expanded areas blend seamlessly with the original image, maintaining consistency in style and quality.`
    },
    {
      'question': t`Can AI Photo Extender maintain the original image style?`,
      'answer': t`Absolutely! AI Photo Extender is specifically designed to preserve the original image style throughout the expansion process. Our AI photo extender's algorithms are trained to understand and replicate unique image characteristics, ensuring visual consistency between the original and expanded content.`
    },
    {
      'question': t`Does AI Photo Extender increase image resolution?`,
      'answer': t`Yes, AI Photo Extender not only expands image content but also offers the option to increase pixel dimensions. This feature of our AI photo extender allows users to enhance both the size and quality of their images, resulting in higher resolution outputs.`
    },
    {
      'question': t`How many images can AI Photo Extender process at once?`,
      'answer': t`AI Photo Extender is capable of processing up to four images simultaneously. This multi-image feature of our AI photo extender enables users to efficiently enhance multiple images in a single session, saving time and effort.`
    }
  ]


  return (
    <section className=" mx-auto py-14  bg-gray  text-zinc-900">
      <div className="container md:px-4 mx-auto">
        <div className="grid grid-cols-12 max-w-7xl mx-auto text-center md:text-left">
          <div className="col-span-12 lg:col-span-8 mb-2">
            <h2 className="font-bold text-3xl  md:text-[45px] leading-none mb-6">
              {t`Frequently Asked Questions`}
            </h2>
          </div>
        </div>

        <div className="grid grid-cols-12 gap-6 justify-between max-w-7xl mx-auto">
          <div className="hidden md:block  col-span-12 md:col-span-4 mb-6 md:mb-0">
            <div
              className="bg-center bg-no-repeat bg-cover min-h-[150px] w-full rounded-2xl h-full"
              style={{
                backgroundImage:
                  'url(https://public-image.fafafa.ai/fa-image/2024/05/7258f277dfa7c92eb74c196f08e87ca8.jpg)'
              }}
            ></div>
          </div>
          <div className="col-span-12 md:col-span-8 lg:pl-12">
            {items.map((faq, i) => (
              <FaqItem faq={faq} key={i} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

export default FAQs