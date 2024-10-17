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
      'question': t`What is Generative Fill AI?`,
      'answer': t`Generative Fill AI is an innovative online tool that leverages advanced artificial intelligence to expand and enhance images. This cutting-edge generative fill AI technology can seamlessly extend image content in all directions while maintaining the original style. Our generative fill AI tool also offers image resolution enhancement and can process multiple images simultaneously.`
    },
    {
      'question': t`How does Generative Fill AI work?`,
      'answer': t`Generative Fill AI employs state-of-the-art algorithms to analyze uploaded images and generate additional content. The deep learning model behind our generative fill AI ensures that the expanded areas blend seamlessly with the original image, maintaining consistency in style and quality throughout the process.`
    },
    {
      'question': t`Can Generative Fill AI maintain the original image style?`,
      'answer': t`Absolutely! Our generative fill AI is specifically designed to preserve the original image style throughout the expansion process. The algorithms powering this generative fill AI technology are trained to understand and replicate unique image characteristics, ensuring visual consistency between the original and expanded content.`
    },
    {
      'question': t`Does Generative Fill AI increase image resolution?`,
      'answer': t`Yes, our generative fill AI not only expands image content but also offers the option to increase pixel dimensions. This feature allows users to enhance both the size and quality of their images, resulting in higher resolution outputs through the power of generative fill AI technology.`
    },
    {
      'question': t`How many images can Generative Fill AI process at once?`,
      'answer': t`Our generative fill AI tool is capable of processing up to four images simultaneously. This multi-image feature enables users to efficiently enhance multiple images in a single session, showcasing the versatility and efficiency of generative fill AI technology.`
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