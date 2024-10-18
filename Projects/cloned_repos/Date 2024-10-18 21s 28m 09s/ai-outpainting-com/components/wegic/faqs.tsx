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
      'question':
        t`What is AI Outpainting?`,
      'answer':
        t`AI Outpainting is a website that utilizes advanced AI technology to expand and enhance images uploaded by users. It can extend the content of images in four directions (up, down, left, and right), while maintaining the original style of the image. It also has the capability to increase the pixel dimensions of the images and simultaneously expand up to four images.`
    }
    ,
    {
      'question': t`How does AI Outpainting work?`,
      'answer':
        t`AI Outpainting employs cutting-edge AI algorithms to analyze the uploaded images and generate additional content based on the existing image data. By utilizing a deep learning model, the AI technology is able to seamlessly expand the image content in a way that remains consistent with the original style.`
    }
    ,
    {
      'question':
        t`Can AI Outpainting maintain the style of the original image while expanding it?`,
      'answer':
        t`Yes, AI Outpainting is designed to preserve the style of the original image throughout the expansion process. The AI algorithms are trained to understand and replicate the unique characteristics of the image, ensuring that the expanded content remains visually consistent with the original.`
    }
    ,
    {
      'question':
        t`Can AI Outpainting increase the pixel dimensions of an image?`,
      'answer':
        t`
Absolutely! In addition to expanding the content of the image, AI Outpainting also provides the option to increase the pixel dimensions of the uploaded images. This allows users to enhance the overall resolution and quality of their images.`
    }
    ,
    {
      'question':
        t`How many images can I expand at once using AI Outpainting?`,
      'answer':
        t`AI Outpainting allows users to simultaneously expand up to five images. This multi-image expansion feature enables users to conveniently enhance multiple images in a time-efficient manner.`
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