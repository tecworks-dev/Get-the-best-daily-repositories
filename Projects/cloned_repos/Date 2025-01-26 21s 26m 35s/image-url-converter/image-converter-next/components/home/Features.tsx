import { motion } from "framer-motion";
import { FeatureCard } from "./FeatureCard";

export function Features() {
  return (
    <section className="py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Powerful Features
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Why choose our image conversion tool
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + index * 0.1 }}
            >
              <FeatureCard {...feature} />
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
export const features = [
  {
    title: "Permanent Storage",
    description:
      "Built on Cloudflare R2 storage, ensuring your image links remain valid permanently",
    icon: "🗄️",
    color: "from-blue-500 to-blue-600",
  },
  {
    title: "Global CDN",
    description:
      "Leveraging Cloudflare's CDN network for fast access worldwide",
    icon: "🚀",
    color: "from-purple-500 to-purple-600",
  },
  {
    title: "Free to Use",
    description: "Developed using Cloudflare's free tier, available at no cost",
    icon: "💎",
    color: "from-green-500 to-green-600",
  },
  {
    title: "Secure & Reliable",
    description:
      "Advanced cloud storage technology ensuring data security and stable access",
    icon: "🛡️",
    color: "from-orange-500 to-orange-600",
  },
  {
    title: "User Friendly",
    description:
      "Intuitive interface design - just paste the image URL to start conversion",
    icon: "✨",
    color: "from-pink-500 to-pink-600",
  },
  {
    title: "Open Source",
    description: "Completely open source code, free to deploy and customize",
    icon: "📖",
    color: "from-teal-500 to-teal-600",
  },
] as const;

export type Feature = (typeof features)[number];
