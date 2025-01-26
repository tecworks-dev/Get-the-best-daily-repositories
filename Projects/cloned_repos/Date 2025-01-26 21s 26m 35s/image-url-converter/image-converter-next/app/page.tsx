"use client";

import { Features } from "@/components/home/Features";
import { siteConfig } from "@/config/site";
import { motion } from "framer-motion";
import { Clipboard, Image as ImageIcon, Link2 } from "lucide-react";
import { useState } from "react";
import toast from "react-hot-toast";

export default function Home() {
  const [imageUrl, setImageUrl] = useState("");
  const [uploadedUrl, setUploadedUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setUploadedUrl("");

    try {
      let response;

      if (selectedFile) {
        // Handle file upload
        const formData = new FormData();
        formData.append("file", selectedFile);

        response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });
      } else if (imageUrl) {
        // Handle URL upload (existing logic)
        response = await fetch("/api/upload", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ imageUrl }),
        });
      } else {
        throw new Error("Please provide an image URL or file");
      }

      const data = await response.json();

      if (data.success) {
        setUploadedUrl(data.url);
        toast.success("Image converted successfully!");
      } else {
        setError(data.error || "Conversion failed");
        toast.error(data.error || "Conversion failed");
      }
    } catch (err) {
      const errorMessage = "An error occurred during processing";
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied to clipboard");
    } catch (err) {
      toast.error("Failed to copy");
    }
  };

  // Add file input handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setImageUrl(""); // Clear URL input when file is selected
    }
  };

  return (
    <div className="min-h-screen ">
      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center space-y-8 mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-6">
              {siteConfig.name}
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              {siteConfig.description}
            </p>
          </motion.div>

          {/* Steps */}
          <div className="flex justify-center items-center space-x-8 text-gray-500 dark:text-gray-400">
            <Step icon={<Link2 />} text="Paste Image URL" number={1} />
            <Arrow />
            <Step icon={<ImageIcon />} text="Auto Convert" number={2} />
            <Arrow />
            <Step icon={<Clipboard />} text="Get Permanent Link" number={3} />
          </div>
        </div>

        {/* Converter Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="max-w-3xl mx-auto"
        >
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 border border-gray-200 dark:border-gray-700">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Upload Image File
                  </label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>

                <div className="text-center text-gray-500 dark:text-gray-400">
                  OR
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Enter Image URL
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={imageUrl}
                      onChange={(e) => {
                        setImageUrl(e.target.value);
                        setSelectedFile(null); // Clear file input when URL is entered
                      }}
                      placeholder="https://example.com/image.jpg"
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white pr-12"
                    />
                    <ImageIcon
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400"
                      size={20}
                    />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3 px-4 rounded-lg text-white font-medium transition-all transform hover:scale-[1.02] ${
                  loading
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <LoadingSpinner />
                    <span className="ml-2">Processing...</span>
                  </span>
                ) : (
                  "Convert Now"
                )}
              </button>
            </form>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-6 p-4 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg border border-red-100 dark:border-red-800"
              >
                <p className="flex items-center">
                  <span className="mr-2">⚠️</span>
                  {error}
                </p>
              </motion.div>
            )}

            {/* Success Result */}
            {uploadedUrl && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 p-6 bg-green-50 dark:bg-green-900/30 rounded-lg border border-green-100 dark:border-green-800"
              >
                <h3 className="text-green-600 dark:text-green-400 font-medium mb-4 flex items-center">
                  <span className="mr-2">✨</span>
                  Conversion Successful!
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={uploadedUrl}
                      readOnly
                      className="flex-1 p-3 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg font-mono text-sm"
                    />
                    <button
                      onClick={() => handleCopy(uploadedUrl)}
                      className="px-4 py-3 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 border border-gray-200 dark:border-gray-600 transition-colors"
                    >
                      <Clipboard size={18} />
                    </button>
                  </div>
                  <PreviewCard url={uploadedUrl} />
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>

        <Features />
      </main>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function Step({
  icon,
  text,
  number,
}: {
  icon: React.ReactNode;
  text: string;
  number: number;
}) {
  return (
    <div className="flex flex-col items-center space-y-2">
      <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center relative">
        <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-blue-500 text-white text-sm flex items-center justify-center">
          {number}
        </div>
        {icon}
      </div>
      <span className="text-sm">{text}</span>
    </div>
  );
}

function Arrow() {
  return (
    <div className="text-gray-300 dark:text-gray-600">
      <svg
        className="w-6 h-6"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M14 5l7 7m0 0l-7 7m7-7H3"
        />
      </svg>
    </div>
  );
}

function PreviewCard({ url }: { url: string }) {
  return (
    <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600">
      <div className="aspect-w-16 aspect-h-9">
        <img
          src={url}
          alt="Converted image preview"
          className="object-cover w-full h-full"
          onError={(e) => {
            e.currentTarget.src = "/placeholder.png";
          }}
        />
      </div>
    </div>
  );
}
