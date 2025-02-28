/** @type {import('next').NextConfig} */
const nextConfig = {
  // ... your existing config ...
  webpack: (config, { isServer }) => {
    config.ignoreWarnings = [
      { module: /node_modules\/punycode/ }
    ];
    return config;
  },
}

module.exports = nextConfig 