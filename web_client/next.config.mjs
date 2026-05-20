/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/rag',
        destination: 'http://localhost:8000/rag',
      },
      {
        source: '/api/agent',
        destination: 'http://localhost:8000/agent',
      },
    ];
  },

  httpAgentOptions: {
    keepAlive: true,
  },

  experimental: {
    proxyTimeout: 120000, // 120s
  },
};

export default nextConfig;
