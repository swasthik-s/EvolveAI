import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    // On Vercel, /api/chat is the Python function. Locally, proxy it to uvicorn.
    if (process.env.VERCEL) return [];
    const origin = process.env.EVOLVEAI_DEV_API_ORIGIN || "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${origin}/api/:path*` }];
  },
};

export default nextConfig;
