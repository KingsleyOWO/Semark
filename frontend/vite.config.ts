import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5070,
    // Opt-in only: an explicit `true` here would let any DNS-rebound hostname reach
    // this dev server (and, through the /api proxy, the auth-less backend). Default
    // (no env var) falls back to Vite's built-in policy, which already accepts
    // localhost and direct IP-literal Hosts, so the documented IP/localhost access
    // patterns keep working; named-host access requires an explicit opt-in.
    allowedHosts: (process.env.SEMARK_FRONTEND_ALLOWED_HOSTS ?? '').split(',').map(s => s.trim()).filter(Boolean),
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://localhost:8585',
        changeOrigin: true,
      },
    },
  },
})
