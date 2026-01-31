import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow accessing dev server from LAN hostnames (guards against DNS rebinding by default).
    // Add custom LAN hostnames here if needed.
    allowedHosts: ['localhost', '127.0.0.1'],
  },
  preview: {
    // Allow accessing preview server from LAN hostnames.
    allowedHosts: ['localhost', '127.0.0.1'],
  },
})
