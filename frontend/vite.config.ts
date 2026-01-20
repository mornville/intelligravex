import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow accessing dev server from LAN hostnames (guards against DNS rebinding by default).
    // If you visit http://ashutosh-jha-macbook-pro:5173/, this must be allowed.
    allowedHosts: [
      'ashutosh-jha-macbook-pro',
      'mornvillepi4.local',
      'localhost',
      '127.0.0.1',
    ],
  },
  preview: {
    // Allow accessing preview server from LAN hostnames.
    allowedHosts: ['mornvillepi4.local', 'localhost', '127.0.0.1'],
  },
})
