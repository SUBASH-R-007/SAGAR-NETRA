import React from 'react'
import ReactDOM from 'react-dom/client'
// Bundled faces (offline-first — no external font links). Vite inlines the
// woff2 files into dist/assets at build time.
import '@fontsource/archivo/600.css'
import '@fontsource/archivo/700.css'
import '@fontsource/public-sans/400.css'
import '@fontsource/public-sans/500.css'
import '@fontsource/public-sans/600.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import 'leaflet/dist/leaflet.css'
import './styles.css'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
