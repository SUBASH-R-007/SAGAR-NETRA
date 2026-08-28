import React from 'react'
import ReactDOM from 'react-dom/client'
// Bundled faces (offline-first — no external font links). Vite inlines the
// woff2 files into dist/assets at build time.
import '@fontsource/noto-sans/400.css'
import '@fontsource/noto-sans/500.css'
import '@fontsource/noto-sans/600.css'
import '@fontsource/noto-sans/700.css'
import '@fontsource/noto-sans-devanagari/400.css'
import '@fontsource/noto-sans-devanagari/600.css'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import 'leaflet/dist/leaflet.css'
import './styles.css'
// Per-view sheets: same tokens, kept separate so each view's rules stay
// findable next to the component that uses them.
import './styles/overview.css'
import './styles/recovery.css'
import './styles/system.css'
import './styles/enhancements.css'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
