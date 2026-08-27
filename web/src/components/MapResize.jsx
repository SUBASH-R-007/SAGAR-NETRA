import { useEffect } from 'react'
import { useMap } from 'react-leaflet'

// Leaflet computes its pixel size ONCE at initialization. If the container's
// layout has not settled at that instant (tab just mounted, flexbox still
// resolving, fonts loading), the map believes it is 0x0 and renders a blank
// panel with tiles parked off-screen — the classic "map is not visible" bug,
// timing-dependent and therefore invisible on fast reloads. This helper
// re-measures on the next two animation frames and then keeps the map honest
// with a ResizeObserver for every later layout change (window resize, rail
// collapse, tab switches that reflow the content column).
export default function MapResize() {
  const map = useMap()
  useEffect(() => {
    const invalidate = () => map.invalidateSize({ animate: false })
    const raf1 = requestAnimationFrame(() => {
      invalidate()
      requestAnimationFrame(invalidate)
    })
    const container = map.getContainer()
    const observer = new ResizeObserver(invalidate)
    observer.observe(container)
    if (container.parentElement) observer.observe(container.parentElement)
    return () => {
      cancelAnimationFrame(raf1)
      observer.disconnect()
    }
  }, [map])
  return null
}
