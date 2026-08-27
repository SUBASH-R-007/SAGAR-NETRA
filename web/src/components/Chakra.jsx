// Ashoka Chakra — the 24-spoke wheel of the national flag, drawn as precise
// inline geometry: one rim circle, 24 spokes at exact 15-degree intervals, a
// small hub. Stroke is currentColor; the parent sets `color: var(--chakra)`.
//
// NOTE FOR THE TEAM: this is deliberately the Chakra, NOT the State Emblem of
// India (the lion capital) — use of the State Emblem is legally restricted
// under the State Emblem of India (Prohibition of Improper Use) Act, 2005.
// For the final submission, replace this <Chakra /> slot with the official
// emblem asset once the team has the authorised artwork.

const SPOKES = Array.from({ length: 24 }, (_, i) => i * 15)

export default function Chakra({ size = 44 }) {
  return (
    <svg
      className="chakra"
      width={size}
      height={size}
      viewBox="0 0 44 44"
      role="img"
      aria-label="Ashoka Chakra"
      focusable="false"
    >
      {/* rim */}
      <circle cx="22" cy="22" r="20" fill="none" stroke="currentColor" strokeWidth="2" />
      {/* 24 spokes, hub edge to inner rim, one line rotated by i * 15deg */}
      {SPOKES.map((deg) => (
        <line
          key={deg}
          x1="22"
          y1="18.6"
          x2="22"
          y2="3.4"
          stroke="currentColor"
          strokeWidth="1.1"
          transform={`rotate(${deg} 22 22)`}
        />
      ))}
      {/* hub */}
      <circle cx="22" cy="22" r="3" fill="currentColor" />
    </svg>
  )
}
