/**
 * ChannelCanvas — SVG replacement for gui/channel_canvas.py::_LinkDiagram.
 * Renders TX LED + emission cone + RX PV cell + ground line. All transforms
 * via inline `transform` strings on `<g>`, so updates from slider drags are
 * effectively free (single SVG repaint).
 */

import { useLinkBudgetInput } from "@/store/configStore";

export function ChannelCanvas() {
  const cfg = useLinkBudgetInput();
  // Map the half-angle to a triangle opening width (clamped for visual sanity).
  const coneHalf = Math.min(140, Math.max(20, cfg.led_half_angle_deg * 8));
  const txX = 70;
  const rxX = 70 + Math.min(540, Math.max(120, cfg.distance_m * 700));
  const txAngle = -cfg.tx_angle_deg;
  const rxTilt = cfg.rx_tilt_deg;

  return (
    <svg viewBox="0 0 700 300" className="h-72 w-full">
      {/* Ground line */}
      <line x1="20" y1="240" x2="680" y2="240" stroke="rgba(255,255,255,0.15)" />

      {/* TX assembly */}
      <g transform={`translate(${txX} 150) rotate(${txAngle})`}>
        <circle r="14" fill="#67e8f9" />
        <circle r="22" fill="none" stroke="rgba(103,232,249,0.35)" />
        <polygon
          points={`0,0 ${rxX - txX},${-coneHalf / 2} ${rxX - txX},${coneHalf / 2}`}
          fill="url(#beamGradient)"
          opacity="0.65"
        />
      </g>

      {/* RX PV cell */}
      <g transform={`translate(${rxX} 150) rotate(${rxTilt})`}>
        <rect x="-10" y="-32" width="20" height="64" rx="3" fill="#fcd34d" />
        <rect x="-10" y="-32" width="20" height="64" rx="3" fill="none" stroke="rgba(252,211,77,0.6)" />
      </g>

      {/* Distance label */}
      <text
        x={(txX + rxX) / 2}
        y="270"
        textAnchor="middle"
        className="fill-slate-400"
        fontSize="11"
      >
        {(cfg.distance_m * 100).toFixed(1)} cm
      </text>

      <defs>
        <linearGradient id="beamGradient" x1="0" x2="1">
          <stop offset="0%" stopColor="#67e8f9" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#fcd34d" stopOpacity="0.15" />
        </linearGradient>
      </defs>
    </svg>
  );
}
