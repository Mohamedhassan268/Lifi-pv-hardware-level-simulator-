import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // LiFi-PV palette: cyan for light/data, amber for harvested power.
        beam: {
          50: "#ecfeff",
          300: "#67e8f9",
          400: "#22d3ee",
          500: "#06b6d4",
        },
        harvest: {
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
        },
      },
      transitionTimingFunction: {
        // Matches the EASE token in src/lib/motion.ts (cubic-bezier(0.4,0,0.2,1)).
        smooth: "cubic-bezier(0.4, 0, 0.2, 1)",
      },
    },
  },
  plugins: [],
} satisfies Config;
