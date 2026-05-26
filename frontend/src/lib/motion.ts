/**
 * Shared motion tokens. All routes and components should pull EASE/DUR from
 * here so the app has a single rhythm.
 *
 * EASE matches Tailwind's `ease-smooth` (cubic-bezier(0.4, 0, 0.2, 1)) — the
 * standard "ease-in-out" the user asked for.
 */

export const EASE = [0.4, 0, 0.2, 1] as const;

export const DUR = {
  fast: 0.18,
  base: 0.28,
  slow: 0.45,
} as const;
