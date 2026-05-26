/**
 * ComponentLibrary — filterable + searchable list of all 14 hardware parts.
 * Mirrors gui/tab_component_library.py. Selecting a row sets the active part
 * which ComponentParamTable reads.
 */

import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";

import { api, type ComponentSummary } from "@/api/client";
import { DUR, EASE } from "@/lib/motion";
import { Card } from "@/primitives/Card";

interface ComponentLibraryProps {
  selected: string | null;
  onSelect: (part: string) => void;
}

const ROW_ANIM = {
  hidden: { opacity: 0, y: 4 },
  show: { opacity: 1, y: 0 },
};

export function ComponentLibrary({ selected, onSelect }: ComponentLibraryProps) {
  const [all, setAll] = useState<ComponentSummary[]>([]);
  const [cats, setCats] = useState<string[]>([]);
  const [category, setCategory] = useState<string>("All");
  const [query, setQuery] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listComponents(), api.listComponentCategories()])
      .then(([rows, c]) => {
        setAll(rows);
        setCats(c);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return all.filter((row) => {
      const inCat = category === "All" || row.category === category;
      const inQuery =
        !q ||
        row.part.toLowerCase().includes(q) ||
        row.category.toLowerCase().includes(q);
      return inCat && inQuery;
    });
  }, [all, category, query]);

  return (
    <Card>
      <header className="mb-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Component Library
        </h3>
        <p className="mt-1 text-xs text-slate-500">
          {all.length} parts · {cats.length} categories
        </p>
      </header>

      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-100
                     focus:outline-none focus:ring-2 focus:ring-beam-400/40"
        >
          <option value="All" className="bg-slate-900">All categories</option>
          {cats.map((c) => (
            <option key={c} value={c} className="bg-slate-900">
              {c}
            </option>
          ))}
        </select>
        <input
          type="search"
          placeholder="Search part number…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-100
                     placeholder:text-slate-500
                     focus:outline-none focus:ring-2 focus:ring-beam-400/40"
        />
      </div>

      {error && (
        <p className="mb-3 text-sm text-rose-400">{error}</p>
      )}

      <motion.ul
        initial="hidden"
        animate="show"
        variants={{ show: { transition: { staggerChildren: 0.02 } } }}
        className="max-h-[28rem] divide-y divide-white/5 overflow-y-auto rounded-xl border border-white/10"
      >
        {visible.map((row) => {
          const active = row.part === selected;
          return (
            <motion.li
              key={row.class}
              variants={ROW_ANIM}
              transition={{ duration: DUR.fast, ease: EASE }}
            >
              <button
                type="button"
                onClick={() => onSelect(row.part)}
                className={
                  "flex w-full items-center justify-between px-4 py-3 text-left transition-colors " +
                  (active
                    ? "bg-beam-400/[0.10] text-beam-200"
                    : "hover:bg-white/[0.04]")
                }
              >
                <div className="flex flex-col">
                  <span className="font-mono text-sm">{row.part}</span>
                  <span className="text-xs text-slate-500">{row.class}</span>
                </div>
                <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
                  {row.category}
                </span>
              </button>
            </motion.li>
          );
        })}
        {visible.length === 0 && !error && (
          <li className="px-4 py-6 text-center text-sm text-slate-500">
            No parts match this filter.
          </li>
        )}
      </motion.ul>
    </Card>
  );
}
