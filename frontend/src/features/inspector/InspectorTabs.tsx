/**
 * Shared tab strip for the InspectorRoute. Pure presentational: parent owns
 * the active-tab state and renders the body.
 */

export type InspectorTabId = "schematics" | "validation" | "messages" | "probes";

interface TabSpec {
  id: InspectorTabId;
  label: string;
  badge?: number | string;
}

export function InspectorTabs({
  active,
  tabs,
  onSelect,
}: {
  active: InspectorTabId;
  tabs: TabSpec[];
  onSelect: (id: InspectorTabId) => void;
}) {
  return (
    <div className="flex border-b border-slate-800">
      {tabs.map((t) => {
        const isActive = t.id === active;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onSelect(t.id)}
            className={`relative px-4 py-2 text-sm transition ${
              isActive
                ? "text-slate-100"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span>{t.label}</span>
            {t.badge !== undefined ? (
              <span className="ml-2 inline-block rounded-full bg-slate-700 px-2 py-0.5 text-[10px] font-mono text-slate-200">
                {t.badge}
              </span>
            ) : null}
            {isActive ? (
              <span className="absolute inset-x-2 -bottom-px h-px bg-sky-400" />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
