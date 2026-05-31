/**
 * FormToggle — segmented Schematic | Block switch, floating at the top of the
 * workspace. Shown only when the current project enabled BOTH forms (otherwise
 * there's nothing to toggle), and only on the builder/schematic routes.
 */

import { useUIStore } from "@/store/uiStore";

export function FormToggle() {
  const route = useUIStore((s) => s.route);
  const setRoute = useUIStore((s) => s.setRoute);
  const forms = useUIStore((s) => s.forms);

  const bothEnabled = forms.schematic && forms.block;
  const onForm = route === "schematic" || route === "builder";
  if (!bothEnabled || !onForm) return null;

  return (
    <div
      className="pointer-events-auto absolute left-1/2 top-2 z-30 -translate-x-1/2
                 rounded-lg border border-white/10 bg-slate-900/90 p-0.5 backdrop-blur
                 shadow-[0_4px_20px_-8px_rgba(0,0,0,0.6)]"
      role="tablist"
      aria-label="Project form"
    >
      <Seg label="Schematic" active={route === "schematic"} onClick={() => setRoute("schematic")} />
      <Seg label="Block diagram" active={route === "builder"} onClick={() => setRoute("builder")} />
    </div>
  );
}

function Seg({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={
        "rounded-md px-3 py-1 text-[11px] font-medium uppercase tracking-wider transition-colors " +
        (active
          ? "bg-beam-400 text-slate-950"
          : "text-slate-400 hover:bg-white/[0.06] hover:text-slate-200")
      }
    >
      {label}
    </button>
  );
}
