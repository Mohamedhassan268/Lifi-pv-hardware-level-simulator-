/**
 * TopBar — instrument-style header strip. Hidden on the landing route.
 *
 * Layout (left → right):
 *   [logo dot] OptiSim   | preset readout (live config)
 *                                  | route buttons (Builder · Sweeps · AC)
 *                                  | Home
 *
 * Dense, hairline-bordered, no shadow or spring. Numeric labels in
 * monospace via the `readout` utility.
 */

import { useConfigStore } from "@/store/configStore";
import { Button } from "@/primitives/Button";
import { useUIStore } from "@/store/uiStore";

export function TopBar() {
  const route = useUIStore((s) => s.route);
  const setRoute = useUIStore((s) => s.setRoute);
  const forms = useUIStore((s) => s.forms);
  const revealed = useUIStore((s) => s.revealed);
  const presetName = useConfigStore((s) => s.presetName);

  return (
    <header
      className="sticky top-0 z-20 flex items-stretch gap-0 border-b border-hair
                 bg-slate-950/95 backdrop-blur"
    >
      <button
        type="button"
        onClick={() => setRoute("landing")}
        className="flex items-center gap-2 border-r border-hair px-3 py-1.5 text-left hover:bg-white/[0.03]"
        aria-label="Home"
      >
        <span className="inline-block h-1.5 w-1.5 bg-beam-400" />
        <span className="text-[12px] font-semibold tracking-wide text-slate-100">
          OptiSim
        </span>
      </button>

      <div className="flex items-center gap-2 border-r border-hair px-3">
        <span className="label-inst">Preset</span>
        <span className={"readout text-[11px] " + (presetName ? "text-slate-200" : "text-slate-500")}>
          {presetName ?? "—"}
        </span>
      </div>

      <span className="flex-1" />

      <div className="flex items-center gap-1 px-2 py-1.5">
        {/* System-level views — always available. */}
        <RouteTab label="Greenhouse" active={route === "greenhouse"} onClick={() => setRoute("greenhouse")} />
        <RouteTab label="Measured" active={route === "measured"} onClick={() => setRoute("measured")} />
        {/* Design tabs — scoped to the task this workspace was opened for. */}
        {forms.block && (
          <RouteTab label="Builder" active={route === "builder"} onClick={() => setRoute("builder")} />
        )}
        {forms.schematic && (
          <RouteTab label="Schematic" active={route === "schematic"} onClick={() => setRoute("schematic")} />
        )}
        {forms.block && (
          <RouteTab label="Setup" active={route === "setup"} onClick={() => setRoute("setup")} />
        )}
        {/* Analysis tabs — appear progressively as results are produced. */}
        {revealed.engine && (
          <RouteTab label="Engine" active={route === "engine"} onClick={() => setRoute("engine")} />
        )}
        {revealed.sweeps && (
          <RouteTab label="Sweeps" active={route === "sweeps"} onClick={() => setRoute("sweeps")} />
        )}
        {revealed.ac && (
          <RouteTab label="AC" active={route === "ac"} onClick={() => setRoute("ac")} />
        )}
        {revealed.inspector && (
          <RouteTab label="Inspector" active={route === "inspector"} onClick={() => setRoute("inspector")} />
        )}
      </div>

      <div className="flex items-center border-l border-hair px-2 py-1.5">
        <Button variant="ghost" onClick={() => setRoute("landing")}>
          Home
        </Button>
      </div>
    </header>
  );
}

function RouteTab({
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
      onClick={onClick}
      className={
        "px-2 py-1 text-[11px] uppercase tracking-wider border-b-2 transition-colors " +
        (active
          ? "border-beam-400 text-beam-300"
          : "border-transparent text-slate-400 hover:text-slate-200")
      }
    >
      {label}
    </button>
  );
}
