/**
 * PipelineTransport — Run / Step / Cancel controls for stepwise execution.
 *
 * Visual style: engineering-instrument. Mono labels, hairline borders, no
 * gradients. Each control double-functions as a status indicator (current
 * stage highlighted when paused).
 */

import { usePipelineExecutionStore } from "@/store/pipelineExecutionStore";

import { useGlassBoxPipeline } from "./useGlassBoxPipeline";

export function PipelineTransport() {
  const { start, step, cancel } = useGlassBoxPipeline();
  const running = usePipelineExecutionStore((s) => s.running);
  const pausedAt = usePipelineExecutionStore((s) => s.pausedAt);
  const mode = usePipelineExecutionStore((s) => s.mode);
  const setMode = usePipelineExecutionStore((s) => s.setMode);

  return (
    <div className="flex items-center gap-2 border border-slate-800/80 bg-slate-950/80 px-3 py-2 font-mono text-[11px] text-slate-200">
      <ModeToggle mode={mode} onChange={setMode} disabled={running} />
      <Divider />
      <Btn
        label="Run"
        onClick={() => start(mode)}
        disabled={running && pausedAt === null}
        accent="cyan"
      />
      <Btn
        label="Step"
        onClick={step}
        disabled={!running || pausedAt === null}
        accent="amber"
      />
      <Btn
        label="Cancel"
        onClick={cancel}
        disabled={!running}
        accent="rose"
      />
      <Divider />
      <Status running={running} pausedAt={pausedAt} mode={mode} />
    </div>
  );
}

// ---------------------------------------------------------------------------

interface ModeToggleProps {
  mode: "continuous" | "stepwise";
  onChange: (m: "continuous" | "stepwise") => void;
  disabled: boolean;
}

function ModeToggle({ mode, onChange, disabled }: ModeToggleProps) {
  return (
    <div className="inline-flex border border-slate-800">
      {(["continuous", "stepwise"] as const).map((m) => (
        <button
          key={m}
          type="button"
          disabled={disabled}
          onClick={() => onChange(m)}
          className={[
            "px-2 py-0.5 text-[10px] uppercase tracking-[0.15em]",
            mode === m
              ? "bg-slate-800 text-cyan-200"
              : "text-slate-500 hover:text-slate-300",
            disabled ? "cursor-not-allowed opacity-50" : "",
          ].join(" ")}
        >
          {m}
        </button>
      ))}
    </div>
  );
}

interface BtnProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  accent: "cyan" | "amber" | "rose";
}

function Btn({ label, onClick, disabled, accent }: BtnProps) {
  const accentClass = {
    cyan: "text-cyan-300 hover:text-cyan-100",
    amber: "text-amber-300 hover:text-amber-100",
    rose: "text-rose-300 hover:text-rose-100",
  }[accent];
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={[
        "border border-slate-800 px-2 py-0.5 text-[10px] uppercase tracking-[0.18em]",
        accentClass,
        disabled ? "cursor-not-allowed opacity-30" : "hover:border-slate-700",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

function Divider() {
  return <div className="h-4 w-px bg-slate-800" />;
}

interface StatusProps {
  running: boolean;
  pausedAt: string | null;
  mode: "continuous" | "stepwise";
}

function Status({ running, pausedAt, mode }: StatusProps) {
  let text: string;
  if (!running) text = "idle";
  else if (pausedAt) text = `paused after ${pausedAt}`;
  else text = `running (${mode})`;
  return (
    <span className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
      {text}
    </span>
  );
}
