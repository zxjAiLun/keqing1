import type { CSSProperties, ReactNode } from "react";

export function PageShell({
  children,
  width = 1120,
}: {
  children: ReactNode;
  width?: number;
}) {
  return (
    <div style={{ minHeight: "100%", padding: "24px" }}>
      <div style={{ maxWidth: width, margin: "0 auto" }}>{children}</div>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
        gap: 16,
        flexWrap: "wrap",
        marginBottom: 24,
      }}
    >
      <div style={{ minWidth: 0 }}>
        {eyebrow && (
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              marginBottom: 8,
            }}
          >
            {eyebrow}
          </div>
        )}
        <h1 style={{ fontSize: 28, fontWeight: 800, color: "var(--text-primary)", marginBottom: 8 }}>
          {title}
        </h1>
        {description && (
          <div style={{ maxWidth: 760, fontSize: 14, lineHeight: 1.6, color: "var(--text-secondary)" }}>
            {description}
          </div>
        )}
      </div>
      {actions && <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>{actions}</div>}
    </div>
  );
}

export function SectionTitle({ title, description }: { title: string; description?: ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: description ? 4 : 0 }}>
        {title}
      </div>
      {description && <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{description}</div>}
    </div>
  );
}

export function MetricCard({ label, value, tone = "accent" }: { label: string; value: ReactNode; tone?: "accent" | "success" | "warning"; }) {
  const color =
    tone === "success" ? "var(--success)" : tone === "warning" ? "var(--warning)" : "var(--accent)";
  return (
    <div
      style={{
        background: "var(--card-bg)",
        border: "1px solid var(--card-border)",
        borderRadius: "var(--radius-md)",
        padding: "14px 16px",
        minWidth: 120,
        boxShadow: "var(--card-shadow)",
      }}
    >
      <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "Menlo, monospace" }}>{value}</div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
    </div>
  );
}

export const subtleButtonStyle: CSSProperties = {
  height: 38,
  padding: "0 14px",
  borderRadius: "var(--radius-sm)",
  border: "1px solid var(--border)",
  background: "var(--card-bg)",
  color: "var(--text-primary)",
  fontSize: 13,
  fontWeight: 600,
  cursor: "pointer",
};
