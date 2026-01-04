import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const DATA_URL = "/data/all_tests_summary.csv";
const POLL_MS = 300000;
const numberFields = [
  "t50_mean",
  "t90_mean",
  "t100_mean",
  "messages_mean",
  "compete_p_t90_mean",
  "lambda_t100_mean",
  "p_ge1_t100_mean",
  "p_ge2_t100_mean",
  "security_margin_t50_mean",
];

const parseCsv = (text) =>
  new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: ({ data }) => {
        const rows = data
          .filter((row) => row.scenario && row.protocol)
          .map((row) => {
            const parsed = { ...row };
            numberFields.forEach((key) => {
              const raw = row[key];
              parsed[key] = raw === undefined || raw === "" ? null : Number(raw);
            });
            return parsed;
          });
        resolve(rows);
      },
      error: reject,
    });
  });

const formatNumber = (value, digits = 3) => {
  if (value === null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
};

const humanize = (value) => (value ? value.replaceAll("_", " ") : "");

const protocolInfo = {
  "naive": "Floods full blocks to all neighbors immediately. High bandwidth, simple behavior.",
  "two-phase": "Announce/request flow before sending full block payloads. Reduces bandwidth at the cost of extra round trips.",
  "push": "Push-style gossip: a node proactively sends blocks to a fanout of peers.",
  "pull": "Pull-style gossip: peers periodically request blocks from others.",
  "push-pull": "Hybrid gossip using both push and pull to improve coverage under churn.",
  "bitcoin-compact": "Compact block relay: send short block then request missing transactions if reconstruction fails.",
};

const scenarioInfo = {
  "baseline_naive": "Baseline with naive flooding on the default random-regular topology.",
  "baseline_two_phase": "Baseline using the two-phase announce/request protocol.",
  "push_fanout": "Push gossip with limited fanout to reduce message overhead.",
  "pull_interval": "Pull gossip with periodic requests; higher latency but structured polling.",
  "push_pull": "Hybrid push-pull gossip for resilience and faster coverage.",
  "scale_free": "Two-phase protocol on a scale-free topology (hub-heavy network).",
  "small_world": "Two-phase protocol on a small-world topology with rewiring.",
  "relay_overlay": "Two-phase with a relay overlay that improves latency/bandwidth between relay nodes.",
  "compact_blocks": "Compact block protocol with mempool overlap assumptions.",
  "bottlenecks": "Two-phase protocol with injected slow nodes (latency/bandwidth bottlenecks).",
  "churn_delay": "Push-pull protocol under churn and additional latency/bandwidth delays.",
  "macro_metrics": "Baseline two-phase run used to compute macro security metrics.",
};

const graphNodes = [
  { id: "A", x: 60, y: 60 },
  { id: "B", x: 180, y: 30 },
  { id: "C", x: 300, y: 60 },
  { id: "D", x: 90, y: 160 },
  { id: "E", x: 220, y: 140 },
  { id: "F", x: 360, y: 150 },
  { id: "G", x: 70, y: 260 },
  { id: "H", x: 200, y: 260 },
  { id: "I", x: 320, y: 260 },
  { id: "J", x: 420, y: 210 },
];

const graphEdges = [
  ["A", "B"],
  ["A", "D"],
  ["B", "C"],
  ["B", "E"],
  ["C", "F"],
  ["D", "E"],
  ["D", "G"],
  ["E", "F"],
  ["E", "H"],
  ["F", "J"],
  ["G", "H"],
  ["H", "I"],
  ["I", "J"],
];

const findNode = (id) => graphNodes.find((node) => node.id === id);

const makeSteps = (variants) => variants.map((step, index) => ({
  t: step.t ?? index * 1.2,
  nodes: step.nodes,
  caption: step.caption,
  badge: step.badge,
}));

const demoPresets = {
  protocol: {
    "naive": {
      title: "Naive flooding",
      summary: "All neighbors forward full blocks immediately.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source broadcasts full block." },
        { t: 1, nodes: ["A", "B", "D"], caption: "Neighbors relay to all peers.", badge: "block" },
        { t: 2, nodes: ["A", "B", "C", "D", "E", "G"], caption: "Flood expands rapidly." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "High message overhead." },
        { t: 4, nodes: graphNodes.map((n) => n.id), caption: "Full saturation." },
      ]),
    },
    "two-phase": {
      title: "Two-phase announce/request",
      summary: "Announce, then request full payload to reduce bandwidth.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Announce from source.", badge: "announce" },
        { t: 1, nodes: ["A", "B", "D"], caption: "Peers request the block.", badge: "request" },
        { t: 2, nodes: ["A", "B", "D"], caption: "Payload arrives to requesters.", badge: "payload" },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "G"], caption: "Announce propagates again." },
        { t: 4, nodes: ["A", "B", "C", "D", "E", "F", "G", "H", "I"], caption: "Coverage grows with fewer bytes." },
      ]),
    },
    "push": {
      title: "Push gossip",
      summary: "Nodes push blocks to a limited fanout.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source pushes to fanout." },
        { t: 1, nodes: ["A", "B", "D"], caption: "Each node pushes to a few peers." },
        { t: 2, nodes: ["A", "B", "C", "D", "E", "G"], caption: "Spread continues." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "Fanout limits reduce overhead." },
        { t: 4, nodes: ["A", "B", "C", "D", "E", "F", "G", "H", "I"], caption: "Some nodes lag behind." },
      ]),
    },
    "pull": {
      title: "Pull gossip",
      summary: "Nodes poll for blocks on a timer.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source holds new block." },
        { t: 1.5, nodes: ["A", "B"], caption: "First pull request succeeds.", badge: "pull" },
        { t: 3, nodes: ["A", "B", "D"], caption: "Polling slowly fans out." },
        { t: 4.5, nodes: ["A", "B", "C", "D", "E"], caption: "More pulls complete." },
        { t: 6, nodes: ["A", "B", "C", "D", "E", "F", "G"], caption: "Latency increases with polling." },
      ]),
    },
    "push-pull": {
      title: "Push-pull gossip",
      summary: "Hybrid push and pull for resilience.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source pushes while others pull." },
        { t: 1, nodes: ["A", "B", "D"], caption: "Push reaches near neighbors." },
        { t: 2, nodes: ["A", "B", "C", "D", "E"], caption: "Pull catches missing peers." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "Hybrid keeps coverage strong." },
        { t: 4, nodes: ["A", "B", "C", "D", "E", "F", "G", "H", "I"], caption: "Fast and resilient spread." },
      ]),
    },
    "bitcoin-compact": {
      title: "Bitcoin compact blocks",
      summary: "Compact block + missing tx reconciliation.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Compact block sent.", badge: "compact" },
        { t: 1, nodes: ["A", "B", "D"], caption: "Reconstruction succeeds for some." },
        { t: 2, nodes: ["A", "B", "D", "E"], caption: "Missing tx requested.", badge: "missing tx" },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "G"], caption: "Compact + missing tx completes." },
        { t: 4, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "Bandwidth saved overall." },
      ]),
    },
  },
  scenario: {
    "baseline_naive": { inherits: "naive", title: "Baseline naive", summary: "Default topology with naive flooding." },
    "baseline_two_phase": { inherits: "two-phase", title: "Baseline two-phase", summary: "Default topology with announce/request." },
    "push_fanout": { inherits: "push", title: "Push fanout", summary: "Reduced fanout push gossip." },
    "pull_interval": { inherits: "pull", title: "Pull interval", summary: "Polling-based propagation." },
    "push_pull": { inherits: "push-pull", title: "Push-pull hybrid", summary: "Hybrid protocol baseline." },
    "scale_free": {
      inherits: "two-phase",
      title: "Scale-free topology",
      summary: "Hub nodes accelerate propagation.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source hits a hub." },
        { t: 1, nodes: ["A", "B", "C", "D"], caption: "Hub fans out quickly.", badge: "hub" },
        { t: 2, nodes: ["A", "B", "C", "D", "E", "F", "G"], caption: "High-degree nodes dominate." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F", "G", "H", "I"], caption: "Fast coverage on hubs." },
      ]),
    },
    "small_world": {
      inherits: "two-phase",
      title: "Small-world topology",
      summary: "Local clusters with occasional shortcuts.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Local cluster spreads first." },
        { t: 1.2, nodes: ["A", "B", "D"], caption: "Cluster fills in." },
        { t: 2.4, nodes: ["A", "B", "C", "D", "E"], caption: "Shortcut bridges clusters." },
        { t: 3.6, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "Rapid global reach." },
      ]),
    },
    "relay_overlay": {
      inherits: "two-phase",
      title: "Relay overlay",
      summary: "Fast relay nodes carry priority traffic.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Relay nodes receive first.", badge: "relay" },
        { t: 1, nodes: ["A", "B", "E"], caption: "Overlay edges speed delivery." },
        { t: 2, nodes: ["A", "B", "C", "E", "F"], caption: "Relay backbone spreads." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F", "H"], caption: "Overlay feeds base network." },
      ]),
    },
    "compact_blocks": { inherits: "bitcoin-compact", title: "Compact blocks", summary: "Compact reconstruction with mempool overlap." },
    "bottlenecks": {
      inherits: "two-phase",
      title: "Bottlenecks",
      summary: "Slow nodes delay propagation.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source sends announce." },
        { t: 1.5, nodes: ["A", "B", "D"], caption: "Bottleneck slows edge.", badge: "slow link" },
        { t: 3, nodes: ["A", "B", "C", "D", "E"], caption: "Payload takes longer." },
        { t: 4.5, nodes: ["A", "B", "C", "D", "E", "F", "G"], caption: "Tail latency increases." },
      ]),
    },
    "churn_delay": {
      inherits: "push-pull",
      title: "Churn + delay",
      summary: "Nodes drop or delay messages.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Source pushes/pulls." },
        { t: 1, nodes: ["A", "B"], caption: "Some peers offline.", badge: "churn" },
        { t: 2.5, nodes: ["A", "B", "D"], caption: "Delayed links slow spread." },
        { t: 4, nodes: ["A", "B", "C", "D", "E"], caption: "Recovery via pull." },
      ]),
    },
    "macro_metrics": {
      inherits: "two-phase",
      title: "Macro metrics",
      summary: "Security metrics derived from timing tails.",
      steps: makeSteps([
        { t: 0, nodes: ["A"], caption: "Baseline propagation run." },
        { t: 1.4, nodes: ["A", "B", "D"], caption: "Track t50/t90/t100." },
        { t: 3, nodes: ["A", "B", "C", "D", "E", "F"], caption: "Compute competing block risk." },
        { t: 4.2, nodes: ["A", "B", "C", "D", "E", "F", "G", "H"], caption: "Security margin derived." },
      ]),
    },
  },
};

const buildDemo = (kind, key) => {
  if (kind === "protocol") return demoPresets.protocol[key];
  const scenario = demoPresets.scenario[key];
  if (!scenario) return null;
  const base = demoPresets.protocol[scenario.inherits];
  return {
    title: scenario.title,
    summary: scenario.summary,
    steps: scenario.steps ?? base.steps,
  };
};

function App() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [scenario, setScenario] = useState("all");
  const [selectedProtocols, setSelectedProtocols] = useState([]);
  const [search, setSearch] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [theme, setTheme] = useState("nebula");
  const [sortKey, setSortKey] = useState("t90_mean");
  const [sortDir, setSortDir] = useState("asc");
  const [demoKey, setDemoKey] = useState(null);
  const [demoKind, setDemoKind] = useState(null);
  const [demoPlaying, setDemoPlaying] = useState(true);
  const [demoTime, setDemoTime] = useState(0);
  const [demoSpeed, setDemoSpeed] = useState(1);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (!demoKey) return;
    let frameId = null;
    let last = performance.now();

    const tick = (now) => {
      if (demoPlaying) {
        const delta = (now - last) / 1000;
        setDemoTime((prev) => prev + delta * demoSpeed);
      }
      last = now;
      frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);
    return () => {
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [demoKey, demoPlaying, demoSpeed]);

  useEffect(() => {
    let intervalId = null;

    const load = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${DATA_URL}?t=${Date.now()}`, {
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Failed to load: ${response.status}`);
        }
        const text = await response.text();
        const parsed = await parseCsv(text);
        setRows(parsed);
        setLastUpdated(new Date());
        setError("");
      } catch (err) {
        setError(err.message || "Failed to load data");
      } finally {
        setLoading(false);
      }
    };

    load();

    if (autoRefresh) {
      intervalId = setInterval(load, POLL_MS);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh]);

  const scenarios = useMemo(() => {
    const all = Array.from(new Set(rows.map((row) => row.scenario))).sort();
    return ["all", ...all];
  }, [rows]);

  const protocols = useMemo(
    () => Array.from(new Set(rows.map((row) => row.protocol))).sort(),
    [rows]
  );

  useEffect(() => {
    if (selectedProtocols.length === 0 && protocols.length > 0) {
      setSelectedProtocols(protocols);
    }
  }, [protocols, selectedProtocols.length]);

  const filtered = useMemo(() => {
    return rows.filter((row) => {
      const scenarioMatch = scenario === "all" || row.scenario === scenario;
      const protocolMatch =
        selectedProtocols.length === 0 ||
        selectedProtocols.includes(row.protocol);
      const searchMatch =
        search.trim() === "" ||
        row.scenario.toLowerCase().includes(search.toLowerCase()) ||
        row.protocol.toLowerCase().includes(search.toLowerCase());
      return scenarioMatch && protocolMatch && searchMatch;
    });
  }, [rows, scenario, selectedProtocols, search]);

  const chartRows = useMemo(
    () =>
      filtered.map((row) => ({
        ...row,
        label: `${humanize(row.scenario)} · ${humanize(row.protocol)}`,
      })),
    [filtered]
  );
  const summary = useMemo(() => {
    if (!filtered.length) {
      return {
        records: 0,
        scenarios: 0,
        protocols: 0,
        bestT90: null,
        lowestMessages: null,
      };
    }
    const bestT90 = filtered.reduce((best, row) => {
      if (row.t90_mean === null) return best;
      return best === null || row.t90_mean < best.t90_mean ? row : best;
    }, null);
    const lowestMessages = filtered.reduce((best, row) => {
      if (row.messages_mean === null) return best;
      return best === null || row.messages_mean < best.messages_mean
        ? row
        : best;
    }, null);
    return {
      records: filtered.length,
      scenarios: new Set(filtered.map((row) => row.scenario)).size,
      protocols: new Set(filtered.map((row) => row.protocol)).size,
      bestT90,
      lowestMessages,
    };
  }, [filtered]);

  const sortedRows = useMemo(() => {
    const key = sortKey;
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const aVal = a[key];
      const bVal = b[key];
      if (aVal === null || Number.isNaN(aVal)) return 1;
      if (bVal === null || Number.isNaN(bVal)) return -1;
      if (aVal === bVal) return 0;
      return aVal > bVal ? dir : -dir;
    });
  }, [filtered, sortKey, sortDir]);

  const toggleProtocol = (protocol) => {
    setSelectedProtocols((prev) =>
      prev.includes(protocol)
        ? prev.filter((item) => item !== protocol)
        : [...prev, protocol]
    );
  };

  const openDemo = (kind, key) => {
    setDemoKind(kind);
    setDemoKey(key);
    setDemoPlaying(true);
    setDemoTime(0);
  };

  const closeDemo = () => {
    setDemoKey(null);
    setDemoKind(null);
  };

  const demo = demoKey ? buildDemo(demoKind, demoKey) : null;
  const demoSteps = demo?.steps?.length ? demo.steps : [];
  const demoDuration = demoSteps.length ? demoSteps[demoSteps.length - 1].t + 1.5 : 0;
  const effectiveTime = demoDuration ? demoTime % demoDuration : 0;
  const demoProgress = demoDuration ? effectiveTime / demoDuration : 0;
  const currentStepIndex = demoSteps.length
    ? demoSteps.findIndex((step, idx) => {
        const next = demoSteps[idx + 1];
        return effectiveTime < (next ? next.t : demoDuration);
      })
    : 0;
  const currentStep = demoSteps.length
    ? demoSteps[Math.max(0, currentStepIndex)]
    : null;

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Block Propagation • Visual Lab</p>
          <h1>Propagation Pulse</h1>
          <p className="subtitle">
            A vivid live dashboard for propagation behavior, compact blocks, and
            network bottlenecks.
          </p>
        </div>
        <div className="hero-card">
          <div className="status">
            <span className={autoRefresh ? "live" : "paused"} />
            {autoRefresh ? "Live sync" : "Paused"}
          </div>
          <div className="meta">
            <div>
              <span>Last refresh</span>
              <strong>
                {lastUpdated
                  ? lastUpdated.toLocaleTimeString()
                  : "Waiting..."}
              </strong>
            </div>
            <div>
              <span>Source</span>
              <strong>outputs/all_tests_summary.csv</strong>
            </div>
          </div>
          <div className="toggle-row">
            <button
              className="ghost"
              onClick={() => setAutoRefresh((prev) => !prev)}
            >
              {autoRefresh ? "Pause refresh" : "Resume refresh"}
            </button>
            <button
              className="ghost"
              onClick={() =>
                setTheme((prev) => (prev === "nebula" ? "solar" : "nebula"))
              }
            >
              Theme: {theme === "nebula" ? "Nebula" : "Solar"}
            </button>
          </div>
        </div>
      </header>

      <section className="controls">
        <div className="control">
          <label>Scenario</label>
          <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
            {scenarios.map((item) => (
              <option key={item} value={item}>
                {item === "all" ? "All scenarios" : humanize(item)}
              </option>
            ))}
          </select>
        </div>
        <div className="control search">
          <label>Search</label>
          <input
            type="search"
            placeholder="Filter by scenario or protocol"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <div className="control">
          <label>Protocols</label>
          <div className="protocols">
            {protocols.map((protocol) => (
              <label key={protocol} className="pill">
                <input
                  type="checkbox"
                  checked={selectedProtocols.includes(protocol)}
                  onChange={() => toggleProtocol(protocol)}
                />
                <span>{humanize(protocol)}</span>
              </label>
            ))}
          </div>
        </div>
      </section>

      <section className="summary">
        <div className="stat-card">
          <span>Records</span>
          <strong>{summary.records}</strong>
        </div>
        <div className="stat-card">
          <span>Scenarios</span>
          <strong>{summary.scenarios}</strong>
        </div>
        <div className="stat-card">
          <span>Protocols</span>
          <strong>{summary.protocols}</strong>
        </div>
        <div className="stat-card">
          <span>Fastest T90</span>
          <strong>
            {summary.bestT90
              ? `${humanize(summary.bestT90.protocol)} • ${formatNumber(
                  summary.bestT90.t90_mean,
                  2
                )}s`
              : "—"}
          </strong>
        </div>
        <div className="stat-card">
          <span>Lowest Messages</span>
          <strong>
            {summary.lowestMessages
              ? `${humanize(summary.lowestMessages.protocol)} • ${formatNumber(
                  summary.lowestMessages.messages_mean,
                  0
                )}`
              : "—"}
          </strong>
        </div>
      </section>

      {error && <div className="banner error">{error}</div>}
      {loading && <div className="banner">Loading data…</div>}

      <section className="layout">
        <aside className="side-panel left">
          <div className="panel sticky-panel">
            <div className="panel-header">
              <h3>Protocols</h3>
              <p>Quick meaning for each protocol label</p>
            </div>
            <div className="explainer-grid">
              {Object.entries(protocolInfo).map(([key, text]) => (
                <button
                  key={key}
                  type="button"
                  className="explainer-card clickable"
                  onClick={() => openDemo("protocol", key)}
                >
                  <span>{humanize(key)}</span>
                  <p>{text}</p>
                </button>
              ))}
            </div>
          </div>
        </aside>

        <div className="main-content">
          <section className="charts">
            <div className="panel">
              <div className="panel-header">
                <h3>Latency Curve</h3>
                <p>T50, T90, T100 by protocol</p>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis
                    dataKey="label"
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                    height={60}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => `${formatNumber(value, 3)}s`} />
                  <Legend />
                  <Line type="monotone" dataKey="t50_mean" stroke="var(--accent)" />
                  <Line type="monotone" dataKey="t90_mean" stroke="var(--accent-2)" />
                  <Line type="monotone" dataKey="t100_mean" stroke="var(--accent-3)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <div className="panel-header">
                <h3>Bandwidth Cost</h3>
                <p>Message volume per protocol</p>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis
                    dataKey="label"
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                    height={60}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => formatNumber(value, 0)} />
                  <Bar
                    dataKey="messages_mean"
                    fill="var(--accent)"
                    radius={[6, 6, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <div className="panel-header">
                <h3>Latency vs. Messages</h3>
                <p>Tradeoff surface for protocols</p>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="messages_mean" type="number" name="Messages" />
                  <YAxis dataKey="t90_mean" type="number" name="T90 (s)" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter data={chartRows} fill="var(--accent-2)" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="panel">
              <div className="panel-header">
                <h3>Security Margin</h3>
                <p>Propagation safety buffer by protocol</p>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis
                    dataKey="label"
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                    height={60}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => formatNumber(value, 3)} />
                  <Area
                    type="monotone"
                    dataKey="security_margin_t50_mean"
                    stroke="var(--accent-3)"
                    fill="var(--accent-3-soft)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="table-wrap">
            <div className="panel-header">
              <h3>Scenario Table</h3>
              <p>All metrics (filtered)</p>
            </div>
            <div className="table-controls">
              <label>
                Sort by
                <select value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
                  <option value="messages_mean">Messages</option>
                  <option value="t50_mean">T50</option>
                  <option value="t90_mean">T90</option>
                  <option value="t100_mean">T100</option>
                  <option value="security_margin_t50_mean">Security</option>
                </select>
              </label>
              <label>
                Order
                <select value={sortDir} onChange={(event) => setSortDir(event.target.value)}>
                  <option value="asc">Ascending</option>
                  <option value="desc">Descending</option>
                </select>
              </label>
            </div>
            <div className="table">
              <div className="table-row table-head">
                <span>Scenario</span>
                <span>Protocol</span>
                <span>T50</span>
                <span>T90</span>
                <span>T100</span>
                <span>Messages</span>
                <span>Security</span>
              </div>
              {sortedRows.map((row) => (
                <div key={`${row.scenario}-${row.protocol}`} className="table-row">
                  <span>{humanize(row.scenario)}</span>
                  <span>{humanize(row.protocol)}</span>
                  <span>{formatNumber(row.t50_mean, 3)}s</span>
                  <span>{formatNumber(row.t90_mean, 3)}s</span>
                  <span>{formatNumber(row.t100_mean, 3)}s</span>
                  <span>{formatNumber(row.messages_mean, 0)}</span>
                  <span>{formatNumber(row.security_margin_t50_mean, 3)}</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        <aside className="side-panel right">
          <div className="panel sticky-panel">
            <div className="panel-header">
              <h3>Scenarios</h3>
              <p>What each scenario name represents</p>
            </div>
            <div className="explainer-grid">
              {Object.entries(scenarioInfo).map(([key, text]) => (
                <button
                  key={key}
                  type="button"
                  className="explainer-card clickable"
                  onClick={() => openDemo("scenario", key)}
                >
                  <span>{humanize(key)}</span>
                  <p>{text}</p>
                </button>
              ))}
            </div>
          </div>
        </aside>
      </section>

      {demo && (
        <div className="demo-overlay">
          <div className="demo-panel">
            <div className="demo-header">
              <div>
                <p className="eyebrow">Interactive demo</p>
                <h2>{demo.title}</h2>
                <p className="subtitle">{demo.summary}</p>
              </div>
              <button className="ghost" onClick={closeDemo} type="button">
                Close
              </button>
            </div>
            <div className="demo-body">
              <svg viewBox="0 0 480 300" className="demo-graph">
                {graphEdges.map(([a, b]) => {
                  const na = findNode(a);
                  const nb = findNode(b);
                  const active =
                    currentStep &&
                    currentStep.nodes.includes(a) &&
                    currentStep.nodes.includes(b);
                  return (
                    <line
                      key={`${a}-${b}`}
                      x1={na.x}
                      y1={na.y}
                      x2={nb.x}
                      y2={nb.y}
                      className={active ? "edge active" : "edge"}
                    />
                  );
                })}
                {graphNodes.map((node) => {
                  const active = currentStep?.nodes.includes(node.id);
                  return (
                    <g key={node.id}>
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={active ? 14 : 11}
                        className={active ? "node active" : "node"}
                      />
                      <text x={node.x} y={node.y + 4} textAnchor="middle">
                        {node.id}
                      </text>
                    </g>
                  );
                })}
                {currentStep?.badge && (
                  <text x="20" y="280" className="badge">
                    {currentStep.badge}
                  </text>
                )}
              </svg>
              <div className="demo-caption">
                <strong>{currentStep?.caption}</strong>
                <div className="progress">
                  <span style={{ width: `${demoProgress * 100}%` }} />
                </div>
              </div>
            </div>
            <div className="demo-controls">
              <button
                className="ghost"
                type="button"
                onClick={() => setDemoPlaying((prev) => !prev)}
              >
                {demoPlaying ? "Pause" : "Play"}
              </button>
              <button
                className="ghost"
                type="button"
                onClick={() => {
                  setDemoTime(0);
                  setDemoPlaying(true);
                }}
              >
                Restart
              </button>
              <label>
                Speed
                <select
                  value={demoSpeed}
                  onChange={(event) => setDemoSpeed(Number(event.target.value))}
                >
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
