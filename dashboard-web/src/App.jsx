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
const SIM_URL = "http://localhost:8000/simulate";
const POLL_MS = 300000;
const BASE_BLOCK_SIZE = 1_000_000;
const BASE_BANDWIDTH = 10;
const BASE_LATENCY = 0.125;
const numberFields = [
  "t50_mean",
  "t90_mean",
  "t100_mean",
  "messages_mean",
  "compete_p_t90_mean",
  "lambda_t100_mean",
  "p_ge1_t100_mean",
  "p_ge2_t100_mean",
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
  if (value === null || Number.isNaN(value)) return "--";
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
  "macro_metrics": "Baseline two-phase run used to compute macro metrics.",
};

const shuffleWithSeed = (items, seed) => {
  const result = [...items];
  let state = seed || 1;
  for (let i = result.length - 1; i > 0; i -= 1) {
    state = (state * 1664525 + 1013904223) % 4294967296;
    const j = Math.floor((state / 4294967296) * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
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
  const [compareBase, setCompareBase] = useState("");
  const [compareTarget, setCompareTarget] = useState("");
  const [compareMode, setCompareMode] = useState("absolute");
  const [playScenario, setPlayScenario] = useState("");
  const [playMode, setPlayMode] = useState("preset");
  const [playCustomProtocol, setPlayCustomProtocol] = useState("bitcoin-compact");
  const [playCustomTopology, setPlayCustomTopology] = useState("random-regular");
  const [playScaleFreeM, setPlayScaleFreeM] = useState(3);
  const [playBlockSize, setPlayBlockSize] = useState(BASE_BLOCK_SIZE);
  const [playBandwidth, setPlayBandwidth] = useState(BASE_BANDWIDTH);
  const [playLatency, setPlayLatency] = useState(BASE_LATENCY);
  const [playOverlap, setPlayOverlap] = useState(0.9);
  const [playDisruption, setPlayDisruption] = useState(0);
  const [liveResult, setLiveResult] = useState(null);
  const [liveError, setLiveError] = useState("");
  const [liveLoading, setLiveLoading] = useState(false);
  const [chartOrder, setChartOrder] = useState("sorted");
  const [randomSeed, setRandomSeed] = useState(1);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

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

  const scenarioOptions = useMemo(
    () => Array.from(new Set(rows.map((row) => row.scenario))).sort(),
    [rows]
  );

  const protocols = useMemo(
    () => Array.from(new Set(rows.map((row) => row.protocol))).sort(),
    [rows]
  );

  useEffect(() => {
    if (selectedProtocols.length === 0 && protocols.length > 0) {
      setSelectedProtocols(protocols);
    }
  }, [protocols, selectedProtocols.length]);

  useEffect(() => {
    if (!rows.length) return;
    if (playMode === "preset" && !playScenario) {
      setPlayScenario(rows[0].scenario);
    }
  }, [rows, playScenario, playMode]);

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
    () => {
      const rows = filtered.map((row) => ({
        ...row,
        label: `${humanize(row.scenario)} - ${humanize(row.protocol)}`,
      }));
      if (chartOrder === "random") {
        return shuffleWithSeed(rows, randomSeed);
      }
      if (chartOrder === "sorted") {
        return [...rows].sort((a, b) => {
          if (a.t90_mean === null) return 1;
          if (b.t90_mean === null) return -1;
          return a.t90_mean - b.t90_mean;
        });
      }
      return rows;
    },
    [filtered, chartOrder, randomSeed]
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

  const comparisonOptions = useMemo(
    () =>
      filtered.map((row) => ({
        key: `${row.scenario}||${row.protocol}`,
        label: `${humanize(row.scenario)} - ${humanize(row.protocol)}`,
        row,
      })),
    [filtered]
  );

  useEffect(() => {
    if (!comparisonOptions.length) return;
    if (!comparisonOptions.some((opt) => opt.key === compareBase)) {
      setCompareBase(comparisonOptions[0].key);
    }
    if (!comparisonOptions.some((opt) => opt.key === compareTarget)) {
      const fallback =
        comparisonOptions.length > 1 ? comparisonOptions[1].key : comparisonOptions[0].key;
      setCompareTarget(fallback);
    }
  }, [comparisonOptions, compareBase, compareTarget]);

  const compareMetrics = [
    { key: "t50_mean", label: "T50", digits: 3 },
    { key: "t90_mean", label: "T90", digits: 3 },
    { key: "t100_mean", label: "T100", digits: 3 },
    { key: "messages_mean", label: "Messages", digits: 0 },
  ];

  const comparison = useMemo(() => {
    const base = comparisonOptions.find((opt) => opt.key === compareBase)?.row;
    const target = comparisonOptions.find((opt) => opt.key === compareTarget)?.row;
    if (!base || !target) return null;
    const rows = compareMetrics.map((metric) => {
      const baseVal = base[metric.key];
      const targetVal = target[metric.key];
      let delta = null;
      if (baseVal !== null && targetVal !== null) {
        if (compareMode === "percent") {
          delta = baseVal === 0 ? null : ((targetVal - baseVal) / baseVal) * 100;
        } else {
          delta = targetVal - baseVal;
        }
      }
      return {
        metric: metric.label,
        delta,
        baseVal,
        targetVal,
        digits: metric.digits,
      };
    });
    return { base, target, rows };
  }, [compareBase, compareTarget, compareMode, comparisonOptions]);

  const formatDelta = (value, digits, mode) => {
    if (value === null || Number.isNaN(value)) return "--";
    const sign = value > 0 ? "+" : "";
    const suffix = mode === "percent" ? "%" : "";
    return `${sign}${value.toFixed(digits)}${suffix}`;
  };

  const renderScatterTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const data = payload[0].payload;
    return (
      <div className="tooltip-card">
        <strong>{humanize(data.scenario)} - {humanize(data.protocol)}</strong>
        <div>T90: {formatNumber(data.t90_mean, 3)}s</div>
        <div>Messages: {formatNumber(data.messages_mean, 0)}</div>
      </div>
    );
  };

  const playgroundBase = useMemo(() => {
    if (playMode !== "preset") {
      return null;
    }
    return rows.find((row) => row.scenario === playScenario);
  }, [rows, playScenario, playMode]);

  const playgroundApprox = useMemo(() => {
    if (!playgroundBase) return null;
    const protocol = playgroundBase.protocol;
    const latencyFactor = playLatency / BASE_LATENCY;
    const bandwidthFactor = BASE_BANDWIDTH / playBandwidth;
    const sizeFactor = playBlockSize / BASE_BLOCK_SIZE;
    const disruptionFactor = 1 + playDisruption * 2;
    const compactFactor =
      protocol === "bitcoin-compact" ? 1 - playOverlap * 0.2 : 1;
    const timeFactor =
      (0.5 * latencyFactor + 0.3 * bandwidthFactor + 0.2 * sizeFactor) *
      disruptionFactor *
      compactFactor;
    const messageFactor = 1 + playDisruption * 0.6;
    return {
      t50: playgroundBase.t50_mean * timeFactor,
      t90: playgroundBase.t90_mean * timeFactor,
      t100: playgroundBase.t100_mean * timeFactor,
      messages: playgroundBase.messages_mean * messageFactor,
    };
  }, [
    playgroundBase,
    playLatency,
    playBandwidth,
    playBlockSize,
    playDisruption,
    playOverlap,
  ]);

  const runLiveSim = async () => {
    setLiveLoading(true);
    setLiveError("");
    try {
      const isCustom = playMode === "custom";
      const protocol = isCustom
        ? playCustomProtocol
        : playgroundBase?.protocol ?? "two-phase";
      const scenarioName = isCustom ? "custom" : playScenario;
      const requestPayload = {
        protocol,
        scenario: scenarioName,
        runs: 3,
        seed: 42,
        block_size_bytes: Math.round(playBlockSize),
        bandwidth_mbps: playBandwidth,
        latency_min: playLatency,
        latency_max: playLatency,
        mempool_overlap_mean: playOverlap,
        mempool_overlap_std: 0.05,
        churn_prob: playDisruption,
        delay_prob: playDisruption,
        delay_latency_mult: 2.0,
        delay_bandwidth_mult: 0.7,
      };
      if (isCustom) {
        requestPayload.topology = playCustomTopology;
        if (playCustomTopology === "scale-free") {
          requestPayload.scale_free_m = playScaleFreeM;
        }
      }
      const response = await fetch(SIM_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const responsePayload = await response.json();
      setLiveResult(responsePayload.summary);
    } catch (err) {
      setLiveError(err.message || "Failed to run simulation.");
    } finally {
      setLiveLoading(false);
    }
  };

  const resetPlayground = () => {
    setPlayBlockSize(BASE_BLOCK_SIZE);
    setPlayBandwidth(BASE_BANDWIDTH);
    setPlayLatency(BASE_LATENCY);
    setPlayOverlap(0.9);
    setPlayDisruption(0);
    setLiveResult(null);
    setLiveError("");
  };

  const toggleProtocol = (protocol) => {
    setSelectedProtocols((prev) =>
      prev.includes(protocol)
        ? prev.filter((item) => item !== protocol)
        : [...prev, protocol]
    );
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Block Propagation - Visual Lab</p>
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
              ? `${humanize(summary.bestT90.protocol)} - ${formatNumber(
                  summary.bestT90.t90_mean,
                  2
                )}s`
              : "--"}
          </strong>
        </div>
        <div className="stat-card">
          <span>Lowest Messages</span>
          <strong>
            {summary.lowestMessages
              ? `${humanize(summary.lowestMessages.protocol)} - ${formatNumber(
                  summary.lowestMessages.messages_mean,
                  0
                )}`
              : "--"}
          </strong>
        </div>
      </section>

      {error && <div className="banner error">{error}</div>}
      {loading && <div className="banner">Loading data...</div>}

      <section className="layout">
        <aside className="side-panel left">
          <div className="panel sticky-panel">
            <div className="panel-header">
              <h3>Protocols</h3>
              <p>Quick meaning for each protocol label</p>
            </div>
            <div className="explainer-grid">
              {Object.entries(protocolInfo).map(([key, text]) => (
                <div key={key} className="explainer-card">
                  <span>{humanize(key)}</span>
                  <p>{text}</p>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <div className="main-content">
          <section className="playground panel">
            <div className="panel-header">
              <h3>Playground</h3>
              <p>Adjust one scenario and compare approximate vs live results</p>
            </div>
            <div className="playground-controls">
              <label>
                Mode
                <select
                  value={playMode}
                  onChange={(event) => setPlayMode(event.target.value)}
                >
                  <option value="preset">Preset scenario</option>
                  <option value="custom">Custom config</option>
                </select>
              </label>
              <label>
                Scenario
                <select
                  value={playScenario}
                  onChange={(event) => setPlayScenario(event.target.value)}
                  disabled={playMode !== "preset"}
                >
                  {scenarioOptions.map((value) => (
                    <option key={value} value={value}>
                      {humanize(value)}
                    </option>
                  ))}
                </select>
              </label>
              {playMode === "custom" && (
                <>
                  <label>
                    Protocol
                    <select
                      value={playCustomProtocol}
                      onChange={(event) =>
                        setPlayCustomProtocol(event.target.value)
                      }
                    >
                      {protocols.map((value) => (
                        <option key={value} value={value}>
                          {humanize(value)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Topology
                    <select
                      value={playCustomTopology}
                      onChange={(event) =>
                        setPlayCustomTopology(event.target.value)
                      }
                    >
                      <option value="random-regular">Random regular</option>
                      <option value="scale-free">Scale-free</option>
                      <option value="small-world">Small-world</option>
                      <option value="star">Star</option>
                      <option value="line">Line</option>
                    </select>
                  </label>
                  {playCustomTopology === "scale-free" && (
                    <label>
                      Scale-free m
                      <input
                        type="number"
                        min="1"
                        max="50"
                        step="1"
                        value={playScaleFreeM}
                        onChange={(event) =>
                          setPlayScaleFreeM(Number(event.target.value))
                        }
                      />
                    </label>
                  )}
                </>
              )}
              <label>
                Block size (MB)
                <input
                  type="range"
                  min="0.5"
                  max="4"
                  step="0.1"
                  value={(playBlockSize / 1_000_000).toFixed(1)}
                  onChange={(event) =>
                    setPlayBlockSize(Number(event.target.value) * 1_000_000)
                  }
                />
                <span>{(playBlockSize / 1_000_000).toFixed(1)} MB</span>
              </label>
              <label>
                Bandwidth (Mbps)
                <input
                  type="range"
                  min="2"
                  max="50"
                  step="1"
                  value={playBandwidth}
                  onChange={(event) => setPlayBandwidth(Number(event.target.value))}
                />
                <span>{playBandwidth} Mbps</span>
              </label>
              <label>
                Latency (s)
                <input
                  type="range"
                  min="0.02"
                  max="0.4"
                  step="0.01"
                  value={playLatency.toFixed(2)}
                  onChange={(event) => setPlayLatency(Number(event.target.value))}
                />
                <span>{playLatency.toFixed(2)} s</span>
              </label>
              <label>
                Mempool overlap
                <input
                  type="range"
                  min="0.5"
                  max="1"
                  step="0.02"
                  value={playOverlap.toFixed(2)}
                  onChange={(event) => setPlayOverlap(Number(event.target.value))}
                />
                <span>{playOverlap.toFixed(2)}</span>
              </label>
              <label>
                Churn / delay
                <input
                  type="range"
                  min="0"
                  max="0.3"
                  step="0.02"
                  value={playDisruption.toFixed(2)}
                  onChange={(event) => setPlayDisruption(Number(event.target.value))}
                />
                <span>{playDisruption.toFixed(2)}</span>
              </label>
            </div>
            <div className="playground-results">
              <div className="playground-card">
                <h4>Approx</h4>
                {playgroundApprox ? (
                  <>
                    <div>T50: {formatNumber(playgroundApprox.t50, 3)}s</div>
                    <div>T90: {formatNumber(playgroundApprox.t90, 3)}s</div>
                    <div>T100: {formatNumber(playgroundApprox.t100, 3)}s</div>
                    <div>Messages: {formatNumber(playgroundApprox.messages, 0)}</div>
                  </>
                ) : (
                  <div>
                    {playMode === "custom"
                      ? "Approx is disabled for custom configs."
                      : "Pick a scenario to start."}
                  </div>
                )}
              </div>
              <div className="playground-card">
                <h4>Live simulation</h4>
                {liveResult ? (
                  <>
                    <div>T50: {formatNumber(liveResult.t50, 3)}s</div>
                    <div>T90: {formatNumber(liveResult.t90, 3)}s</div>
                    <div>T100: {formatNumber(liveResult.t100, 3)}s</div>
                    <div>Messages: {formatNumber(liveResult.messages, 0)}</div>
                  </>
                ) : (
                  <div>Run the live simulation.</div>
                )}
                {liveError && <div className="banner error">{liveError}</div>}
                <button
                  className="ghost"
                  type="button"
                  onClick={runLiveSim}
                  disabled={liveLoading}
                >
                  {liveLoading ? "Running..." : "Run exact simulation"}
                </button>
                <button
                  className="ghost"
                  type="button"
                  onClick={resetPlayground}
                >
                  Reset playground
                </button>
                <p className="hint">Server: python dashboard-web/sim_server.py</p>
              </div>
            </div>
          </section>

          <section className="comparison panel">
            <div className="panel-header">
              <h3>Comparisons</h3>
              <p>Compare two scenarios or protocols side by side</p>
            </div>
            <div className="comparison-controls">
              <label>
                Baseline
                <select value={compareBase} onChange={(event) => setCompareBase(event.target.value)}>
                  {comparisonOptions.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Compare to
                <select
                  value={compareTarget}
                  onChange={(event) => setCompareTarget(event.target.value)}
                >
                  {comparisonOptions.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Mode
                <select value={compareMode} onChange={(event) => setCompareMode(event.target.value)}>
                  <option value="absolute">Absolute delta</option>
                  <option value="percent">Percent change</option>
                </select>
              </label>
            </div>
            {comparison ? (
              <div className="comparison-grid">
                <div className="comparison-metrics">
                  {comparison.rows.map((row) => (
                    <div key={row.metric} className="comparison-row">
                      <span>{row.metric}</span>
                      <strong>
                        {formatDelta(row.delta, row.digits, compareMode)}
                      </strong>
                      <small>
                        {formatNumber(row.baseVal, row.digits)}{" -> "}{formatNumber(row.targetVal, row.digits)}
                      </small>
                    </div>
                  ))}
                </div>
                <div className="comparison-chart">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={comparison.rows}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="metric" />
                      <YAxis />
                      <Tooltip
                        formatter={(value, name, props) =>
                          formatDelta(value, props.payload.digits, compareMode)
                        }
                      />
                      <Bar dataKey="delta" fill="var(--accent-2)" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="banner">Select two items to compare.</div>
            )}
          </section>

          <section className="charts">
            <div className="charts-controls">
              <label>
                Chart order
                <select
                  value={chartOrder}
                  onChange={(event) => {
                    const value = event.target.value;
                    setChartOrder(value);
                    if (value === "random") {
                      setRandomSeed((prev) => prev + 1);
                    }
                  }}
                >
                  <option value="sorted">Sorted by T90</option>
                  <option value="random">Random</option>
                  <option value="original">Original order</option>
                </select>
              </label>
            </div>
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
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} content={renderScatterTooltip} />
                  <Scatter data={chartRows} fill="var(--accent-2)" />
                </ScatterChart>
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
            </div>
            {sortedRows.map((row) => (
              <div key={`${row.scenario}-${row.protocol}`} className="table-row">
                <span>{humanize(row.scenario)}</span>
                <span>{humanize(row.protocol)}</span>
                <span>{formatNumber(row.t50_mean, 3)}s</span>
                <span>{formatNumber(row.t90_mean, 3)}s</span>
                <span>{formatNumber(row.t100_mean, 3)}s</span>
                <span>{formatNumber(row.messages_mean, 0)}</span>
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
                <div key={key} className="explainer-card">
                  <span>{humanize(key)}</span>
                  <p>{text}</p>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </section>
    </div>
  );
}

export default App;
