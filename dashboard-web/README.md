# Block Propagation Dashboard (React)

This dashboard visualizes `outputs/all_tests_summary.csv` and includes:
- protocol/scenario explainers with demos,
- comparison panel,
- playground controls with optional live simulation.

## Setup

```bash
npm install
```

## Run the dashboard

```bash
npm run dev
```

This watches the CSV file and mirrors it into `public/data/` for live updates.

## Optional: live simulation server

```bash
npm run sim
```

The server runs at `http://localhost:8000` and powers the Playground "Run exact simulation" button.
