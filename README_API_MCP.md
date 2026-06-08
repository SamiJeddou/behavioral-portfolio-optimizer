# Beyond Mean-Variance — Callable Engines (REST API + MCP)

The optimisation engines are UI-free in `core/`. Two thin adapters expose them to the
outside world. Neither contains any optimisation logic — they translate requests into
typed calls on `core.optimise` and return JSON.

```
core/            UI-free engines (pricing, grid, scenario, backtest, markets) + typed API
  optimise.py    optimise_scenario() / scenario_frontier() / optimise_grid()
api/main.py      FastAPI REST adapter
mcp_server.py    MCP server (AI-agent tools)
```

## REST API

```bash
pip install fastapi "uvicorn[standard]"
uvicorn api.main:app --reload          # docs at http://127.0.0.1:8000/docs
```

Optional auth: set `BMV_API_KEY=somesecret` in the environment to require an
`x-api-key` header on every call. Leave unset for an open demo.

Example — maximise return subject to a -20% CVaR floor, with a protective put on asset 0:

```bash
curl -s http://127.0.0.1:8000/optimise/scenario -H "Content-Type: application/json" -d '{
  "universe": {"names":["AAPL","MSFT","GLD","TLT"],
               "means":[0.12,0.10,0.05,0.03],
               "sigmas":[0.28,0.24,0.15,0.10],
               "corr":[[1,0.5,0.1,-0.2],[0.5,1,0.05,-0.1],[0.1,0.05,1,0],[-0.2,-0.1,0,1]]},
  "constraint": {"kind":"es_thesis","H":-0.15,"alpha":0.05,"L":-0.20},
  "derivatives": [{"type":"put","underlying_idx":0,"params":{"strike":0.9},"label":"AAPL put 0.90"}],
  "scenarios": 10000, "w_max": 0.5
}'
```

Returns:

```json
{"labels":["AAPL","MSFT","GLD","TLT","AAPL put 0.90"],
 "weights":[0.50,0.34,0.14,0.0,0.02],
 "expected_return":0.0896,"shortfall_stat":-0.20,"std_dev":0.171,"feasible":true,"meta":{...}}
```

Endpoints: `POST /optimise/scenario`, `POST /optimise/frontier`, `POST /optimise/grid`,
`GET /health`. Full request/response schemas are auto-documented at `/docs`.

## MCP server (for AI agents)

```bash
pip install mcp
python mcp_server.py
```

Add to an MCP client (e.g. Claude Desktop) config:

```json
{ "mcpServers": {
    "beyond-mean-variance": {
      "command": "python",
      "args": ["/abs/path/to/mcp_server.py"]
    } } }
```

Tools exposed: `optimise_scenario_tool`, `trace_frontier_tool`. The client can then call
them in natural language ("build a portfolio of AAPL/MSFT/GLD with a -20% tail floor").

## Notes

- The **scenario** engine is the one exposed for scale (cost is linear in assets).
  The exact **grid** engine is available via `/optimise/grid` but is `m^N` — use it for
  small/exact cases.
- The CVaR LP is ~10s per solve; fine for batch / agent / notebook use. For a busy public
  REST endpoint, run it behind a job queue or make the route async.
- A vendor data feed plugs in via `core.markets.DataSource` — implement `.prices()` and
  pass it to `universe_from_prices()`; nothing else changes.
- Research and educational project. Not investment advice.
