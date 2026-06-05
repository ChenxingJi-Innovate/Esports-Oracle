// Vercel serverless function (.mjs = ESM): the "Explain" reasoning agent.
//
// Given a match + the statistical model's probability, it:
//   1. re-examines whether the base probability is sensible vs the feature
//      contributions (it is told the model's equation),
//   2. can call web_search (tool-calling) for late-breaking info the stats
//      can't see (roster/stand-in/injury/patch/very recent results),
//   3. returns KEEP or ADJUST + a concise reasoning.
//
// The DeepSeek key lives ONLY here (server-side, from env). The user picks the
// model (deepseek-v4-flash | deepseek-v4-pro) per request.

export const config = { maxDuration: 60 };

const MODELS = new Set(["deepseek-v4-flash", "deepseek-v4-pro"]);
const DS_URL = "https://api.deepseek.com/chat/completions";

const SYSTEM = `You are the reasoning layer of a CS2/LoL tier-1 match predictor.
The base model is a logistic: p_map = sigmoid(sum w_i * z(feature_i)).
Features: rating_diff (from world rank), form_diff, map_diff, player_diff, h2h_diff.
Approx weights: rating .35, form .22, map .17, player .17, h2h .11 (damped, anchored to a prior, self-trained daily on graded results). Series probability comes from best-of math.

For the given match:
1. Sanity-check the base probability against the feature contributions. Is it over/under-confident?
2. Use web_search (max 2 calls) for anything the stats can't see: roster changes, stand-ins, injuries, patch shifts, results in the last few days. Skip searching if the matchup is clearly decided by the stats.
3. Decide KEEP or ADJUST, then explain in <=140 words, plain and specific.

Reply with your FINAL answer as a single JSON object only:
{"verdict":"keep"|"adjust","adjusted_p_a":<number 0-1 or null>,"confidence":"low|medium|high","reasoning":"..."}`;

const TOOLS = [{
  type: "function",
  function: {
    name: "web_search",
    description: "Search the web for recent esports info the statistical model cannot see (roster/stand-in/injury/patch/very recent results).",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "search query" } },
      required: ["query"],
    },
  },
}];

async function dsCall(key, model, messages) {
  const r = await fetch(DS_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, tools: TOOLS, tool_choice: "auto",
                           temperature: 0.3, max_tokens: 3000 }),
  });
  if (!r.ok) throw new Error(`deepseek ${r.status}: ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

// keyless best-effort search (DuckDuckGo Instant Answer). Swap for a stronger
// provider later; the tool-calling contract stays identical.
async function webSearch(q) {
  try {
    const u = `https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_html=1&t=esports-oracle`;
    const j = await (await fetch(u)).json();
    const parts = [j.AbstractText, j.Answer];
    (j.RelatedTopics || []).slice(0, 6).forEach((t) => t && t.Text && parts.push(t.Text));
    return parts.filter(Boolean).join(" | ").slice(0, 1500) || "(no results)";
  } catch (e) { return `(search failed: ${e.message})`; }
}

function extractJson(text) {
  if (!text) return null;
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
}

// Core agent loop. Exported so it can be tested without the HTTP wrapper.
export async function reason({ match, model, key, today }) {
  const m = match || {};
  const chosen = MODELS.has(model) ? model : "deepseek-v4-flash";
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content:
      `Match: ${m.team_a} vs ${m.team_b} (${m.game || "?"}, ${m.event || "?"}, ${m.fmt || "BO3"}).
Base p(${m.team_a} wins series) = ${m.p_a}. Per-map = ${m.p_map_a ?? "n/a"}.
Feature diffs (A minus B): ${JSON.stringify(m.features || {})}.
Today is ${today || new Date().toISOString().slice(0, 10)}.` },
  ];

  const searched = [];
  for (let round = 0; round < 4; round++) {
    const data = await dsCall(key, chosen, messages);
    const msg = data.choices?.[0]?.message;
    if (!msg) throw new Error("empty deepseek response");
    messages.push(msg);

    const calls = msg.tool_calls || [];
    if (calls.length === 0) {
      const text = msg.content || msg.reasoning_content || "";
      const parsed = extractJson(text) || { verdict: "keep", adjusted_p_a: null, confidence: "low", reasoning: text.slice(0, 800) };
      return { model: chosen, searched, ...parsed };
    }
    for (const c of calls) {
      let args = {}; try { args = JSON.parse(c.function.arguments || "{}"); } catch {}
      const result = await webSearch(args.query || "");
      searched.push({ query: args.query || "", got: result.slice(0, 160) });
      messages.push({ role: "tool", tool_call_id: c.id, content: result });
    }
  }
  return { model: chosen, searched, verdict: "keep", adjusted_p_a: null,
           confidence: "low", reasoning: "Reasoning loop did not converge within the round budget." };
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const key = process.env.DEEPSEEK_API_KEY;
  if (!key) return res.status(500).json({ error: "DEEPSEEK_API_KEY not configured on the server" });

  let body = {};
  try { body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {}); } catch {}
  const m = body.match || {};
  if (!m.team_a || !m.team_b) return res.status(400).json({ error: "match.team_a and match.team_b required" });

  try {
    const out = await reason({ match: m, model: body.model, key });
    return res.status(200).json(out);
  } catch (e) {
    return res.status(502).json({ error: String(e.message || e) });
  }
}
